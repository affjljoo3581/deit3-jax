# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from collections import defaultdict
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import webdataset as wds
from chex import Array, ArrayTree
from jax.tree_util import DictKey


class AverageMeter:
    def __init__(self, use_latest: list[str] = []):
        self.buffer = defaultdict(list)
        self.use_latest = use_latest

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def summary(self, prefix: str = "") -> dict[str, float]:
        buffer = {k: np.array(v) for k, v in self.buffer.items()}
        self.buffer.clear()

        return {
            f"{prefix}{k}": v[-1] if k in self.use_latest else np.mean(v)
            for k, v in buffer.items()
        }


def save_checkpoint_in_background(
    args: argparse.Namespace, params_bytes: bytes, postfix: str = "last"
):
    def thread_fn():
        filename = os.path.join(args.output_dir, f"{args.name}-{postfix}.msgpack")
        with wds.gopen(filename, "wb") as fp:
            fp.write(params_bytes)

    threading.Thread(target=thread_fn).start()


class Mixup(nn.Module):
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0

    def apply_mixup(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.mixup_alpha,) * 2)
        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = ratio * images + (1 - ratio) * images[randperm]
        labels = ratio * labels + (1 - ratio) * labels[randperm]
        return images, labels

    def apply_cutmix(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.cutmix_alpha,) * 2)
        image_mask = self.random_bounding_box(ratio, images.shape[2], images.shape[1])
        label_mask = image_mask.mean((1, 2))

        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = image_mask * images + (1 - image_mask) * images[randperm]
        labels = label_mask * labels + (1 - label_mask) * labels[randperm]
        return images, labels

    def random_bounding_box(self, ratio: Array, width: int, height: int) -> Array:
        size = (1 - ratio) ** 0.5
        xstart, ystart = jax.random.uniform(self.make_rng("mixup"), (2,))
        xrange, yrange = jnp.linspace(0, 1, width), jnp.linspace(0, 1, height)

        xmask = (xstart - 0.5 * size <= xrange) & (xrange < xstart + 0.5 * size)
        ymask = (ystart - 0.5 * size <= yrange) & (yrange < ystart + 0.5 * size)
        return ~(xmask[None, None, :, None] & ymask[None, :, None, None])

    def __call__(self, images: Array, labels: Array) -> tuple[Array, Array]:
        if self.mixup_alpha == 0 and self.cutmix_alpha == 0:
            return images, labels
        if self.mixup_alpha > 0 and self.cutmix_alpha == 0:
            return self.apply_mixup(images, labels)
        if self.mixup_alpha == 0 and self.cutmix_alpha > 0:
            return self.apply_cutmix(images, labels)

        # If both mixup and cutmix are enabled, only one operation will be selected and
        # applied. Since jax does not support conditional branching on JIT, mixup and
        # cutmix are performed first and only one output will be selected.
        images1, labels1 = self.apply_mixup(images, labels)
        images2, labels2 = self.apply_cutmix(images, labels)

        cond = jax.random.uniform(self.make_rng("mixup")) > 0.5
        return jnp.where(cond, images1, images2), jnp.where(cond, labels1, labels2)


def fixed_sincos2d_embeddings(ncols: int, nrows: int, dim: int) -> Array:
    freqs = 1 / (10000 ** jnp.linspace(0, 1, dim // 4))
    x = jnp.outer(jnp.arange(0, nrows, dtype=jnp.float32), freqs)
    y = jnp.outer(jnp.arange(0, ncols, dtype=jnp.float32), freqs)

    x = jnp.broadcast_to(x[None, :, :], (ncols, nrows, dim // 4))
    y = jnp.broadcast_to(y[:, None, :], (ncols, nrows, dim // 4))
    return jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=2)


def modified_lamb(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mask: optax.MaskOrFn = None,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay=weight_decay, mask=mask),
        # Change to use trust ratio on weight decay parameters only.
        optax.masked(optax.scale_by_trust_ratio(), mask=mask),
        optax.scale_by_learning_rate(learning_rate),
    )


def get_layer_index_fn(path: tuple[DictKey, ...], _: Any, num_layers: int = 12) -> int:
    if path[0].key == "model" and path[1].key.startswith("layer_"):
        return int(re.match(r"layer_(\d+)", path[1].key).group(1)) + 1
    if path[0].key == "model" and path[1].key == "embed":
        return 0
    return num_layers


def load_pretrained_params(args: argparse.Namespace, params: ArrayTree) -> ArrayTree:
    with wds.gopen(args.pretrained_ckpt) as fp:
        new_params = flax.serialization.msgpack_restore(fp.read())

    # The positional embeddings will be resized when there is a difference in image
    # resolutions between pretraining and finetuning stage.
    if (
        args.posemb == "learnable"
        and new_params["model"]["embed"]["wpe"].shape
        != params["model"]["embed"]["wpe"].shape
    ):
        new_params["model"]["embed"]["wpe"] = jax.image.resize(
            new_params["model"]["embed"]["wpe"],
            params["model"]["embed"]["wpe"].shape,
            method="bicubic",
        )

    # Reinitialize the classifier head if the model was pretrained on different dataset
    # and `args.label_mapping` is not specified.
    if (
        "head" not in new_params["model"]
        or args.label_mapping is None
        and new_params["model"]["head"]["kernel"].shape
        != params["model"]["head"]["kernel"].shape
    ):
        new_params["model"]["head"] = params["model"]["head"]

    # If `args.label_mapping` is specified, then the same labels will automatically
    # replaced with the pretrained ones.
    if args.label_mapping:
        with wds.gopen(args.label_mapping) as fp:
            label_mapping = json.load(fp)
            src, dst = label_mapping["src"], label_mapping["dst"]

        kernel = np.zeros_like(params["model"]["head"]["kernel"])
        kernel[:, dst] = new_params["model"]["head"]["kernel"][:, src]

        bias = np.full_like(params["model"]["head"]["bias"], fill_value=-10.0)
        bias[dst] = new_params["model"]["head"]["bias"][src]

        new_params["model"]["head"] = {"kernel": kernel, "bias": bias}
    return new_params
