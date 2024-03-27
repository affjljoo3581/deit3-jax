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

import flax
import jax.numpy as jnp
import numpy as np
import torch


def main(args: argparse.Namespace):
    with open(args.checkpoint, "rb") as fp:
        params = flax.serialization.msgpack_restore(fp.read())

    pos_embed = params["model"]["embed"]["wpe"]
    pos_embed = pos_embed.reshape(1, -1, pos_embed.shape[-1])
    wte = params["model"]["embed"]["wte"]["kernel"].transpose(3, 2, 0, 1)

    state_dict = {
        "cls_token": params["model"]["embed"]["cls_token"],
        "pos_embed": jnp.pad(pos_embed, ((0, 0), (1, 0), (0, 0))),
        "patch_embed.proj.weight": wte,
        "patch_embed.proj.bias": params["model"]["embed"]["wte"]["bias"],
    }
    if "norm" in params["model"]:
        state_dict["norm.weight"] = params["model"]["norm"]["scale"]
        state_dict["norm.bias"] = params["model"]["norm"]["bias"]
    if "head" in params["model"] and not args.exclude_heads:
        state_dict["head.weight"] = params["model"]["head"]["kernel"].transpose(1, 0)
        state_dict["head.bias"] = params["model"]["head"]["bias"]

    for name, layer in params["model"].items():
        if not name.startswith("layer_"):
            continue
        layer_idx = int(name[6:])

        wq = layer["attn"]["wq"]["kernel"]
        wk = layer["attn"]["wk"]["kernel"]
        wv = layer["attn"]["wv"]["kernel"]
        wo = layer["attn"]["wo"]["kernel"]

        wq = wq.reshape(wq.shape[0], -1)
        wk = wk.reshape(wk.shape[0], -1)
        wv = wv.reshape(wv.shape[0], -1)
        wo = wo.reshape(wv.shape[0], -1)
        qkv = jnp.concatenate((wq, wk, wv), axis=1).transpose(1, 0)

        state_dict[f"blocks.{layer_idx}.attn.qkv.weight"] = qkv
        state_dict[f"blocks.{layer_idx}.attn.qkv.bias"] = jnp.concatenate(
            (
                layer["attn"]["wq"]["bias"].reshape(-1),
                layer["attn"]["wk"]["bias"].reshape(-1),
                layer["attn"]["wv"]["bias"].reshape(-1),
            ),
        )
        state_dict[f"blocks.{layer_idx}.attn.proj.weight"] = wo.transpose(1, 0)
        state_dict[f"blocks.{layer_idx}.attn.proj.bias"] = layer["attn"]["wo"]["bias"]

        fc1 = layer["ff"]["w1"]["kernel"].transpose(1, 0)
        fc2 = layer["ff"]["w2"]["kernel"].transpose(1, 0)
        state_dict[f"blocks.{layer_idx}.mlp.fc1.weight"] = fc1
        state_dict[f"blocks.{layer_idx}.mlp.fc1.bias"] = layer["ff"]["w1"]["bias"]
        state_dict[f"blocks.{layer_idx}.mlp.fc2.weight"] = fc2
        state_dict[f"blocks.{layer_idx}.mlp.fc2.bias"] = layer["ff"]["w2"]["bias"]

        state_dict[f"blocks.{layer_idx}.norm1.weight"] = layer["norm1"]["scale"]
        state_dict[f"blocks.{layer_idx}.norm1.bias"] = layer["norm1"]["bias"]
        state_dict[f"blocks.{layer_idx}.norm2.weight"] = layer["norm2"]["scale"]
        state_dict[f"blocks.{layer_idx}.norm2.bias"] = layer["norm2"]["bias"]

        if "scale1" in layer:
            state_dict[f"blocks.{layer_idx}.ls1.gamma"] = layer["scale1"]
        if "scale2" in layer:
            state_dict[f"blocks.{layer_idx}.ls2.gamma"] = layer["scale2"]

    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}
    torch.save(state_dict, args.checkpoint.replace(".msgpack", ".pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--exclude-heads", action="store_true", default=False)
    main(parser.parse_args())
