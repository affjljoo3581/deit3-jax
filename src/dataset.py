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
import copy
import itertools
from collections.abc import Iterator
from functools import partial
from typing import Any

import jax
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import webdataset as wds
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from torch.utils.data import DataLoader, default_collate

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])


def auto_augment_factory(args: argparse.Namespace) -> T.Transform:
    aa_hparams = {
        "translate_const": int(args.image_size * 0.45),
        "img_mean": tuple((IMAGENET_DEFAULT_MEAN * 0xFF).astype(int)),
    }
    if args.auto_augment == "none":
        return T.Identity()
    if args.auto_augment.startswith("rand"):
        return rand_augment_transform(args.auto_augment, aa_hparams)
    if args.auto_augment.startswith("augmix"):
        aa_hparams["translate_pct"] = 0.3
        return augment_and_mix_transform(args.auto_augment, aa_hparams)
    return auto_augment_transform(args.auto_augment, aa_hparams)


def create_transforms(args: argparse.Namespace) -> tuple[nn.Module, nn.Module]:
    if args.random_crop == "rrc":
        train_transforms = [T.RandomResizedCrop(args.image_size, interpolation=3)]
    elif args.random_crop == "src":
        train_transforms = [
            T.Resize(args.image_size, interpolation=3),
            T.RandomCrop(args.image_size, padding=4, padding_mode="reflect"),
        ]
    elif args.random_crop == "none":
        train_transforms = [
            T.Resize(args.image_size, interpolation=3),
            T.CenterCrop(args.image_size),
        ]

    train_transforms += [
        T.RandomHorizontalFlip(),
        auto_augment_factory(args),
        T.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter),
        T.RandomErasing(args.random_erasing, value="random"),
        T.PILToTensor(),
    ]
    valid_transforms = [
        T.Resize(int(args.image_size / args.test_crop_ratio), interpolation=3),
        T.CenterCrop(args.image_size),
        T.PILToTensor(),
    ]
    return T.Compose(train_transforms), T.Compose(valid_transforms)


def repeat_samples(samples: Iterator[Any], repeats: int = 1) -> Iterator[Any]:
    for sample in samples:
        for _ in range(repeats):
            yield copy.deepcopy(sample)


def collate_and_shuffle(batch: list[Any], repeats: int = 1) -> Any:
    return default_collate(sum([batch[i::repeats] for i in range(repeats)], []))


def collate_and_pad(batch: list[Any], batch_size: int = 1) -> Any:
    pad = tuple(torch.full_like(x, fill_value=-1) for x in batch[0])
    return default_collate(batch + [pad] * (batch_size - len(batch)))


def create_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader | None, DataLoader | None]:
    train_dataloader, valid_dataloader = None, None
    train_transform, valid_transform = create_transforms(args)

    if args.train_dataset_shards is not None:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(args.train_dataset_shards, seed=args.shuffle_seed),
            itertools.cycle,
            wds.detshuffle(),
            wds.slice(jax.process_index(), None, jax.process_count()),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.detshuffle(),
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
            partial(repeat_samples, repeats=args.augment_repeats),
            wds.map_tuple(train_transform, torch.tensor),
        )
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size // jax.process_count() // args.grad_accum,
            num_workers=args.train_loader_workers,
            collate_fn=partial(collate_and_shuffle, repeats=args.augment_repeats),
            drop_last=True,
            prefetch_factor=20,
            persistent_workers=True,
        )
    if args.valid_dataset_shards is not None:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(args.valid_dataset_shards),
            wds.slice(jax.process_index(), None, jax.process_count()),
            wds.split_by_worker,
            wds.cached_tarfile_to_samples(),
            wds.decode("pil"),
            wds.to_tuple("jpg", "cls"),
            wds.map_tuple(valid_transform, torch.tensor),
        )
        valid_dataloader = DataLoader(
            dataset,
            batch_size=(batch_size := args.valid_batch_size // jax.process_count()),
            num_workers=args.valid_loader_workers,
            collate_fn=partial(collate_and_pad, batch_size=batch_size),
            drop_last=False,
            prefetch_factor=20,
            persistent_workers=True,
        )
    return train_dataloader, valid_dataloader
