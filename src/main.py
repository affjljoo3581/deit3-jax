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
import random
import warnings

import jax
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.serialization import msgpack_serialize
from flax.training.common_utils import shard
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from training import TrainState, create_train_state, training_step, validation_step
from utils import AverageMeter, save_checkpoint_in_background

warnings.filterwarnings("ignore")


def evaluate(state: TrainState, dataloader: DataLoader) -> dict[str, float]:
    average_meter = AverageMeter()
    for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
        metrics = validation_step(state, shard(jax.tree_map(np.asarray, batch)))
        average_meter.update(**jax.device_get(unreplicate(metrics)))

    metrics = average_meter.summary("val/")
    num_samples = metrics.pop("val/num_samples")
    return jax.tree_map(lambda x: x / num_samples, metrics)


def main(args: argparse.Namespace):
    train_dataloader, valid_dataloader = create_dataloaders(args)
    train_dataloader_iter = iter(train_dataloader)
    state = create_train_state(args).replicate()

    if jax.process_index() == 0:
        wandb.init(name=args.name, project=args.project, config=args)
    average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0

    for step in tqdm.trange(1, args.training_steps + 1, dynamic_ncols=True):
        for _ in range(args.grad_accum):
            batch = shard(jax.tree_map(np.asarray, next(train_dataloader_iter)))
            state, metrics = training_step(state, batch)
            average_meter.update(**unreplicate(metrics))

        if (
            jax.process_index() == 0
            and args.log_interval > 0
            and step % args.log_interval == 0
        ):
            metrics = average_meter.summary(prefix="train/")
            metrics["processed_samples"] = step * args.train_batch_size
            wandb.log(metrics, step)

        if args.eval_interval > 0 and (
            step % args.eval_interval == 0 or step == args.training_steps
        ):
            if jax.process_index() == 0:
                params_bytes = msgpack_serialize(unreplicate(state.params))
                save_checkpoint_in_background(args, params_bytes, postfix="last")
            if valid_dataloader is None:
                continue

            metrics = evaluate(state, valid_dataloader)
            if jax.process_index() == 0:
                if metrics["val/acc1"] > max_val_acc1:
                    max_val_acc1 = metrics["val/acc1"]
                    save_checkpoint_in_background(args, params_bytes, postfix="best")

                metrics["val/acc1/best"] = max_val_acc1
                metrics["processed_samples"] = step * args.train_batch_size
                wandb.log(metrics, step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-shards")
    parser.add_argument("--valid-dataset-shards")
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--valid-batch-size", type=int, default=256)
    parser.add_argument("--train-loader-workers", type=int, default=40)
    parser.add_argument("--valid-loader-workers", type=int, default=5)

    parser.add_argument("--random-crop", default="rrc")
    parser.add_argument("--color-jitter", type=float, default=0.0)
    parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--random-erasing", type=float, default=0.25)
    parser.add_argument("--augment-repeats", type=int, default=3)
    parser.add_argument("--test-crop-ratio", type=float, default=0.875)

    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--criterion", default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--labels", type=int, default=-1)
    parser.add_argument("--layerscale", action="store_true", default=False)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--posemb", default="learnable")
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--droppath", type=float, default=0.1)
    parser.add_argument("--grad-ckpt", action="store_true", default=False)

    parser.add_argument("--init-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--mixup-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--dropout-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--shuffle-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--pretrained-ckpt")
    parser.add_argument("--label-mapping")

    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)

    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--training-steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=0)

    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--ipaddr")
    parser.add_argument("--hostname")
    parser.add_argument("--output-dir", default=".")
    main(parser.parse_args())
