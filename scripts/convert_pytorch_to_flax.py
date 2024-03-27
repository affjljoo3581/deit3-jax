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
import timm
import torch


def main(args: argparse.Namespace):
    if args.from_timm:
        model = timm.create_model(
            args.checkpoint,
            pretrained=True,
            **({"num_classes": 0} if args.exclude_heads else {}),
        )
        state_dict = model.state_dict()
    else:
        state_dict = torch.load(args.checkpoint)

    cls_token = state_dict["cls_token"] + state_dict["pos_embed"][:, :1, :]
    pos_embed = state_dict["pos_embed"][:, 1:, :].squeeze(0)
    pos_embed = pos_embed.unflatten(0, (int(pos_embed.size(0) ** 0.5), -1))
    wte = state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0)

    params = {
        "model.embed.cls_token": cls_token,
        "model.embed.wpe": pos_embed,
        "model.embed.wte.kernel": wte,
        "model.embed.wte.bias": state_dict["patch_embed.proj.bias"],
        "model.norm.scale": state_dict["norm.weight"],
        "model.norm.bias": state_dict["norm.bias"],
    }
    if "head.weight" in state_dict and not args.exclude_heads:
        params["model.head.kernel"] = state_dict["head.weight"].transpose(1, 0)
        params["model.head.bias"] = state_dict["head.bias"]

    layer_idx = 0
    while f"blocks.{layer_idx}.norm1.weight" in state_dict:
        wqkv = state_dict[f"blocks.{layer_idx}.attn.qkv.weight"].transpose(1, 0)
        wq, wk, wv = wqkv.unflatten(1, (3, args.num_heads, -1)).permute(1, 0, 2, 3)
        wo = state_dict[f"blocks.{layer_idx}.attn.proj.weight"].transpose(1, 0)
        wo = wo.unflatten(0, (args.num_heads, -1))

        params[f"model.layer_{layer_idx}.attn.wq.kernel"] = wq
        params[f"model.layer_{layer_idx}.attn.wk.kernel"] = wk
        params[f"model.layer_{layer_idx}.attn.wv.kernel"] = wv
        params[f"model.layer_{layer_idx}.attn.wo.kernel"] = wo

        bqkv = state_dict[f"blocks.{layer_idx}.attn.qkv.bias"]
        bq, bk, bv = bqkv.view(3, args.num_heads, -1)
        bo = state_dict[f"blocks.{layer_idx}.attn.proj.bias"]
        params[f"model.layer_{layer_idx}.attn.wq.bias"] = bq
        params[f"model.layer_{layer_idx}.attn.wk.bias"] = bk
        params[f"model.layer_{layer_idx}.attn.wv.bias"] = bv
        params[f"model.layer_{layer_idx}.attn.wo.bias"] = bo

        wfc1 = state_dict[f"blocks.{layer_idx}.mlp.fc1.weight"].transpose(1, 0)
        wfc2 = state_dict[f"blocks.{layer_idx}.mlp.fc2.weight"].transpose(1, 0)
        bfc1 = state_dict[f"blocks.{layer_idx}.mlp.fc1.bias"]
        bfc2 = state_dict[f"blocks.{layer_idx}.mlp.fc2.bias"]
        params[f"model.layer_{layer_idx}.ff.w1.kernel"] = wfc1
        params[f"model.layer_{layer_idx}.ff.w2.kernel"] = wfc2
        params[f"model.layer_{layer_idx}.ff.w1.bias"] = bfc1
        params[f"model.layer_{layer_idx}.ff.w2.bias"] = bfc2

        snorm1 = state_dict[f"blocks.{layer_idx}.norm1.weight"]
        snorm2 = state_dict[f"blocks.{layer_idx}.norm2.weight"]
        bnorm1 = state_dict[f"blocks.{layer_idx}.norm1.bias"]
        bnorm2 = state_dict[f"blocks.{layer_idx}.norm2.bias"]
        params[f"model.layer_{layer_idx}.norm1.scale"] = snorm1
        params[f"model.layer_{layer_idx}.norm2.scale"] = snorm2
        params[f"model.layer_{layer_idx}.norm1.bias"] = bnorm1
        params[f"model.layer_{layer_idx}.norm2.bias"] = bnorm2

        if (scale := state_dict.get(f"blocks.{layer_idx}.ls1.gamma", None)) is not None:
            params[f"model.layer_{layer_idx}.scale1"] = scale
        if (scale := state_dict.get(f"blocks.{layer_idx}.ls2.gamma", None)) is not None:
            params[f"model.layer_{layer_idx}.scale2"] = scale

        layer_idx += 1

    with open(args.checkpoint.replace(".pth", ".msgpack"), "wb") as fp:
        params = {k: v.numpy() for k, v in params.items()}
        params = flax.traverse_util.unflatten_dict(params, sep=".")
        fp.write(flax.serialization.msgpack_serialize(params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--from-timm", action="store_true", default=False)
    parser.add_argument("--exclude-heads", action="store_true", default=False)
    main(parser.parse_args())
