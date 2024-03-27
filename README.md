# deit3-jax

## Introduction

This project aims to re-implement [DeiT](https://arxiv.org/abs/2012.12877) and [DeiT-III](https://arxiv.org/abs/2204.07118) using Jax/Flax and running on TPUs. Given that [the original repository](https://github.com/facebookresearch/deit) is written in PyTorch, this project provides an alternative codebase for training a variant of ViT on TPUs.

## Pretrained Checkpoints

We have trained ViTs using both DeiT and DeiT-III recipes. All experiments were done on a `v4-64` pod slice, and you can see the training details in the [wandb logs](https://wandb.ai/affjljoo3581/deit3-jax).

### DeiT Reproduction
| Name | Data | Resolution | Epochs | Time | Reimpl. | Original | Config | Wandb | Model |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| T/16 | in1k | 224 | 300 | 2h 40m | 73.1% | 72.2% | [config](config/deit/deit-t16-224-in1k-300ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/icdx9h5c) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit-t16-224-in1k-300ep-best.msgpack?download=true) |
| S/16 | in1k | 224 | 300 | 2h 43m | 79.68% | 79.8% | [config](config/deit/deit-s16-224-in1k-300ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/hvp0ab58) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit-s16-224-in1k-300ep-best.msgpack?download=true) |
| B/16 | in1k | 224 | 300 | 4h 40m | 81.46% | 81.8% | [config](config/deit/deit-b16-224-in1k-300ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/98gmcuko) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit-b16-224-in1k-300ep-best.msgpack?download=true) |

### DeiT-III on ImageNet-1k
| Name | Data | Resolution | Epochs | Time | Reimpl. | Original | Config | Wandb | Model |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S/16 | in1k | 224 | 400 | 2h 38m | 80.7% | 80.4% | [config](config/deit3/in1k/deit3-s16-224-in1k-400ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/31b1e6as) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-s16-224-in1k-400ep-best.msgpack?download=true) |
| S/16 | in1k | 224 | 800 | 5h 19m | 81.44% | 81.4% | [config](config/deit3/in1k/deit3-s16-224-in1k-800ep.sh) | [log](https://wandb.ai/affjljoo3581/deit3-jax/runs/yr3hvjo6) | [ckpt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-s16-224-in1k-800ep-best.msgpack?download=true) |
| B/16 | in1k | 192 &rarr; 224 | 400 | 4h 42m | 83.6% | 83.5% | [pt](config/deit3/in1k/deit3-b16-pt-192-in1k-400ep.sh) / [ft](config/deit3/in1k/deit3-b16-pt-192-in1k-400ep-ft-224-in1k-20ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/xxvllnlg) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/a0vwhzi4) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-192-in1k-400ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-192-in1k-400ep-ft-224-in1k-20ep-best.msgpack?download=true) |
| B/16 | in1k | 192 &rarr; 224 | 800 | 9h 28m | 83.91% | 83.8% | [pt](config/deit3/in1k/deit3-b16-pt-192-in1k-800ep.sh) / [ft](config/deit3/in1k/deit3-b16-pt-192-in1k-800ep-ft-224-in1k-20ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/byqdoj52) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/3x8e2osw) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-192-in1k-800ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-192-in1k-800ep-ft-224-in1k-20ep-best.msgpack?download=true) |
| L/16 | in1k | 192 &rarr; 224 | 400 | 14h 10m | 84.62% | 84.5% | [pt](config/deit3/in1k/deit3-l16-pt-192-in1k-400ep.sh) / [ft](config/deit3/in1k/deit3-l16-pt-192-in1k-400ep-ft-224-in1k-20ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/mgbaib5r) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/ifupxgzo) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-l16-pt-192-in1k-400ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-l16-pt-192-in1k-400ep-ft-224-in1k-20ep-best.msgpack?download=true) |
| L/16 | in1k | 192 &rarr; 224 | 800 | - | - | 84.9% | [pt](config/deit3/in1k/deit3-l16-pt-192-in1k-800ep.sh) / [ft](config/deit3/in1k/deit3-l16-pt-192-in1k-800ep-ft-224-in1k-20ep.sh) | - | - |
| H/14 | in1k | 154 &rarr; 224 | 400 | 19h 10m | 85.12% | 85.1% | [pt](config/deit3/in1k/deit3-h14-pt-154-in1k-400ep.sh) / [ft](config/deit3/in1k/deit3-h14-pt-154-in1k-400ep-ft-224-in1k-20ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/1kjdjog9) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/owe92sze) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-h14-pt-154-in1k-400ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-h14-pt-154-in1k-400ep-ft-224-in1k-20ep-best.msgpack?download=true) |
| H/14 | in1k | 154 &rarr; 224 | 800 | - | - | 85.2% | [pt](config/deit3/in1k/deit3-h14-pt-154-in1k-800ep.sh) / [ft](config/deit3/in1k/deit3-h14-pt-154-in1k-800ep-ft-224-in1k-20ep.sh) | - | - |

### DeiT-III on ImageNet-21k
| Name | Data | Resolution | Epochs | Time | Reimpl. | Original | Config | Wandb | Model |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S/16 | in21k | 224 | 90 | 7h 30m | 83.04% | 82.6% | [pt](config/deit3/in21k/deit3-s16-pt-224-in21k-90ep.sh) / [ft](config/deit3/in21k/deit3-s16-pt-224-in21k-90ep-ft-224-in1k-50ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/e7g7tby0) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/fb11x57v) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-s16-pt-224-in21k-90ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-s16-pt-224-in21k-90ep-ft-224-in1k-50ep-best.msgpack?download=true) |
| S/16 | in21k | 224 | 240 | 20h 6m | 83.39% | 83.1% | [pt](config/deit3/in21k/deit3-s16-pt-224-in21k-240ep.sh) / [ft](config/deit3/in21k/deit3-s16-pt-224-in21k-240ep-ft-224-in1k-50ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/stn991x2) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/455purcq) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-s16-pt-224-in21k-240ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-s16-pt-224-in21k-240ep-ft-224-in1k-50ep-best.msgpack?download=true) |
| B/16 | in21k | 224 | 90 | 12h 12m | 85.35% | 85.2% | [pt](config/deit3/in21k/deit3-b16-pt-224-in21k-90ep.sh) / [ft](config/deit3/in21k/deit3-b16-pt-224-in21k-90ep-ft-224-in1k-50ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/z7mlj9i1) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/h1gpiqyh) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-224-in21k-90ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-224-in21k-90ep-ft-224-in1k-50ep-best.msgpack?download=true) |
| B/16 | in21k | 224 | 240 | 33h 9m | 85.68% | 85.7% | [pt](config/deit3/in21k/deit3-b16-pt-224-in21k-240ep.sh) / [ft](config/deit3/in21k/deit3-b16-pt-224-in21k-240ep-ft-224-in1k-50ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/eab416px) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/ewmkd6cm) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-224-in21k-240ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-b16-pt-224-in21k-240ep-ft-224-in1k-50ep-best.msgpack?download=true) |
| L/16 | in21k | 224 | 90 | 37h 13m | 86.83% | 86.8% | [pt](config/deit3/in21k/deit3-l16-pt-224-in21k-90ep.sh) / [ft](config/deit3/in21k/deit3-l16-pt-224-in21k-90ep-ft-224-in1k-50ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/5jqjaon1) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/h2g403l0) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-l16-pt-224-in21k-90ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-l16-pt-224-in21k-90ep-ft-224-in1k-50ep-best.msgpack?download=true) |
| L/16 | in21k | 224 | 240 | - | - | 87% | [pt](config/deit3/in21k/deit3-l16-pt-224-in21k-240ep.sh) / [ft](config/deit3/in21k/deit3-l16-pt-224-in21k-240ep-ft-224-in1k-50ep.sh) | - | - |
| H/14 | in21k | 126 &rarr; 224 | 90 | 35h 51m | 86.78% | 87.2% | [pt](config/deit3/in21k/deit3-h14-pt-126-in21k-90ep.sh) / [ft](config/deit3/in21k/deit3-h14-pt-126-in21k-90ep-ft-224-in1k-50ep.sh) | [pt](https://wandb.ai/affjljoo3581/deit3-jax/runs/1856u3pv) / [ft](https://wandb.ai/affjljoo3581/deit3-jax/runs/790jjb6a) | [pt](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-h14-pt-126-in21k-90ep-last.msgpack?download=true) / [ft](https://huggingface.co/affjljoo3581/deit3-jax/resolve/main/deit3-h14-pt-126-in21k-90ep-ft-224-in1k-50ep-best.msgpack?download=true) |
| H/14 | in21k | 126 &rarr; 224 | 240 | - | - | - | [pt](config/deit3/in21k/deit3-h14-pt-126-in21k-240ep.sh) / [ft](config/deit3/in21k/deit3-h14-pt-126-in21k-240ep-ft-224-in1k-50ep.sh) | - | - |


## Getting Started

### Environment Setup

To begin, create a TPU instance for training ViTs. We have tested on both `v3-8` and `v4-64`. We recommend using the `v4-64` pod slice. If you do not have any TPU quota, visit [this link](https://sites.research.google/trc/about/) and apply for the TRC program.

```bash
$ gcloud compute tpus tpu-vm create tpu-name \
    --zone=us-central2-b \
    --accelerator-type=v4-64 \
    --version=tpu-ubuntu2204-base 
```

Once the TPU instance is created, clone this repository and install the required dependencies. All dependencies and installation steps are sepcified in the [scripts/setup.sh](./scripts/setup.sh) file. Note that you should use the `gcloud` command to execute the same command on all nodes simultaneously. The `v4-64` pod slice contains 8 computing nodes, each with 4 v4 chips.

```bash
$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="git clone https://github.com/affjljoo3581/deit3-jax"
```

```bash
$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="bash deit3-jax/scripts/setup.sh"
```

Additionally, log in to your wandb account using the command below. Replace `$WANDB_API_KEY` with your own API key.

```bash
$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="source ~/miniconda3/bin/activate base; wandb login $WANDB_API_KEY"
```

### Prepare Dataset Shards

`deit3-jax` utilizes [webdataset](https://github.com/webdataset/webdataset) to load training samples from various sources, such as huggingface hub and GCS. [Timm](https://github.com/huggingface/pytorch-image-models) provides webdataset versions of [ImageNet-1k](https://huggingface.co/datasets/timm/imagenet-1k-wds) and [ImageNet-21k](https://huggingface.co/datasets/timm/imagenet-w21-wds) on the huggingface hub. We recommend copying the resources to your GCS bucket for faster download speeds. To download both datasets to your bucket, use the following command:

```bash
$ export HF_TOKEN=...
$ export GCS_DATASET_PATH=gs://...

$ bash scripts/prepare-imagenet1k-dataset.sh
$ bash scripts/prepare-imagenet21k-dataset.sh
```

For example, you can list the tarfiles in your bucket like this:

```bash
$ gsutil ls gs://affjljoo3581-tpu-v4-storage/datasets/imagenet-1k-wds/
gs://affjljoo3581-tpu-v4-storage/datasets/imagenet-1k-wds/imagenet1k-train-0000.tar
gs://affjljoo3581-tpu-v4-storage/datasets/imagenet-1k-wds/imagenet1k-train-0001.tar
gs://affjljoo3581-tpu-v4-storage/datasets/imagenet-1k-wds/imagenet1k-train-0002.tar
gs://affjljoo3581-tpu-v4-storage/datasets/imagenet-1k-wds/imagenet1k-train-0003.tar
gs://affjljoo3581-tpu-v4-storage/datasets/imagenet-1k-wds/imagenet1k-train-0004.tar
...
```

However, GCS is not the only way to use webdataset. Instead of prefetching into your own bucket, it is also possible to directly stream from the huggingface hub while training.

```bash
$ export TRAIN_SHARDS=https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-train-{0000..1023}.tar
$ export VALID_SHARDS=https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-validation-{00..63}.tar

$ python3 src/main.py \
    --train-dataset-shards "pipe:curl -s -L $TRAIN_SHARDS -H 'Authorization:Bearer $HF_TOKEN'" \
    --valid-dataset-shards "pipe:curl -s -L $VALID_SHARDS -H 'Authorization:Bearer $HF_TOKEN'" \
    ...
```
Since intermittent decreases in download performance may occur when streaming from the huggingface hub, we recommend using the GCS bucket for stable download speed and consistent training.

### Train ViTs

You can now train your ViTs using the command below. Replace `$CONFIG_FILE` with the path to the configuration file you want to use. Instead, you can customize your own training recipes by adjusting the [hyperparameters](#hyperparameters). The various training presets are available in the [config](./config) folder.

```bash
$ export GCS_MODEL_DIR=gs://...

$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="source ~/miniconda3/bin/activate base; cd deit3-jax; screen -dmL bash $CONFIG_FILE"
```

The training results will be saved to `$GCS_MODEL_DIR`. You can specify a local directory path instead of a GCS path to save models locally.

### Convert Checkpoints to Timm

To use the pretrained checkpoints, you can convert `.msgpack` to timm-compatible `.pth` files.
```bash
$ python scripts/convert_flax_to_pytorch.py deit3-s16-224-in1k-400ep-best.msgpack
$ ls
deit3-s16-224-in1k-400ep-best.msgpack  deit3-s16-224-in1k-400ep-best.pth
```

After converting `.msgpack` to `.pth`, you can load it with timm:
```python
>>> import torch
>>> import timm
>>> model = timm.create_model("vit_small_patch16_224", init_values=1e-4)
>>> model.load_state_dict(torch.load("deit3-s16-224-in1k-400ep-best.pth"))
<All keys matched successfully>
```

## Hyperparameters

### Image Augmentations
* `--random-crop`: Type of random cropping. Choose `none` for nothing, `rrc` for RandomResizedCrop, and `src` for SimpleResizedCrop proposed in DeiT-III.
* `--color-jitter`: Factor for color jitter augmentation.
* `--auto-augment`: Name of auto-augment policy used in Timm (e.g. `rand-m9-mstd0.5-inc1`).
* `--random-erasing`: Probability of random erasing augmentation.
* `--augment-repeats`: Number of augmentation repetitions.
* `--test-crop-ratio`: Center crop ratio for test preprocessing.
* `--mixup`: Factor (alpha) for Mixup augmentation. Disable by setting to 0.
* `--cutmix`: Factor (alpha) for CutMix augmentation. Disable by setting to 0.
* `--criterion`: Type of classification loss. Choose `ce` for softmax cross entropy and `bce` for sigmoid cross entropy.
* `--label-smoothing`: Factor for label smoothing.

### ViT Architecture
* `--layers`: Number of layers.
* `--dim`: Number of hidden features.
* `--heads`: Number of attention heads.
* `--labels`: Number of classification labels.
* `--layerscale`: Flag to enable LayerScale.
* `--patch-size`: Patch size in ViT embedding layer.
* `--image-size`: Input image size.
* `--posemb`: Type of positional embeddings in ViT. Choose `learnable` for learnable parameters and `sincos2d` for sinusoidal encoding.
* `--pooling`: Type of pooling strategy. Choose `cls` for using `[CLS]` token and `gap` for global average pooling.
* `--dropout`: Dropout rate.
* `--droppath`: DropPath rate.
* `--grad-ckpt`: Flag to enable gradient checkpointing for reducing memory footprint.

### Optimization
* `--optimizer`: Type of optimizer. Choose `adamw` for AdamW and `lamb` for LAMB.
* `--learning-rate`: Peak learning rate.
* `--weight-decay`: Decoupled weight decay rate.
* `--adam-b1`: Adam beta1.
* `--adam-b2`: Adam beta2.
* `--adam-eps`: Adam epsilon.
* `--lr-decay`: Layerwise learning rate decay rate.
* `--clip-grad`: Maximum gradient norm.
* `--grad-accum`: Number of gradient accumulation steps.
* `--warmup-steps`: Number of learning rate warmup steps.
* `--training-steps`: Number of total training steps.
* `--log-interval`: Number of logging intervals.
* `--eval-interval`: Number of evaluation intervals.

### Random Seeds
* `--init-seed`: Random seed for weight initialization.
* `--mixup-seed`: Random seed for Mixup and CutMix augmentations.
* `--dropout-seed`: Random seed for Dropout regularization.
* `--shuffle-seed`: Random seed for dataset shuffling.
* `--pretrained-ckpt`: Pretrained model path to load from.
* `--label-mapping`: Label mapping file to reuse the pretrained classification head for transfer learning.

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.

# Acknowledgement
Thanks to the [TPU Research Cloud](https://sites.research.google/trc/about/) program for providing resources. All models are trained on the TPU `v4-64` pod slice.