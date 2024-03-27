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

seq -f "%04g" 0 2047 | xargs -P 0 -i wget https://huggingface.co/datasets/timm/imagenet-w21-wds/resolve/main/imagenet_w21-train-{}.tar --header "Authorization:Bearer $HF_TOKEN"
gsutil -m cp *.tar $GCS_DATASET_PATH/imagenet-w21-wds/