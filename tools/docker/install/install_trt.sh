#!/usr/bin/env bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

## Select version.
TRT_VERSION=$1

set -e

# Install TensorRT.
echo \
  deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 / \
  > /etc/apt/sources.list.d/nvidia-ml.list \
  && \
apt-get update && apt-get install -y \
  libnvinfer-dev=${TRT_VERSION} \
  libnvinfer7=${TRT_VERSION} \
  libnvinfer-plugin-dev=${TRT_VERSION} \
  libnvinfer-plugin7=${TRT_VERSION} \
    && \
  rm -rf /var/lib/apt/lists/*