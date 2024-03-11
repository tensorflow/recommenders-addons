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
HOROVOD_VERSION=$1


# Install horovod
if [[ "$2" != "--only-cpu" ]]; then
  ln -sf /usr/include/nccl.h /usr/local/include/nccl.h
  HOROVOD_GPU_OPERATIONS=NCCL \
  HOROVOD_NCCL_INCLUDE=/usr/include/ \
  HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu/ \
  HOROVOD_WITH_TENSORFLOW=1 \
  HOROVOD_WITHOUT_PYTORCH=1 \
  HOROVOD_WITHOUT_MXNET=1 \
  HOROVOD_WITH_MPI=1 \
  HOROVOD_WITHOUT_GLOO=1 \
  python -m pip install --no-cache-dir --use-pep517 horovod==$HOROVOD_VERSION
else
  HOROVOD_WITH_TENSORFLOW=1 \
  HOROVOD_WITHOUT_PYTORCH=1 \
  HOROVOD_WITHOUT_MXNET=1 \
  HOROVOD_WITH_MPI=1 \
  HOROVOD_WITHOUT_GLOO=1 \
  python -m pip install --no-cache-dir --use-pep517 horovod==$HOROVOD_VERSION
fi
