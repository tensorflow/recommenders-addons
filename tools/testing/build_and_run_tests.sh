#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
# usage: bash tools/testing/build_and_run_tests.sh

set -x -e

SKIP_CUSTOM_OP_TESTS_FLAG=${1}

export CC_OPT_FLAGS='-mavx'

python -m pip install --upgrade pip
cp tools/install_deps/pytest*.txt ./
bash tools/docker/install/install_pytest.sh

# Fix incorrect numpy installation
if [ "$TF_VERSION" = "2.6.3" ] ; then
  python -m pip install numpy==1.19.5 --force-reinstall
fi

TF_NEED_CUDA=$TF_NEED_CUDA python ./configure.py
bash tools/install_so_files.sh

# clean pip cache
python -m pip cache purge

# use 10 workers if a gpu is available, otherwise,
# one worker per cpu core. Kokoro has 38 cores, that'd be too much
# for the gpu memory, until we change the device placement to
# use multiple gpus when they are available.
EXTRA_ARGS="-n 1"
if ! [ -x "$(command -v nvidia-smi)" ]; then
  EXTRA_ARGS="-n auto"
fi

# Lack of HorovodJoin CPU kernels when install Horovod with NCCL
if [ "$(uname)" != "Darwin" ]; then
  # Mac only with MPI
  python -m pip uninstall horovod -y
  bash /install/install_horovod.sh $HOROVOD_VERSION --only-cpu
fi
# TODO(jamesrong): Test on GPU.
CUDA_VISIBLE_DEVICES="" mpirun -np 2 -H localhost:2 --allow-run-as-root pytest -v ./tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/horovod_sync_train_test.py
# Reinstall Horovod after tests
if [ "$(uname)" != "Darwin" ]; then
  # Mac only with MPI
  python -m pip uninstall horovod -y
  bash /install/install_horovod.sh $HOROVOD_VERSION
fi

# Only use GPU 0 if available.
if [ -x "$(command -v nvidia-smi)" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

python -m pytest -v -s --functions-durations=20 --modules-durations=5 $SKIP_CUSTOM_OP_TESTS_FLAG $EXTRA_ARGS ./tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/



# Release disk space
bazel clean --expunge
rm -f ./tensorflow_recommenders_addons/dynamic_embedding/core/_*_ops.so
