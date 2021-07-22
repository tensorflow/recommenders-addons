#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Select bazel version.
REDIS_VERSION=$1

set +e
local_redis_ver=$(redis-server --version 2>&1 | grep -i v= | awk '{print $3}' | awk -F"=" '{print $2}')

if [[ "$local_redis_ver" == "$REDIS_VERSION" ]]; then
  exit 0
fi

set -e

# Install bazel.
install_dir=${2:-"/usr/local/bin"}
mkdir -p ${install_dir}
cd ${install_dir}
mkdir ${install_dir}/redis-$REDIS_VERSION
if [[ ! -f "bazel-$REDIS_VERSION.tar.gz" ]]; then
  curl -fSsL -O https://github.com/redis/redis/archive/refs/tags/$REDIS_VERSION.tar.gz
fi
tar tar -cvxf ${install_dir}/bazel-$REDIS_VERSION.tar.gz -C ${install_dir}/redis-$REDIS_VERSION
cd ${install_dir}/redis-$REDIS_VERSION
make -j4
make PREFIX=${install_dir} install
rm -f ${install_dir}/bazel-$REDIS_VERSION.tar.gz

# Enable bazel auto completion.
echo "export PATH=${install_dir}:$PATH" >> ~/.bashrc
