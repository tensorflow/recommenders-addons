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

# Select RocksDB version.
ROCKSDB_VERSION=$1

set -e

# Install additional packages needed fo RocksDB
apt-get update
apt-get install -y \
  unar \
  zlib1g-dev \
  libgflags-dev \
  libsnappy-dev \
  libbz2-dev \
  liblz4-dev \
  libzstd-dev \
  && \
rm -rf /var/lib/apt/lists/*

ln -sf /usr/lib/x86_64-linux-gnu/libbz2*.so* /usr/local/lib/
ln -sf /usr/lib/x86_64-linux-gnu/liblz4*.so* /usr/local/lib/
ln -sf /usr/lib/x86_64-linux-gnu/libzstd*.so* /usr/local/lib/

install_dir=${2:-"/usr/local/lib"}
mkdir -p ${install_dir}
mkdir -p /tmp
cd /tmp
rm -rf /tmp/v$ROCKSDB_VERSION.tar.gz
wget -O /tmp/v$ROCKSDB_VERSION.tar.gz \
  https://github.com/facebook/rocksdb/archive/refs/tags/v$ROCKSDB_VERSION.tar.gz >/dev/null
tar -xvf /tmp/v$ROCKSDB_VERSION.tar.gz >/dev/null
cd /tmp/rocksdb-$ROCKSDB_VERSION

# Warning: we assume the official TensorFlow installed in CI environment is
# compiled by "-D_GLIBCXX_USE_CXX11_ABI=0"
DEBUG_LEVEL=0 make static_lib -j \
  EXTRA_CXXFLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0" \
  EXTRA_CFLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
chmod -R 777 /tmp/rocksdb-$ROCKSDB_VERSION/librocksdb*
cp /tmp/rocksdb-$ROCKSDB_VERSION/librocksdb* ${install_dir}
rm -f /tmp/$ROCKSDB_VERSION.tar.gz
rm -rf /tmp/rocksdb-${ROCKSDB_VERSION}
