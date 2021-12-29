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


install_dir=${2:-"/usr/local/lib"}
mkdir -p ${install_dir}
mkdir -p /tmp
cd /tmp
rm -rf /tmp/v$ROCKSDB_VERSION.tar.gz
wget -O /tmp/v$ROCKSDB_VERSION.tar.gz https://github.com/facebook/rocksdb/archive/refs/tags/v$ROCKSDB_VERSION.tar.gz >/dev/null
tar -xvf /tmp/v$ROCKSDB_VERSION.tar.gz >/dev/null
cd /tmp/rocksdb-$ROCKSDB_VERSION
DEBUG_LEVEL=0 make static_lib -j EXTRA_CXXFLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0" EXTRA_CFLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
chmod -R 777 /tmp/rocksdb-$ROCKSDB_VERSION/librocksdb.so*
cp /tmp/rocksdb-$ROCKSDB_VERSION/librocksdb.so ${install_dir}
rm -f /tmp/$ROCKSDB_VERSION.tar.gz
rm -rf /tmp/rocksdb-${ROCKSDB_VERSION}

# Enable bazel auto completion.
echo "export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc
