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

## Select version.
OPENMPI_VERSION=$1

set -e

mkdir -p /tmp
cd /tmp
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz >/dev/null
gunzip -c openmpi-${OPENMPI_VERSION}.tar.gz | tar xf -
cd openmpi-${OPENMPI_VERSION}
./configure --prefix=/usr/local/  >/dev/null
make -j4  >/dev/null
make install  >/dev/null
export PATH=/usr/local/bin:/usr/local/:$PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
echo "export PATH=/usr/local/bin/:/usr/local/:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" >> ~/.bashrc

cd /tmp
rm -rf /tmp/openmpi-*