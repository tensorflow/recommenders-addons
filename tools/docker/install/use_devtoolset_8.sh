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

# Use devtoolset-8 as tool chain
rm -r /usr/bin/gcc*
export PATH=/dt8/usr/bin:${PATH}
export PATH=/usr/bin/:/usr/local/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}
ln -sf /dt8/usr/bin/cc /usr/bin/gcc
ln -sf /dt8/usr/bin/gcc /usr/bin/gcc
ln -sf /dt8/usr/bin/g++ /usr/bin/g++

