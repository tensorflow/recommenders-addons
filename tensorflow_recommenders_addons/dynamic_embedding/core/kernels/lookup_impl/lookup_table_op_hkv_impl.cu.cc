/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_impl/lookup_table_op_gpu.h"
namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
namespace gpu {

#define DEFINE_PURE_GPU_HASHTABLE(key_type, value_type) \
    template<> class TableWrapper<key_type, value_type>


DEFINE_PURE_GPU_HASHTABLE(int64, float);
DEFINE_PURE_GPU_HASHTABLE(int64, int32);
DEFINE_PURE_GPU_HASHTABLE(int64, int64);
DEFINE_PURE_GPU_HASHTABLE(int64, int64);

#undef DEFINE_PURE_GPU_HASHTABLE


}  // namespace gpu
}  // namespace lookup
}  // namespace recommenders_addons
}  // namespace tensorflow
