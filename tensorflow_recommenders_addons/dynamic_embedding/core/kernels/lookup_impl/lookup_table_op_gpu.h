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

// This header provides only lightweight abstract interface declarations.
// Heavy template instantiation (TableWrapper, CreateTableImpl,
// nv_hashtable.cuh) lives in lookup_table_op_gpu_impl.h which is included ONLY
// by the split impl .cu.cc files to enable parallel compilation by Bazel.

#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_

#include <cuda_runtime_api.h>

#include <iomanip>
#include <limits>
#include <typeindex>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(val)                                                        \
  do {                                                                         \
    cudaError_t _status = (val);                                               \
    if (_status != cudaSuccess) {                                              \
      LOG(FATAL) << "CUDA error " << _status << ": "                           \
                 << cudaGetErrorString(_status) << " at " << __FILE__ << ":"   \
                 << __LINE__;                                                  \
    }                                                                          \
  } while (0)
#endif

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
namespace gpu {

using GPUDevice = Eigen::ThreadPoolDevice;

template <class V> struct ValueArrayBase {};

template <class V, size_t DIM> struct ValueArray : public ValueArrayBase<V> {
  V value[DIM];
};

template <class T> using ValueType = ValueArrayBase<T>;

template <class K, class V> class TableWrapperBase {
public:
  virtual ~TableWrapperBase() {}
  virtual void upsert(const K *d_keys, const ValueType<V> *d_vals, size_t len,
                      cudaStream_t stream) {}
  virtual void accum(const K *d_keys, const ValueType<V> *d_vals_or_deltas,
                     const bool *d_exists, size_t len, cudaStream_t stream) {}
  virtual void dump(K *d_key, ValueType<V> *d_val, const size_t offset,
                    const size_t search_length, size_t *d_dump_counter,
                    cudaStream_t stream) const {}
  virtual size_t rehash_if_needed(const size_t min_capacity,
                                  cudaStream_t stream,
                                  const size_t new_keys_num = 0,
                                  const size_t last_hint_size = 0) {
    return 0;
  }
  virtual void get(const K *d_keys, ValueType<V> *d_vals, bool *d_status,
                   size_t len, ValueType<V> *d_def_val, cudaStream_t stream,
                   bool is_full_size_default) const {}
  virtual size_t get_size(cudaStream_t stream) const { return 0; }
  virtual size_t get_capacity() const { return 0; }
  virtual void remove(const K *d_keys, size_t len, cudaStream_t stream) {}
  virtual void clear(cudaStream_t stream) {}
};

#define DECLARE_CREATE_TABLE(K, V)                                             \
  void CreateTable0(size_t max_size, size_t runtime_dim,                       \
                    TableWrapperBase<K, V> **);                                \
  void CreateTable1(size_t max_size, size_t runtime_dim,                       \
                    TableWrapperBase<K, V> **);                                \
  void CreateTable2(size_t max_size, size_t runtime_dim,                       \
                    TableWrapperBase<K, V> **);                                \
  void CreateTable3(size_t max_size, size_t runtime_dim,                       \
                    TableWrapperBase<K, V> **);

DECLARE_CREATE_TABLE(int64, float);
DECLARE_CREATE_TABLE(int64, Eigen::half);
DECLARE_CREATE_TABLE(int64, int64);
DECLARE_CREATE_TABLE(int64, int32);
DECLARE_CREATE_TABLE(int64, int8);
DECLARE_CREATE_TABLE(int32, float);

#undef DECLARE_CREATE_TABLE

} // namespace gpu
} // namespace lookup
} // namespace recommenders_addons
} // namespace tensorflow

#endif // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_GPU_H_
