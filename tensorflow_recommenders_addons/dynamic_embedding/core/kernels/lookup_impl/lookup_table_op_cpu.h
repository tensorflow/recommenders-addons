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

#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_CPU_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_CPU_H_

#include <typeindex>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
namespace cpu {

template <class V, size_t DIM>
class ValueArray final : public std::array<V, DIM> {
public:
  inline ValueArray<V, DIM> &
  operator+=(const ValueArray<V, DIM> &rhs) noexcept {
    for (size_t i = 0; i < DIM; i++) {
      (*this)[i] += rhs[i];
    }
    return *this;
  }
};

template <class V, size_t N>
class DefaultValueArray final : public gtl::InlinedVector<V, N> {
public:
  inline DefaultValueArray<V, N> &
  operator+=(const DefaultValueArray<V, N> &rhs) noexcept {
    for (size_t i = 0; i < this->size(); i++) {
      (*this)[i] = ((*this)[i]) + rhs[i];
    }
    return *this;
  }
};

template <>
class DefaultValueArray<tstring, 2> final
    : public gtl::InlinedVector<tstring, 2> {
public:
  inline DefaultValueArray<tstring, 2> &
  operator+=(const DefaultValueArray<tstring, 2> &rhs) noexcept {
    LOG(ERROR) << "Error: the accum is not supported for string value!";
    return *this;
  }
};

template <class V> using Tensor2D = typename tensorflow::TTypes<V, 2>::Tensor;

template <class V>
using ConstTensor2D = const typename tensorflow::TTypes<V, 2>::ConstTensor;

template <class K> struct HybridHash {
  inline std::size_t operator()(K const &s) const noexcept {
    return std::hash<K>{}(s);
  }
};

template <> struct HybridHash<int64> {
  inline std::size_t operator()(int64 const &key) const noexcept {
    uint64_t k = static_cast<uint64_t>(key);
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return static_cast<std::size_t>(k);
  }
};

template <> struct HybridHash<int32> {
  inline int32 operator()(int32 const &key) const noexcept {
    uint32_t k = static_cast<uint32_t>(key);
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;

    return static_cast<int32>(k);
  }
};

template <class K, class V> class TableWrapperBase {
public:
  virtual ~TableWrapperBase() {}
  virtual bool insert_or_assign(K key, ConstTensor2D<V> &value_flat,
                                int64 value_dim, int64 index) const {
    return false;
  }
  virtual bool insert_or_assign(K *key, V *value, int64 value_dim) const {
    return false;
  }
  virtual bool insert_or_accum(K key, ConstTensor2D<V> &value_or_delta_flat,
                               bool exist, int64 value_dim, int64 index) const {
    return false;
  }
  virtual void find(const K &key, Tensor2D<V> &value_flat,
                    ConstTensor2D<V> &default_flat, int64 value_dim,
                    bool is_full_size_default, int64 index) const {}
  virtual void find(const K &key, Tensor2D<V> &value_flat,
                    ConstTensor2D<V> &default_flat, bool &exist,
                    int64 value_dim, bool is_full_size_default,
                    int64 index) const {}
  virtual size_t dump(K *keys, V *values, const size_t search_offset,
                      const size_t search_length) const {
    return 0;
  }
  virtual size_t size() const { return 0; }
  virtual void clear() {}
  virtual bool erase(const K &key) { return false; }
  virtual void prefetch(const K &key) const {}
};

#define DECLARE_CREATE_TABLE(K, V)                                             \
  void CreateTable(size_t init_size, size_t runtime_dim,                       \
                   TableWrapperBase<K, V> **pptable)

DECLARE_CREATE_TABLE(int32, double);
DECLARE_CREATE_TABLE(int32, float);
DECLARE_CREATE_TABLE(int32, int32);
DECLARE_CREATE_TABLE(int32, bfloat16);
DECLARE_CREATE_TABLE(int64, double);
DECLARE_CREATE_TABLE(int64, float);
DECLARE_CREATE_TABLE(int64, int32);
DECLARE_CREATE_TABLE(int64, int64);
DECLARE_CREATE_TABLE(int64, tstring);
DECLARE_CREATE_TABLE(int64, int8);
DECLARE_CREATE_TABLE(int64, Eigen::half);
DECLARE_CREATE_TABLE(int64, bfloat16);
DECLARE_CREATE_TABLE(tstring, bool);
DECLARE_CREATE_TABLE(tstring, double);
DECLARE_CREATE_TABLE(tstring, float);
DECLARE_CREATE_TABLE(tstring, int32);
DECLARE_CREATE_TABLE(tstring, int64);
DECLARE_CREATE_TABLE(tstring, int8);
DECLARE_CREATE_TABLE(tstring, Eigen::half);
DECLARE_CREATE_TABLE(tstring, bfloat16);

#undef CREATE_A_TABLE
#undef CREATE_DEFAULT_TABLE
#undef CREATE_TABLE_PARTIAL_BRANCHES
#undef CREATE_TABLE_ALL_BRANCHES
#undef DECLARE_CREATE_TABLE

} // namespace cpu
} // namespace lookup
} // namespace recommenders_addons
} // namespace tensorflow

#endif // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_CPU_H_
