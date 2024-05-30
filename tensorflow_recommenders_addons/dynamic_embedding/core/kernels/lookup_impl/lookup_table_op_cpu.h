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
#include "tensorflow_recommenders_addons/dynamic_embedding/core/lib/cuckoo/cuckoohash_map.hh"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
namespace cpu {

template <class V, size_t DIM>
class ValueArray final : public std::array<V, DIM> {
 public:
  inline ValueArray<V, DIM>& operator+=(
      const ValueArray<V, DIM>& rhs) noexcept {
    for (size_t i = 0; i < DIM; i++) {
      (*this)[i] += rhs[i];
    }
    return *this;
  }
};

template <class V, size_t N>
class DefaultValueArray final : public gtl::InlinedVector<V, N> {
 public:
  inline DefaultValueArray<V, N>& operator+=(
      const DefaultValueArray<V, N>& rhs) noexcept {
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
  inline DefaultValueArray<tstring, 2>& operator+=(
      const DefaultValueArray<tstring, 2>& rhs) noexcept {
    LOG(ERROR) << "Error: the accum is not supported for string value!";
    return *this;
  }
};

template <class V>
using Tensor2D = typename tensorflow::TTypes<V, 2>::Tensor;

template <class V>
using ConstTensor2D = const typename tensorflow::TTypes<V, 2>::ConstTensor;

template <class K>
struct HybridHash {
  inline std::size_t operator()(K const& s) const noexcept {
    return std::hash<K>{}(s);
  }
};

template <>
struct HybridHash<int64> {
  inline std::size_t operator()(int64 const& key) const noexcept {
    uint64_t k = static_cast<uint64_t>(key);
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return static_cast<std::size_t>(k);
  }
};

template <>
struct HybridHash<int32> {
  inline int32 operator()(int32 const& key) const noexcept {
    uint32_t k = static_cast<uint32_t>(key);
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;

    return static_cast<int32>(k);
  }
};

template <class K, class V>
class TableWrapperBase {
 public:
  virtual ~TableWrapperBase() {}
  virtual bool insert_or_assign(K key, ConstTensor2D<V>& value_flat,
                                int64 value_dim, int64 index) const {
    return false;
  }
  virtual bool insert_or_assign(K* key, V* value, int64 value_dim) const {
    return false;
  }
  virtual bool insert_or_accum(K key, ConstTensor2D<V>& value_or_delta_flat,
                               bool exist, int64 value_dim, int64 index) const {
    return false;
  }
  virtual void find(const K& key, Tensor2D<V>& value_flat,
                    ConstTensor2D<V>& default_flat, int64 value_dim,
                    bool is_full_size_default, int64 index) const {}
  virtual void find(const K& key, Tensor2D<V>& value_flat,
                    ConstTensor2D<V>& default_flat, bool& exist,
                    int64 value_dim, bool is_full_size_default,
                    int64 index) const {}
  virtual size_t dump(K* keys, V* values, const size_t search_offset,
                      const size_t search_length) const {
    return 0;
  }
  virtual size_t size() const { return 0; }
  virtual void clear() {}
  virtual bool erase(const K& key) { return false; }
};

template <class K, class V, size_t DIM>
class TableWrapperOptimized final : public TableWrapperBase<K, V> {
 private:
  using ValueType = ValueArray<V, DIM>;
  using Table = cuckoohash_map<K, ValueType, HybridHash<K>>;

 public:
  explicit TableWrapperOptimized(size_t init_size) : init_size_(init_size) {
    table_ = new Table(init_size);
    LOG(INFO) << "HashTable on CPU is created on optimized mode:"
              << " K=" << std::type_index(typeid(K)).name()
              << ", V=" << std::type_index(typeid(V)).name() << ", DIM=" << DIM
              << ", init_size=" << init_size_;
  }

  ~TableWrapperOptimized() override { delete table_; }

  bool insert_or_assign(K key, ConstTensor2D<V>& value_flat, int64 value_dim,
                        int64 index) const override {
    ValueType value_vec;
    std::copy_n(value_flat.data() + index * value_dim, value_dim,
                value_vec.begin());
    return table_->insert_or_assign(key, value_vec);
  }

  bool insert_or_assign(K* key, V* value, int64 value_dim) const override {
    assert(value_dim == DIM);
    ValueType value_vec;
    std::copy_n(value, value_dim, value_vec.begin());
    return table_->insert_or_assign(*key, value_vec);
  }

  bool insert_or_accum(K key, ConstTensor2D<V>& value_or_delta_flat, bool exist,
                       int64 value_dim, int64 index) const override {
    ValueType value_or_delta_vec;
    std::copy_n(value_or_delta_flat.data() + index * value_dim, value_dim,
                value_or_delta_vec.begin());
    return table_->insert_or_accum(key, value_or_delta_vec, exist);
  }

  void find(const K& key, Tensor2D<V>& value_flat,
            ConstTensor2D<V>& default_flat, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    ValueType value_vec;
    if (table_->find(key, value_vec)) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    } else {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  void find(const K& key, Tensor2D<V>& value_flat,
            ConstTensor2D<V>& default_flat, bool& exist, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    ValueType value_vec;
    exist = table_->find(key, value_vec);
    if (exist) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    } else {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  size_t dump(K* keys, V* values, const size_t search_offset,
              const size_t search_length) const override {
    auto lt = table_->lock_table();
    auto lt_size = lt.size();
    if (search_offset > lt_size || lt_size == 0) {
      return 0;
    }
    auto search_begin = lt.begin();
    for (size_t i = 0; i < search_offset; ++i) {
      ++search_begin;
    }
    auto search_end = search_begin;
    if ((search_offset + search_length) >= lt_size) {
      search_end = lt.end();
    } else {
      for (size_t i = 0; i < search_length; ++i) {
        ++search_end;
      }
    }

    constexpr const size_t value_dim = DIM;
    K* key_ptr = keys;
    V* val_ptr = values;
    size_t dump_counter = 0;
    for (auto it = search_begin; it != search_end;
         ++it, ++key_ptr, val_ptr += value_dim) {
      const K& key = it->first;
      const ValueType& value = it->second;
      *key_ptr = key;
      std::copy_n(value.begin(), value_dim, val_ptr);
      ++dump_counter;
    }
    return dump_counter;
  }

  size_t size() const override { return table_->size(); }

  void clear() override { table_->clear(); }

  bool erase(const K& key) override { return table_->erase(key); }

 private:
  size_t init_size_;
  Table* table_;
};

template <class K, class V>
class TableWrapperDefault final : public TableWrapperBase<K, V> {
 private:
  using ValueType = DefaultValueArray<V, 2>;
  using Table = cuckoohash_map<K, ValueType, HybridHash<K>>;

 public:
  explicit TableWrapperDefault(size_t init_size) : init_size_(init_size) {
    table_ = new Table(init_size);
    LOG(INFO) << "HashTable on CPU is created on default mode:"
              << " K=" << std::type_index(typeid(K)).name()
              << ", V=" << std::type_index(typeid(V)).name()
              << ", init_size=" << init_size_;
  }

  ~TableWrapperDefault() override { delete table_; }

  bool insert_or_assign(K key, ConstTensor2D<V>& value_flat, int64 value_dim,
                        int64 index) const override {
    ValueType value_vec;
    value_vec.reserve(value_dim);
    for (int64 j = 0; j < value_dim; j++) {
      V value = value_flat(index, j);
      value_vec.push_back(value);
    }
    return table_->insert_or_assign(key, value_vec);
  }

  bool insert_or_assign(K* key, V* value, int64 value_dim) const override {
    ValueType value_vec;
    value_vec.reserve(value_dim);
    for (int64 j = 0; j < value_dim; j++) {
      value_vec.push_back(*(value + j));
    }
    return table_->insert_or_assign(*key, value_vec);
  }

  bool insert_or_accum(K key, ConstTensor2D<V>& value_or_delta_flat, bool exist,
                       int64 value_dim, int64 index) const override {
    ValueType value_or_delta_vec;
    value_or_delta_vec.reserve(value_dim);
    for (int64 j = 0; j < value_dim; j++) {
      value_or_delta_vec.push_back(value_or_delta_flat(index, j));
    }
    return table_->insert_or_accum(key, value_or_delta_vec, exist);
  }

  void find(const K& key, typename tensorflow::TTypes<V, 2>::Tensor& value_flat,
            ConstTensor2D<V>& default_flat, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    ValueType value_vec;
    value_vec.reserve(value_dim);
    if (table_->find(key, value_vec)) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    } else {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  void find(const K& key, typename tensorflow::TTypes<V, 2>::Tensor& value_flat,
            ConstTensor2D<V>& default_flat, bool& exist, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    ValueType value_vec;
    value_vec.reserve(value_dim);
    exist = table_->find(key, value_vec);
    if (exist) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    } else {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  size_t dump(K* keys, V* values, const size_t search_offset,
              const size_t search_length) const override {
    auto lt = table_->lock_table();
    auto lt_size = lt.size();
    if (search_offset > lt_size || lt_size == 0) {
      return 0;
    }
    auto search_begin = lt.begin();
    for (size_t i = 0; i < search_offset; ++i) {
      ++search_begin;
    }
    auto search_end = search_begin;
    if ((search_offset + search_length) >= lt_size) {
      search_end = lt.end();
    } else {
      for (size_t i = 0; i < search_length; ++i) {
        ++search_end;
      }
    }

    const auto value_dim = (lt.begin()->second).size();
    K* key_ptr = keys;
    V* val_ptr = values;
    size_t dump_counter = 0;
    for (auto it = search_begin; it != search_end;
         ++it, ++key_ptr, val_ptr += value_dim) {
      const K& key = it->first;
      const ValueType& value = it->second;
      *key_ptr = key;
      std::copy_n(value.begin(), value_dim, val_ptr);
      ++dump_counter;
    }
    return dump_counter;
  }

  size_t size() const override { return table_->size(); }

  void clear() override { table_->clear(); }

  bool erase(const K& key) override { return table_->erase(key); }

 private:
  size_t init_size_;
  Table* table_;
};

template <class K, class V, size_t DIM, bool OPTIMIZE>
struct TableDispatcherImpl {
  using DefaultTable = TableWrapperDefault<K, V>;
  using table_type = DefaultTable;
};

template <class K, class V, size_t DIM>
struct TableDispatcherImpl<K, V, DIM, true> {
  using OptimizedTable = TableWrapperOptimized<K, V, DIM>;
  using table_type = OptimizedTable;
};

template <class K, class V, size_t DIM>
struct TableDispatcher {
  static constexpr bool IS_FIX_RANGE = (DIM <= 100);
  static constexpr bool K_IS_INT64 = std::is_same<K, int64>::value;
  static constexpr bool V_IS_TSTRING = std::is_same<V, tstring>::value;
  static constexpr bool OPTIMIZED =
      (IS_FIX_RANGE && K_IS_INT64 && !V_IS_TSTRING);
  using table_type =
      typename TableDispatcherImpl<K, V, DIM, OPTIMIZED>::table_type;
};

#define CREATE_A_TABLE(DIM)                                                \
  do {                                                                     \
    if (runtime_dim == (DIM + 1)) {                                        \
      using Table = typename TableDispatcher<K, V, (DIM + 1)>::table_type; \
      *pptable = new Table(init_size);                                     \
      return;                                                              \
    };                                                                     \
  } while (0)

#define CREATE_DEFAULT_TABLE()               \
  do {                                       \
    using Table = TableWrapperDefault<K, V>; \
    *pptable = new Table(init_size);         \
    return;                                  \
  } while (0)

#define CREATE_TABLE_PARTIAL_BRANCHES(PREFIX) \
  do {                                        \
    CREATE_A_TABLE((PREFIX)*10 + 0);          \
    CREATE_A_TABLE((PREFIX)*10 + 1);          \
    CREATE_A_TABLE((PREFIX)*10 + 2);          \
    CREATE_A_TABLE((PREFIX)*10 + 3);          \
    CREATE_A_TABLE((PREFIX)*10 + 4);          \
    CREATE_A_TABLE((PREFIX)*10 + 5);          \
    CREATE_A_TABLE((PREFIX)*10 + 6);          \
    CREATE_A_TABLE((PREFIX)*10 + 7);          \
    CREATE_A_TABLE((PREFIX)*10 + 8);          \
    CREATE_A_TABLE((PREFIX)*10 + 9);          \
  } while (0)

// create branches with dim range [1, 100]
#define CREATE_TABLE_ALL_BRANCHES(CENTILE, DECTILE)          \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 0); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 1); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 2); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 3); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 4); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 5); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 6); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 7); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 8); \
  CREATE_TABLE_PARTIAL_BRANCHES(CENTILE * 10 + DECTILE + 9); \
  CREATE_DEFAULT_TABLE();

template <class K, class V, int CENTILE, int DECTILE>
void CreateTableImpl(TableWrapperBase<K, V>** pptable, size_t init_size,
                     size_t runtime_dim) {
  CREATE_TABLE_ALL_BRANCHES(CENTILE, DECTILE);
}

#define DEFINE_CREATE_TABLE(K, V, CENTILE, DECTILE)                           \
  void CreateTable(size_t init_size, size_t runtime_dim,                      \
                   TableWrapperBase<K, V>** pptable) {                        \
    CreateTableImpl<K, V, CENTILE, DECTILE>(pptable, init_size, runtime_dim); \
  }

#define DECLARE_CREATE_TABLE(K, V)                       \
  void CreateTable(size_t init_size, size_t runtime_dim, \
                   TableWrapperBase<K, V>** pptable)

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

}  // namespace cpu
}  // namespace lookup
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_CPU_H_
