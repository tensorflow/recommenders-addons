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

#ifndef TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_CPU_IMPL_H_
#define TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_CPU_IMPL_H_

#include <typeindex>

#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_impl/lookup_table_op_cpu.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/lib/parlayhash/concurrent_hash_map.h"

namespace tensorflow {
namespace recommenders_addons {
namespace lookup {
namespace cpu {

template <class K, class V, size_t DIM>
class TableWrapperOptimized final : public TableWrapperBase<K, V> {
private:
  using ValueType = ValueArray<V, DIM>;
  using Table = concurrent::ConcurrentFlatHashMap<K, ValueType, HybridHash<K>>;

public:
  explicit TableWrapperOptimized(size_t init_size, size_t value_dim)
      : init_size_(init_size), value_dim_(value_dim) {
    table_ = new Table(init_size);
    LOG(INFO) << "HashTable on CPU is created on optimized mode (parlayhash):"
              << " K=" << std::type_index(typeid(K)).name()
              << ", V=" << std::type_index(typeid(V)).name() << ", DIM=" << DIM
              << ", init_size=" << init_size_;
  }

  ~TableWrapperOptimized() override { delete table_; }

  bool insert_or_assign(K key, ConstTensor2D<V> &value_flat, int64 value_dim,
                        int64 index) const override {
    ValueType value_vec;
    std::copy_n(value_flat.data() + index * value_dim, value_dim,
                value_vec.begin());
    return table_->insert_or_assign(key, std::move(value_vec));
  }

  bool insert_or_assign(K *key, V *value, int64 value_dim) const override {
    assert(value_dim <= DIM);
    ValueType value_vec;
    std::copy_n(value, value_dim, value_vec.begin());
    return table_->insert_or_assign(*key, std::move(value_vec));
  }

  bool insert_or_accum(K key, ConstTensor2D<V> &value_or_delta_flat, bool exist,
                       int64 value_dim, int64 index) const override {
    ValueType value_or_delta_vec;
    std::copy_n(value_or_delta_flat.data() + index * value_dim, value_dim,
                value_or_delta_vec.begin());
    return table_->insert_or_accum(key, std::move(value_or_delta_vec), exist);
  }

  void find(const K &key, Tensor2D<V> &value_flat,
            ConstTensor2D<V> &default_flat, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    bool found = table_->find_fn(key, [&](const ValueType &value_vec) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    });
    if (!found) {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  void find(const K &key, Tensor2D<V> &value_flat,
            ConstTensor2D<V> &default_flat, bool &exist, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    exist = table_->find_fn(key, [&](const ValueType &value_vec) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    });
    if (!exist) {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  size_t dump(K *keys, V *values, const size_t search_offset,
              const size_t search_length) const override {
    size_t total_size = table_->size();
    if (search_offset > total_size || total_size == 0) {
      return 0;
    }

    const size_t value_dim = value_dim_;
    size_t skip = search_offset;
    size_t count = 0;

    table_->for_each([&](const auto &pair) {
      if (skip > 0) {
        --skip;
        return;
      }
      if (count >= search_length) {
        return;
      }
      keys[count] = pair.first;
      const ValueType &value = pair.second;
      std::copy_n(value.begin(), value_dim, values + count * value_dim);
      ++count;
    });

    return count;
  }

  size_t size() const override { return table_->size(); }

  void clear() override { table_->clear(); }

  bool erase(const K &key) override { return table_->erase(key); }

  void prefetch(const K &key) const override { table_->prefetch(key); }

private:
  size_t init_size_;
  size_t value_dim_;
  Table *table_;
};

template <class K, class V>
class TableWrapperDefault final : public TableWrapperBase<K, V> {
private:
  using ValueType = DefaultValueArray<V, 2>;
  using Table = concurrent::ConcurrentFlatHashMap<K, ValueType, HybridHash<K>>;

public:
  explicit TableWrapperDefault(size_t init_size, size_t value_dim = 0)
      : init_size_(init_size) {
    table_ = new Table(init_size);
    LOG(INFO) << "HashTable on CPU is created on default mode (parlayhash):"
              << " K=" << std::type_index(typeid(K)).name()
              << ", V=" << std::type_index(typeid(V)).name()
              << ", init_size=" << init_size_;
  }

  ~TableWrapperDefault() override { delete table_; }

  bool insert_or_assign(K key, ConstTensor2D<V> &value_flat, int64 value_dim,
                        int64 index) const override {
    ValueType value_vec;
    value_vec.reserve(value_dim);
    for (int64 j = 0; j < value_dim; j++) {
      V value = value_flat(index, j);
      value_vec.push_back(value);
    }
    return table_->insert_or_assign(key, value_vec);
  }

  bool insert_or_assign(K *key, V *value, int64 value_dim) const override {
    ValueType value_vec;
    value_vec.reserve(value_dim);
    for (int64 j = 0; j < value_dim; j++) {
      value_vec.push_back(*(value + j));
    }
    return table_->insert_or_assign(*key, value_vec);
  }

  bool insert_or_accum(K key, ConstTensor2D<V> &value_or_delta_flat, bool exist,
                       int64 value_dim, int64 index) const override {
    ValueType value_or_delta_vec;
    value_or_delta_vec.reserve(value_dim);
    for (int64 j = 0; j < value_dim; j++) {
      value_or_delta_vec.push_back(value_or_delta_flat(index, j));
    }
    return table_->insert_or_accum(key, value_or_delta_vec, exist);
  }

  void find(const K &key, typename tensorflow::TTypes<V, 2>::Tensor &value_flat,
            ConstTensor2D<V> &default_flat, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    bool found = table_->find_fn(key, [&](const ValueType &value_vec) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    });
    if (!found) {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  void find(const K &key, typename tensorflow::TTypes<V, 2>::Tensor &value_flat,
            ConstTensor2D<V> &default_flat, bool &exist, int64 value_dim,
            bool is_full_size_default, int64 index) const override {
    exist = table_->find_fn(key, [&](const ValueType &value_vec) {
      std::copy_n(value_vec.begin(), value_dim,
                  value_flat.data() + index * value_dim);
    });
    if (!exist) {
      for (int64 j = 0; j < value_dim; j++) {
        value_flat(index, j) =
            is_full_size_default ? default_flat(index, j) : default_flat(0, j);
      }
    }
  }

  size_t dump(K *keys, V *values, const size_t search_offset,
              const size_t search_length) const override {
    size_t total_size = table_->size();
    if (search_offset > total_size || total_size == 0) {
      return 0;
    }

    size_t skip = search_offset;
    size_t count = 0;

    table_->for_each([&](const auto &pair) {
      if (skip > 0) {
        --skip;
        return;
      }
      if (count >= search_length) {
        return;
      }
      const auto value_dim = pair.second.size();
      keys[count] = pair.first;
      const ValueType &value = pair.second;
      std::copy_n(value.begin(), value_dim, values + count * value_dim);
      ++count;
    });

    return count;
  }

  size_t size() const override { return table_->size(); }

  void clear() override { table_->clear(); }

  bool erase(const K &key) override { return table_->erase(key); }

  void prefetch(const K &key) const override { table_->prefetch(key); }

private:
  size_t init_size_;
  Table *table_;
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

template <class K, class V, size_t DIM> struct TableDispatcher {
  static constexpr bool IS_FIX_RANGE = (DIM <= 512);
  static constexpr bool K_IS_INT64 = std::is_same<K, int64>::value;
  static constexpr bool V_IS_TSTRING = std::is_same<V, tstring>::value;
  static constexpr bool OPTIMIZED =
      (IS_FIX_RANGE && K_IS_INT64 && !V_IS_TSTRING);
  using table_type =
      typename TableDispatcherImpl<K, V, DIM, OPTIMIZED>::table_type;
};

#define CREATE_A_TABLE(DIM)                                                    \
  do {                                                                         \
    if (runtime_dim <= (DIM + 1)) {                                            \
      using Table = typename TableDispatcher<K, V, (DIM + 1)>::table_type;     \
      *pptable = new Table(init_size, runtime_dim);                            \
      return;                                                                  \
    };                                                                         \
  } while (0)

#define CREATE_DEFAULT_TABLE()                                                 \
  do {                                                                         \
    using Table = TableWrapperDefault<K, V>;                                   \
    *pptable = new Table(init_size);                                           \
    return;                                                                    \
  } while (0)

#define CREATE_TABLE_10(PREFIX)                                                \
  do {                                                                         \
    CREATE_A_TABLE((PREFIX)*10 + 1);                                           \
    CREATE_A_TABLE((PREFIX)*10 + 3);                                           \
    CREATE_A_TABLE((PREFIX)*10 + 5);                                           \
    CREATE_A_TABLE((PREFIX)*10 + 7);                                           \
    CREATE_A_TABLE((PREFIX)*10 + 9);                                           \
  } while (0)

#define CREATE_TABLE_100(PREFIX)                                               \
  do {                                                                         \
    CREATE_TABLE_10((PREFIX)*10 + 0);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 1);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 2);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 3);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 4);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 5);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 6);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 7);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 8);                                          \
    CREATE_TABLE_10((PREFIX)*10 + 9);                                          \
  } while (0)

#define CREATE_TABLE_512()                                                     \
  do {                                                                         \
    CREATE_A_TABLE(0);                                                         \
    CREATE_TABLE_100(0);                                                       \
    CREATE_TABLE_100(1);                                                       \
    CREATE_TABLE_100(2);                                                       \
    CREATE_TABLE_100(3);                                                       \
    CREATE_TABLE_100(4);                                                       \
    CREATE_TABLE_10(50);                                                       \
    CREATE_A_TABLE(510);                                                       \
    CREATE_A_TABLE(511);                                                       \
  } while (0)

template <class K, class V, int CENTILE, int DECTILE>
void CreateTableImpl(TableWrapperBase<K, V> **pptable, size_t init_size,
                     size_t runtime_dim) {
  CREATE_TABLE_512();
  CREATE_DEFAULT_TABLE();
}

#define DEFINE_CREATE_TABLE(K, V, CENTILE, DECTILE)                            \
  void CreateTable(size_t init_size, size_t runtime_dim,                       \
                   TableWrapperBase<K, V> **pptable) {                         \
    CreateTableImpl<K, V, CENTILE, DECTILE>(pptable, init_size, runtime_dim);  \
  }

} // namespace cpu
} // namespace lookup
} // namespace recommenders_addons
} // namespace tensorflow

#endif // TFRA_CORE_KERNELS_LOOKUP_TABLE_OP_CPU_IMPL_H_
