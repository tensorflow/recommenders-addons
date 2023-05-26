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

#ifndef EV_CORE_KERNELS_EMBEDDING_VAR_H_
#define EV_CORE_KERNELS_EMBEDDING_VAR_H_
#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/status.h"

/* After TensorFlow version 2.10.0, "Status::OK()" upgraded to "OkStatus()".
This code is for compatibility.*/
#if TF_VERSION_INTEGER >= 2100
#define TFOkStatus OkStatus()
#else
#define TFOkStatus Status::OK()
#endif

namespace tensorflow {
namespace {

template <class K, class V>
class EmbeddingVar : public ResourceBase {
 public:
  EmbeddingVar(const string& name, Allocator* alloc = cpu_allocator())
      : name_(name), value_len_(0), default_value_(NULL), alloc_(alloc) {}

  Status Init(const Tensor& default_tensor, const Tensor& empty_key_tensor) {
    dense_hash_map_.max_load_factor(0.8);
    auto empty_key_tensor_scalar = empty_key_tensor.scalar<K>();
    dense_hash_map_.set_empty_key(empty_key_tensor_scalar());
    if (default_tensor.dims() != 1) {
      return errors::InvalidArgument("EV's default_tensor shape must be 1-D");
    } else if (DataTypeToEnum<V>::v() != default_tensor.dtype()) {
      return errors::InvalidArgument(
          "EV's default_tensor DTYPE must be same as Value Type");
    } else {
      value_len_ = default_tensor.NumElements();
      default_value_ = TypedAllocator::Allocate<V>(alloc_, value_len_,
                                                   AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      memcpy(default_value_, &default_tensor_flat(0),
             default_tensor.TotalBytes());
      return TFOkStatus;
    }
  }

  string DebugString() const { return name_; }

  int64 ValueLen() const { return value_len_; }

  int64 Size() const {
    tf_shared_lock l(mu_);
    return dense_hash_map_.size();
  }

  void SetInitialized() { is_initialized_ = true; }

  bool IsInitialized() const { return is_initialized_; }

  typename TTypes<V>::Flat flat(K key, int64 global_step_scalar = -1) {
    V* val = LookupOrCreate(key, default_value_, global_step_scalar);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  V* LookupOrCreate(K key, V* default_v, int64 global_step = -1) {
    V* val = NULL;
    Status s = DoLookup(key, &val);
    if (!s.ok()) {
      V* new_val = TypedAllocator::Allocate<V>(alloc_, value_len_,
                                               AllocationAttributes());
      memcpy(new_val, default_v, sizeof(V) * value_len_);
      if (TFOkStatus != DoInsert(key, new_val, &val)) {
        TypedAllocator::Deallocate<V>(alloc_, new_val, value_len_);
      } else {
        val = new_val;
      }
    }
    // TODO update ev-version using global_step
    return val;
  }

  Status DoLookup(K key, V** val) {
    tf_shared_lock l(mu_);
    auto iter = dense_hash_map_.find(key);
    if (iter == dense_hash_map_.end()) {
      return errors::NotFound("Unable to find Key: ", key, " in DenseHashMap.");
    } else {
      *val = iter->second;
      return TFOkStatus;
    }
  }

  Status DoInsert(K key, const V* val, V** exist_val) {
    mutex_lock l(mu_);
    auto iter = dense_hash_map_.find(key);
    if (iter == dense_hash_map_.end()) {
      dense_hash_map_.insert(
          std::move(std::pair<K, V*>(key, const_cast<V*>(val))));
    } else {
      *exist_val = iter->second;
      return errors::AlreadyExists("already exists Key: ", key,
                                   " in DenseHashMap.");
    }
    return TFOkStatus;
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V*>* value_list) {
    tf_shared_lock l(mu_);
    int64 tot_size = dense_hash_map_.size();
    for (const auto it : dense_hash_map_) {
      key_list->push_back(it.first);
      value_list->push_back(it.second);
    }
    return tot_size;
  }

  mutex* mu() { return &mu_; }

 private:
  std::string name_;
  mutable mutex mu_;
  google::dense_hash_map<K, V*> dense_hash_map_;

  int64 value_len_;
  V* default_value_;
  Allocator* alloc_;
  bool is_initialized_ = false;

  ~EmbeddingVar() override {}
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace
}  // namespace tensorflow

#ifdef TFOkStatus
#undef TFOkStatus
#endif

#endif  // EV_CORE_KERNELS_EMBEDDING_VAR_H_
