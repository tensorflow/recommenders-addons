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

#ifndef EV_CORE_KERNELS_EV_OP_HELPERS_H_
#define EV_CORE_KERNELS_EV_OP_HELPERS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow_recommenders_addons/embedding_variable/core/kernels/embedding_var.h"

namespace tensorflow {

template <typename K, typename V>
mutex* GetTrainingEmbeddingVariableMutex(OpKernelContext* ctx, int input,
                                         EmbeddingVar<K, V>** maybe_resource) {
  *maybe_resource = nullptr;
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    if (LookupResource(ctx, HandleFromInput(ctx, input), maybe_resource).ok()) {
      return (*maybe_resource)->mu();
    } else {
      ctx->CtxFailureWithWarning(
          errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return ctx->input_ref_mutex(input);
}

// Utility structure that releases a sequence of borrowed mutexes when it is
// deleted.
template <typename K, typename V>
struct EmbeddingVariableInputLockHolder {
 public:
  EmbeddingVariableInputLockHolder(
      std::vector<EmbeddingVar<K, V>*> vars,
      std::unique_ptr<std::vector<mutex_lock>> locks)
      : vars_(std::move(vars)), locks_(std::move(locks)) {}

  EmbeddingVariableInputLockHolder(EmbeddingVariableInputLockHolder&& other)
      : vars_(std::move(other.vars_)), locks_(std::move(other.locks_)) {}

  ~EmbeddingVariableInputLockHolder() {
    // Release the locks before unreffing the Vars, because each lock
    // is potentially borrowed from a Var in vars_.
    locks_.reset();
    for (EmbeddingVar<K, V>* var : vars_) {
      var->Unref();
    }
  }

 private:
  std::vector<EmbeddingVar<K, V>*> vars_;
  // NOTE: Use a `std::unique_ptr` instead of moving in a vector directly,
  // because a `std::vector<mutex_lock>` is not movable on all platforms.
  std::unique_ptr<std::vector<mutex_lock>> locks_;
};

template <typename K, typename V>
EmbeddingVariableInputLockHolder<K, V>
MaybeLockEmbeddingVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, const std::vector<int>& input_ids) {
  if (!do_lock) {
    return EmbeddingVariableInputLockHolder<K, V>({}, {});
  }
  std::vector<EmbeddingVar<K, V>*> vars;
  std::vector<mutex*> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids) {
    EmbeddingVar<K, V>* var;
    mutex* mutex = GetTrainingEmbeddingVariableMutex(ctx, input, &var);
    if (var) vars.push_back(var);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  std::unique_ptr<std::vector<mutex_lock>> locks =
      MakeUnique<std::vector<mutex_lock>>();
  locks->reserve(acquire_order.size());

  for (auto input : acquire_order) {
    EmbeddingVar<K, V>* var;
    mutex* mu = GetTrainingEmbeddingVariableMutex(ctx, input, &var);
    core::ScopedUnref scoped_unref(var);
    if (mu != nullptr) {
      locks->emplace_back(*mu);
    }
  }
  return EmbeddingVariableInputLockHolder<K, V>(std::move(vars),
                                                std::move(locks));
}

}  // end namespace tensorflow

#endif  // EV_CORE_KERNELS_EV_OP_HELPERS_H_
