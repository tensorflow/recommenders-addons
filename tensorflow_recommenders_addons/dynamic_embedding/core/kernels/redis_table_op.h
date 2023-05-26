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

#ifndef TFRA_CORE_KERNELS_REDIS_LOOKUP_TABLE_OP_H_
#define TFRA_CORE_KERNELS_REDIS_LOOKUP_TABLE_OP_H_

#include <sw/redis++/connection.h>
#include <sw/redis++/connection_pool.h>
#include <sw/redis++/redis++.h>
#include <sw/redis++/sentinel.h>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/lookup_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

namespace tensorflow {
namespace recommenders_addons {
namespace redis_table {

template <class V, size_t DIM>
using ValueArray = std::array<V, DIM>;

template <class V>
using Flat1D = typename tensorflow::TTypes<V>::Flat;

template <class V>
using Tensor2D = typename tensorflow::TTypes<V, 2>::Tensor;

template <class V>
using ConstFlat1D = const typename tensorflow::TTypes<V>::ConstFlat;

template <class V>
using ConstTensor2D = const typename tensorflow::TTypes<V, 2>::ConstTensor;

using tensorflow::OpKernelContext;
using tensorflow::lookup::CheckTableDataTypes;
using tensorflow::lookup::LookupInterface;

template <class Container, class key_dtype, class value_dtype>
class HashTableOp : public OpKernel {
 public:
  explicit HashTableOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_set_(false) {
    if (ctx->output_type(0) == DT_RESOURCE) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(tensorflow::DT_RESOURCE,
                                        tensorflow::TensorShape({}), &table_));
    } else {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(tensorflow::DT_STRING,
                                        tensorflow::TensorShape({2}), &table_));
    }
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    if (!table_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator =
        [ctx, this](LookupInterface** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          LookupInterface* container = new Container(ctx, this);
          if (!ctx->status().ok()) {
            container->Unref();
            return ctx->status();
          }
          if (ctx->track_allocations()) {
            ctx->record_persistent_memory_allocation(container->MemoryUsed() +
                                                     table_.AllocatedBytes());
          }
          *ret = container;
          return TFOkStatus;
        };

    LookupInterface* table = nullptr;
    OP_REQUIRES_OK(
        ctx,
        cinfo_.resource_manager()->template LookupOrCreate<LookupInterface>(
            cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, CheckTableDataTypes(
                            *table, DataTypeToEnum<key_dtype>::v(),
                            DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      if (!table_set_) {
        auto h = table_.template scalar<ResourceHandle>();
        h() = MakeResourceHandle<LookupInterface>(ctx, cinfo_.container(),
                                                  cinfo_.name());
      }
      ctx->set_output(0, table_);
    } else {
      if (!table_set_) {
        auto h = table_.template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      ctx->set_output_ref(0, &mu_, &table_);
    }
    table_set_ = true;
  }

  ~HashTableOp() override {
    if (table_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<LookupInterface>(cinfo_.container(),
                                                  cinfo_.name())
               .ok()) {
      }
    }
  }

 private:
  mutex mu_;
  Tensor table_ TF_GUARDED_BY(mu_);
  bool table_set_ TF_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(HashTableOp);
};

}  // namespace redis_table
}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_REDIS_LOOKUP_TABLE_OP_H_
