/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TFRA_CORE_KERNELS_CUCKOO_LOOKUP_TABLE_OP_GPU_H_
#define TFRA_CORE_KERNELS_CUCKOO_LOOKUP_TABLE_OP_GPU_H_

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
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

using tensorflow::OpKernelContext;
using tensorflow::lookup::CheckTableDataTypes;
using tensorflow::lookup::LookupInterface;

template <class Container, class key_dtype, class value_dtype>
class HashTableGpuOp : public OpKernel {
 public:
  // ctx is not owned by this class.
  explicit HashTableGpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), table_set_(false) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  // ctx is not owned by this function.
  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);

    if (!table_set_) {
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_on_host(true);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(tensorflow::DT_STRING,
                                  tensorflow::TensorShape({2}), &table_, attr));

      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator =
        [ctx, this](lookup::LookupInterface** ret)
            TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              lookup::LookupInterface* container = new Container(ctx, this);
              if (!ctx->status().ok()) {
                container->Unref();
                return ctx->status();
              }
              if (ctx->track_allocations()) {
                ctx->record_persistent_memory_allocation(
                    container->MemoryUsed() + table_.AllocatedBytes());
              }
              *ret = container;
              return TFOkStatus;
            };

    lookup::LookupInterface* table = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()
                       ->template LookupOrCreate<lookup::LookupInterface>(
                           cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, lookup::CheckTableDataTypes(
                            *table, DataTypeToEnum<key_dtype>::v(),
                            DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      Tensor* handle;
#if GOOGLE_CUDA
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_on_host(true);
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, TensorShape({}), &handle, attr));
#else
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
#endif
      handle->scalar<ResourceHandle>()() =
          MakeResourceHandle<lookup::LookupInterface>(ctx, cinfo_.container(),
                                                      cinfo_.name());
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

  ~HashTableGpuOp() override {
    // If the table object was not shared, delete it.
    if (table_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<lookup::LookupInterface>(cinfo_.container(),
                                                          cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

 private:
  mutex mu_;
  Tensor table_ TF_GUARDED_BY(mu_);
  bool table_set_ TF_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(HashTableGpuOp);
};

}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_CUCKOO_LOOKUP_TABLE_OP_H_
