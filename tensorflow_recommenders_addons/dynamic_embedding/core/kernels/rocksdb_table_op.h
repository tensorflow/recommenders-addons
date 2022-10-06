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

#ifndef TFRA_CORE_KERNELS_ROCKSDB_TABLE_H_
#define TFRA_CORE_KERNELS_ROCKSDB_TABLE_H_

#include "tensorflow/core/kernels/lookup_table_op.h"

namespace tensorflow {
namespace recommenders_addons {

using tensorflow::lookup::LookupInterface;

class PersistentStorageLookupInterface : public LookupInterface {
 public:
  virtual Status Clear(OpKernelContext *ctx) = 0;
};

template <class Container, class key_dtype, class value_dtype>
class RocksDBTableOp : public OpKernel {
 public:
  explicit RocksDBTableOp(OpKernelConstruction *ctx)
      : OpKernel(ctx), table_handle_set_(false) {
    if (ctx->output_type(0) == DT_RESOURCE) {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_RESOURCE,
                                                   tensorflow::TensorShape({}),
                                                   &table_handle_));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_STRING,
                                                   tensorflow::TensorShape({2}),
                                                   &table_handle_));
    }

    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
  }

  void Compute(OpKernelContext *ctx) override {
    mutex_lock l(mu_);

    if (!table_handle_set_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                      use_node_name_sharing_));
    }

    auto creator =
        [ctx, this](LookupInterface **ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          LookupInterface *container = new Container(ctx, this);
          if (!ctx->status().ok()) {
            container->Unref();
            return ctx->status();
          }
          if (ctx->track_allocations()) {
            ctx->record_persistent_memory_allocation(
                container->MemoryUsed() + table_handle_.AllocatedBytes());
          }
          *ret = container;
          return Status::OK();
        };

    LookupInterface *table = nullptr;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()->LookupOrCreate<LookupInterface>(
                       cinfo_.container(), cinfo_.name(), &table, creator));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, CheckTableDataTypes(
                            *table, DataTypeToEnum<key_dtype>::v(),
                            DataTypeToEnum<value_dtype>::v(), cinfo_.name()));

    if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
      if (!table_handle_set_) {
        auto h = table_handle_.template scalar<ResourceHandle>();
        h() = MakeResourceHandle<LookupInterface>(ctx, cinfo_.container(),
                                                  cinfo_.name());
      }
      ctx->set_output(0, table_handle_);
    } else {
      if (!table_handle_set_) {
        auto h = table_handle_.template flat<tstring>();
        h(0) = cinfo_.container();
        h(1) = cinfo_.name();
      }
      ctx->set_output_ref(0, &mu_, &table_handle_);
    }

    table_handle_set_ = true;
  }

  ~RocksDBTableOp() override {
    if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<LookupInterface>(cinfo_.container(), cinfo_.name())
               .ok()) {
        // Took this over from other code, what should we do here?
      }
    }
  }

 private:
  mutex mu_;
  Tensor table_handle_ TF_GUARDED_BY(mu_);
  bool table_handle_set_ TF_GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  bool use_node_name_sharing_;

  TF_DISALLOW_COPY_AND_ASSIGN(RocksDBTableOp);
};

}  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_ROCKSDB_TABLE_H_
