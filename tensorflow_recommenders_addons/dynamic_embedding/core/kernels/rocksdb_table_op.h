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

#ifndef TFRA_CORE_KERNELS_ROCKSDB_TABLE_H_
#define TFRA_CORE_KERNELS_ROCKSDB_TABLE_H_

#include "tensorflow/core/kernels/lookup_table_op.h"

namespace tensorflow {
  namespace recommenders_addons {
    namespace rocksdb_lookup {

      using tensorflow::lookup::LookupInterface;

      class ClearableLookupInterface : public LookupInterface {
      public:
        virtual Status Clear(OpKernelContext *ctx) = 0;
      };

      template<class Container, class key_dtype, class value_dtype>
      class RocksDBTableOp : public OpKernel {
      public:
        explicit RocksDBTableOp(OpKernelConstruction *ctx)
        : OpKernel(ctx), table_handle_set_(false) {
          if (ctx->output_type(0) == DT_RESOURCE) {
            OP_REQUIRES_OK(ctx, ctx->allocate_persistent(
              tensorflow::DT_RESOURCE, tensorflow::TensorShape({}),
              &table_handle_, nullptr
            ));
          }
          else {
            OP_REQUIRES_OK(ctx, ctx->allocate_persistent(
              tensorflow::DT_STRING, tensorflow::TensorShape({2}),
              &table_handle_, nullptr
            ));
          }

          OP_REQUIRES_OK(ctx, ctx->GetAttr("use_node_name_sharing", &use_node_name_sharing_));
        }

        void Compute(OpKernelContext *ctx) override {
          mutex_lock l(mu_);

          if (!table_handle_set_) {
            OP_REQUIRES_OK(ctx, cinfo_.Init(
              ctx->resource_manager(), def(), use_node_name_sharing_
            ));
          }

          auto creator = [ctx, this](LookupInterface **ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            LookupInterface *container = new Container(ctx, this);
            if (!ctx->status().ok()) {
              container->Unref();
              return ctx->status();
            }
            if (ctx->track_allocations()) {
              ctx->record_persistent_memory_allocation(
                container->MemoryUsed() + table_handle_.AllocatedBytes()
              );
            }
            *ret = container;
            return Status::OK();
          };

          LookupInterface *table = nullptr;
          OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<LookupInterface>(
            cinfo_.container(), cinfo_.name(), &table, creator
          ));
          core::ScopedUnref unref_me(table);

          OP_REQUIRES_OK(ctx, CheckTableDataTypes(
            *table, DataTypeToEnum<key_dtype>::v(), DataTypeToEnum<value_dtype>::v(), cinfo_.name()
          ));

          if (ctx->expected_output_dtype(0) == DT_RESOURCE) {
            if (!table_handle_set_) {
              auto h = table_handle_.AccessTensor(ctx)->scalar<ResourceHandle>();
              h() = MakeResourceHandle<LookupInterface>(
                ctx, cinfo_.container(), cinfo_.name()
              );
            }
            ctx->set_output(0, *table_handle_.AccessTensor(ctx));
          }
          else {
            if (!table_handle_set_) {
              auto h = table_handle_.AccessTensor(ctx)->template flat<tstring>();
              h(0) = cinfo_.container();
              h(1) = cinfo_.name();
            }
            ctx->set_output_ref(0, &mu_, table_handle_.AccessTensor(ctx));
          }

          table_handle_set_ = true;
        }

        ~RocksDBTableOp() override {
          if (table_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
            if (!cinfo_.resource_manager()->Delete<LookupInterface>(
              cinfo_.container(), cinfo_.name()
            ).ok()) {
              // Took this over from other code, what should we do here?
            }
          }
        }

      private:
          mutex mu_;
          PersistentTensor table_handle_ TF_GUARDED_BY(mu_);
          bool table_handle_set_ TF_GUARDED_BY(mu_);
          ContainerInfo cinfo_;
          bool use_node_name_sharing_;

          TF_DISALLOW_COPY_AND_ASSIGN(RocksDBTableOp);
      };

      /* --- OP KERNELS ------------------------------------------------------------------------- */
      class RocksDBTableOpKernel : public OpKernel {
      public:
        explicit RocksDBTableOpKernel(OpKernelConstruction *ctx)
          : OpKernel(ctx)
          , expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE : DT_STRING_REF) {
        }

      protected:
        Status LookupResource(OpKernelContext *ctx, const ResourceHandle &p, LookupInterface **value) {
          return ctx->resource_manager()->Lookup<LookupInterface, false>(
            p.container(), p.name(), value
          );
        }

        Status GetResourceHashTable(StringPiece input_name, OpKernelContext *ctx, LookupInterface **table) {
          const Tensor *handle_tensor;
          TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
          const auto &handle = handle_tensor->scalar<ResourceHandle>()();
          return LookupResource(ctx, handle, table);
        }

        Status GetTable(OpKernelContext *ctx, LookupInterface **table) {
          if (expected_input_0_ == DT_RESOURCE) {
            return GetResourceHashTable("table_handle", ctx, table);
          } else {
            return GetReferenceLookupTable("table_handle", ctx, table);
          }
        }

      protected:
        const DataType expected_input_0_;
      };

      class RocksDBTableClear : public RocksDBTableOpKernel {
      public:
        explicit RocksDBTableClear(OpKernelConstruction *ctx): RocksDBTableOpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) override {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          auto *rocksTable = dynamic_cast<ClearableLookupInterface *>(table);

          int64 memory_used_before = 0;
          if (ctx->track_allocations()) {
            memory_used_before = table->MemoryUsed();
          }
          OP_REQUIRES_OK(ctx, rocksTable->Clear(ctx));
          if (ctx->track_allocations()) {
            ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
          }
        }
      };

      class RocksDBTableExport : public RocksDBTableOpKernel {
      public:
        explicit RocksDBTableExport(OpKernelConstruction *ctx): RocksDBTableOpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) override {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
        }
      };

      class RocksDBTableFind : public RocksDBTableOpKernel {
      public:
        explicit RocksDBTableFind(OpKernelConstruction *ctx): RocksDBTableOpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) override {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(), table->value_dtype()};
          DataTypeVector expected_outputs = {table->value_dtype()};
          OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

          const Tensor &key = ctx->input(1);
          const Tensor &default_value = ctx->input(2);

          TensorShape output_shape = key.shape();
          output_shape.RemoveLastDims(table->key_shape().dims());
          output_shape.AppendShape(table->value_shape());
          Tensor *out;
          OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &out));
          OP_REQUIRES_OK(ctx, table->Find(ctx, key, out, default_value));
        }
      };

      class RocksDBTableImport : public RocksDBTableOpKernel {
      public:
        explicit RocksDBTableImport(OpKernelConstruction *ctx): RocksDBTableOpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) override {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(), table->value_dtype()};
          OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

          const Tensor &keys = ctx->input(1);
          const Tensor &values = ctx->input(2);
          OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));

          int64 memory_used_before = 0;
          if (ctx->track_allocations()) {
            memory_used_before = table->MemoryUsed();
          }
          OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
          if (ctx->track_allocations()) {
            ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
          }
        }
      };

      class RocksDBTableInsert : public RocksDBTableOpKernel {
      public:
        explicit RocksDBTableInsert(OpKernelConstruction *ctx): RocksDBTableOpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) override {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(), table->value_dtype()};
          OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

          const Tensor &keys = ctx->input(1);
          const Tensor &values = ctx->input(2);
          OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));

          int64 memory_used_before = 0;
          if (ctx->track_allocations()) {
            memory_used_before = table->MemoryUsed();
          }
          OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
          if (ctx->track_allocations()) {
            ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
          }
        }
      };

      class RocksDBTableRemove : public RocksDBTableOpKernel {
      public:
        explicit RocksDBTableRemove(OpKernelConstruction *ctx): RocksDBTableOpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) override {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype()};
          OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

          const Tensor &key = ctx->input(1);
          OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));

          int64 memory_used_before = 0;
          if (ctx->track_allocations()) {
            memory_used_before = table->MemoryUsed();
          }
          OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
          if (ctx->track_allocations()) {
            ctx->record_persistent_memory_allocation(table->MemoryUsed() - memory_used_before);
          }
        }
      };

      class RocksDBTableSize : public RocksDBTableOpKernel {
      public:
        explicit RocksDBTableSize(OpKernelConstruction *ctx): RocksDBTableOpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) override {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          Tensor *out;
          OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
          out->flat<int64>().setConstant(table->size());
        }
      };

    }  // namespace rocksdb_lookup
  }  // namespace recommenders_addons
}  // namespace tensorflow

#endif  // TFRA_CORE_KERNELS_ROCKSDB_TABLE_H_
