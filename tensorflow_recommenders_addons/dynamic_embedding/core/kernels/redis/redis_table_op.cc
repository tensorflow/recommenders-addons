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

#include <type_traits>
#include <utility>
#include <csignal>
#include <signal.h>

#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/util/work_sharder.h"

#include "redis_impl/redis_connection_pool.hpp"
#include "redis_table_op.h"

using sw::redis::OptionalString;
using sw::redis::Redis;
using sw::redis::RedisCluster;
using namespace sw::redis::redis_connection;

constexpr int kv_pairs_default_size = 1024*8;
/*
In the code, Redis limits the size of arguments that command can set to 1024*1024. 
For example, mset can only set 524287 {(1024*1024-2)/2} keys every times.
The source code is shown in the following link:
https://github.com/redis/redis/blob/be6ce8a92a9acbecfaaa6c57a45037fc1018fefe/src/networking.c#L1851
*/
constexpr long long multi_redis_cmd_max_argc = 1024 * 1024; 

namespace tensorflow
{
  namespace recommenders_addons
  {
    namespace redis_lookup
    {
      template <class K, class V>
      class RedisTableOfTensors final : public LookupInterface
      {
      private:
        TensorShape value_shape_;
        size_t runtime_dim_;
        size_t init_size_;

        std::shared_ptr<void> _table_instance;
        std::shared_ptr<void> _table_conn;

        std::vector<const char *> ptrs_Find = std::vector<const char *>(kv_pairs_default_size);
        std::vector<std::size_t> sizes_Find = std::vector<std::size_t>(kv_pairs_default_size);
        std::vector<const char *> ptrs_Insert = std::vector<const char *>(kv_pairs_default_size);
        std::vector<std::size_t> sizes_Insert = std::vector<std::size_t>(kv_pairs_default_size);

      public:
        Redis_Connection_Params redis_connection_params;

      private:
        void launchFind(OpKernelContext *context, std::shared_ptr<void> table_conn,
                        const Tensor &keys, Tensor *values, const Tensor &default_value,
                        int64 value_dim0)
        {
          auto &worker_threads2 = *context->device()->tensorflow_cpu_worker_threads();
          const int &num_threads = worker_threads2.num_threads;

          const int64 &total = keys.dim_size(0);
          const bool is_full_default = (total == value_dim0);

          if (static_cast<int64>(ptrs_Find.size()) < total + num_threads + 1) {
            ptrs_Find.resize(total + num_threads + 1/*max_num_thread for inserting mget into vector*/);
            sizes_Find.resize(total + num_threads + 1/*max_num_thread for inserting mget into vector*/);
          }

          std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
          auto cmd = [](::sw::redis::Connection &connection, const int64 &total, const Tensor &keys,
                        std::vector<const char *> &ptrs, std::vector<std::size_t> &sizes)
          {
            const int argc = total + 1;            

            static const char *redis_command = "mget";

            const char **ptrs_iter = &ptrs[0];
            *ptrs_iter = redis_command;
            ++ptrs_iter;

            // const int64 dim0_size = keys.dim_size(0);
            // const int64 elems_per_dim0 = keys.NumElements() / dim0_size;
            // const std::size_t key_byte_size = elems_per_dim0 * sizeof(K);

            const char *pk = reinterpret_cast<const char *>(keys.data());

            const auto iter_end_addr = &ptrs[argc];

            for (; ptrs_iter != iter_end_addr; pk += sizeof(K) )
            {
              *ptrs_iter = pk; // Direct access to Tensor data in TensorFlow
              ++ptrs_iter;
            }

            sizes[0] = 4;
            std::fill(sizes.begin() + 1, sizes.end(), sizeof(K));
            // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
            connection.send(argc, ptrs.data(), sizes.data());
          };

          // SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, , wait, 1, 0);
          SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, reply =, command, cmd, total, keys, ptrs_Find, sizes_Find);

          const int64 dim0_size = values->dim_size(0);
          const int64 elems_per_dim0 = values->NumElements() / dim0_size;
          const int64 tmp_size = sizeof(V) * elems_per_dim0;
          V * const values_base_ptr = reinterpret_cast<V *>(values->data());
          const V * const defautV_base_ptr = reinterpret_cast<V *>(default_value.data());
          auto shard2 = [this, &reply, &values_base_ptr, &defautV_base_ptr, &is_full_default, &total, &elems_per_dim0, &tmp_size](int64 begin2, int64 end2)
          {
            // for each ptr in ptrs_Find....
            // if ptr != redis_command:
            //    dst_idx = (ptr - keys.data()) / sizeof(K)
            //    memcpy(tensor[dst_idx * num_elements_emb * sizeof(V)], reply->element[i], tmp_size)
            const auto max_i = std::min(end2, total);
            for (auto i = begin2, offset = begin2 * elems_per_dim0; i < max_i; ++i, offset += elems_per_dim0)
            {
              if (reply->element[i]->type == 1) // #define REDIS_REPLY_STRING 1
              {
                memcpy(reinterpret_cast<void *>(values_base_ptr + offset), reply->element[i]->str, tmp_size); // Direct access to Tensor data in TensorFlow
              }
              else
              {
                if (is_full_default)
                {
                  memcpy(reinterpret_cast<void *>(values_base_ptr + offset),
                         reinterpret_cast<const void *>(defautV_base_ptr + offset),
                         tmp_size); // Direct access to Tensor data in TensorFlow
                }
                else
                {
                  memcpy(reinterpret_cast<void *>(values_base_ptr + offset),
                         reinterpret_cast<const void *>(defautV_base_ptr),
                         tmp_size); // Direct access to Tensor data in TensorFlow
                }
              }
            }
          };
          int64 slices2 = static_cast<int64>(total / num_threads) + 1;
          Shard(num_threads, worker_threads2.workers, total, slices2,
                shard2);

        }

        void launchInsert(OpKernelContext *context, std::shared_ptr<void> table_conn,
                          const Tensor &keys, const Tensor &values)
        {
          auto &worker_threads = *context->device()->tensorflow_cpu_worker_threads();
          const int &num_threads = worker_threads.num_threads;

          const int64 &total = keys.dim_size(0);

          if (static_cast<int64>(ptrs_Insert.size()) < (total << 1) + num_threads + 1) {
            ptrs_Insert.resize((total << 1) + num_threads + 1/*max_num_thread for inserting mset into vector*/);
            sizes_Insert.resize((total << 1) + num_threads + 1/*max_num_thread for inserting mset into vector*/);
          }
          
          std::atomic_llong slice_length(0);
          auto shard = [this, &total, &keys, &values, &slice_length](int64 begin, int64 end)
          {
            const int64 max_i = std::min(total, end);
            int64 vector_slice_begin = std::atomic_fetch_add_explicit(&slice_length, (max_i-begin)*2+1, std::memory_order_relaxed);
            assert(vector_slice_begin + slice_length <= ptrs_Insert.size());

            auto cmd = [](::sw::redis::Connection &connection, const int64 &total, const Tensor &keys, const Tensor &values,
                          std::vector<const char *> &ptrs, std::vector<std::size_t> &sizes, 
                          const int64 &begin, const int64 &max_i, const int64 &vector_slice_begin)
            {
              const int argc = (max_i - begin)*2 + 1;
              const static char *redis_command = "mset";
              constexpr static std::size_t redis_command_byte = 4;

              const char **ptrs_iter=&ptrs[vector_slice_begin];
              std::size_t *size_iter=&sizes[vector_slice_begin];

              *ptrs_iter = redis_command;
              ++ptrs_iter;
              *size_iter = redis_command_byte;
              ++size_iter;

              const int64 Vdim0_size = values.dim_size(0);
              //const int64 Velems_per_dim0 = values.NumElements() / Vdim0_size;
              const std::size_t V_byte_size = values.NumElements() / Vdim0_size * sizeof(V);

              const char *pk = reinterpret_cast<const char *>(keys.data());            
              const char *pv = reinterpret_cast<const char *>(values.data());

              const auto iter_end_addr = &sizes[vector_slice_begin+argc];

              for (; size_iter != iter_end_addr; pk += sizeof(K), pv += V_byte_size)
              {
                *ptrs_iter = pk; // Direct access to Tensor data in TensorFlow  (const char* == the name of the value)
                *(++ptrs_iter) = pv; // Direct access to Tensor data in TensorFlow (whatever bits of the value)
                ++ptrs_iter;
                
                *size_iter = sizeof(K);  // length of key-string
                *(++size_iter) = V_byte_size;  // number of value-bytes
                ++size_iter;
              }

              assert(ptrs[vector_slice_begin]==redis_command);
              // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
              connection.send(argc, &ptrs[vector_slice_begin], &sizes[vector_slice_begin]);
            };

            SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, , command, cmd, total, keys, values, ptrs_Insert, sizes_Insert, 
                                                                                                  begin, max_i, vector_slice_begin);
            // SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, , wait, 1, 0);
          };
          const int64 KVpairs_slices_size = multi_redis_cmd_max_argc >> 1;
          int64 slices_size = std::min( total, KVpairs_slices_size - 1 );
          Shard((total/KVpairs_slices_size)+1, worker_threads.workers, total, slices_size,
                shard);

        }

      public:
        RedisTableOfTensors(OpKernelContext *ctx, OpKernel *kernel)
        {
          int64 env_var = 0;
          int64 init_size = 0;
          OP_REQUIRES_OK(ctx,
                         GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "init_size", &init_size));
          OP_REQUIRES(
              ctx, TensorShapeUtils::IsVector(value_shape_),
              errors::InvalidArgument("Default value must be a vector, got shape ",
                                      value_shape_.DebugString()));
          //The init_size and embedding vector shape are useless for the initialization of Redis.
          init_size_ = static_cast<size_t>(init_size);
          if (init_size_ == 0)
          {
            Status status = ReadInt64FromEnvVar("TF_HASHTABLE_INIT_SIZE",
                                                kv_pairs_default_size, // 8192 KV pairs by default
                                                &env_var);
            if (!status.ok())
            {
              LOG(ERROR) << "Error parsing TF_HASHTABLE_INIT_SIZE: " << status;
            }
            init_size_ = env_var;
          }
          runtime_dim_ = value_shape_.dim_size(0);

          ptrs_Find.reserve(init_size_);
          sizes_Find.reserve(init_size_);
          ptrs_Insert.reserve(init_size_);
          sizes_Insert.reserve(init_size_);

          switch (redis_connection_params.connection_mode)
          {
          case ClusterMode:
          {
            _table_instance = RedisWrapper<RedisCluster>::get_instance();
            std::static_pointer_cast<RedisWrapper<RedisCluster>>(_table_instance)->set_params(redis_connection_params);
            _table_conn = std::static_pointer_cast<RedisWrapper<RedisCluster>>(_table_instance)->conn();
            break;
          }
          case SentinelMode:
          {
            _table_instance = RedisWrapper<Redis>::get_instance();
            std::static_pointer_cast<RedisWrapper<Redis>>(_table_instance)->set_params(redis_connection_params);
            _table_conn = std::static_pointer_cast<RedisWrapper<Redis>>(_table_instance)->conn();
            break;
          }
          case StreamMode:
          {
            std::cerr << "Sorry! connection_mode=" << redis_connection_params.connection_mode << " The Stream connection mode is still being TODO." << std::endl;
            throw(redis_connection_params.connection_mode);
            break;
          }
          default:
          {
            std::cerr << "There are only three Redis connection modes, which Cluster=0/Sentinel=1/Stream=2." << std::endl;
            throw(redis_connection_params.connection_mode);
            break;
          }
          }
        }

        ~RedisTableOfTensors()
        {
          _table_conn.reset();
          _table_instance.reset();
        }

        size_t size() const override
        {
          std::string info_result;
          SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, info_result =, info, "memory");
          auto tmp1 = strtok(const_cast<char *>(info_result.data()), "\n");
          tmp1 = strtok(NULL, "\n");
          tmp1 = strtok(tmp1, ":");
          tmp1 = strtok(NULL, ":");
          return std::stoi(tmp1);
        }

        Status Find(OpKernelContext *ctx, const Tensor &key, Tensor *value,
                    const Tensor &default_value) override
        {
          int64 value_dim = value_shape_.dim_size(0);

          launchFind(ctx, _table_conn, key, value, default_value, value_dim);

          return Status::OK();
        }

        Status DoInsert(bool clear, OpKernelContext *ctx, const Tensor &keys,
                        const Tensor &values)
        {
          // int64 value_dim = value_shape_.dim_size(0);

          if (clear)
          {
            SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, , flushall, false);
          }

          launchInsert(ctx, _table_conn, keys, values);

          return Status::OK();
        }

        Status Insert(OpKernelContext *ctx, const Tensor &keys,
                      const Tensor &values) override
        {
          return DoInsert(false, ctx, keys, values);
        }

        Status Remove(OpKernelContext *ctx, const Tensor &keys) override
        {
          // mutex_lock l(mu_);
          for (int64 i = 0; i < keys.dim_size(0); ++i)
          {
            SWITCH_REDIS_MODE(redis_connection_params.connection_mode, , del, keys.SubSlice(i).tensor_data().data());
          }
          return Status::OK();
        }

        Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                            const Tensor &values) override
        {
          return DoInsert(true, ctx, keys, values);
        }

        Status ExportValues(OpKernelContext *ctx) override
        {
          int64 value_dim = value_shape_.dim_size(0);
          int64 size;

          Tensor *keys;
          Tensor *values;

          if (redis_connection_params.connection_mode != SentinelMode)
          {
            std::string err1 = "Sorry! Cluster mode It's still being finished ExportValues function.";
            return Status(tensorflow::error::UNAVAILABLE, err1);
          }

          SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, size =, dbsize);

          TF_RETURN_IF_ERROR(
              ctx->allocate_output("keys", TensorShape({size}), &keys));
          TF_RETURN_IF_ERROR(ctx->allocate_output(
              "values", TensorShape({size, value_dim}), &values));

          std::vector<std::string> redis_keys(size);
          SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, , keys, "*", std::back_inserter(redis_keys));

          std::vector<std::string> redis_vals(size);
          SWITCH_REDIS_MODE(redis_connection_params.connection_mode, , mget, redis_keys.begin(), redis_keys.end(), std::back_inserter(redis_vals));

          auto shard = [this, keys, values, &redis_keys, &redis_vals, &size](int64 begin, int64 end)
          {
            for (int64 i = begin; i < end; ++i)
            {
              if (i >= size)
              {
                break;
              }
              keys->SubSlice(i).tensor_data() = redis_keys[i];
              values->SubSlice(i).tensor_data() = redis_vals[i];
            }
          };
          auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
          int64 slices = static_cast<int64>(size / worker_threads.num_threads) + 1;
          Shard(worker_threads.num_threads, worker_threads.workers, size, slices,
                shard);

          return Status::OK();
        }

        DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

        DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

        TensorShape key_shape() const final { return TensorShape(); }

        TensorShape value_shape() const override { return value_shape_; }

        int64 MemoryUsed() const override
        {
          int64 ret = 0;
          ret = (int64)size();
          return sizeof(RedisTableOfTensors) + ret;
        }
      };

      class HashTableOpKernel : public OpKernel
      {
      public:
        explicit HashTableOpKernel(OpKernelConstruction *ctx)
            : OpKernel(ctx),
              expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE
                                                                  : DT_STRING_REF) {}

      protected:
        Status LookupResource(OpKernelContext *ctx, const ResourceHandle &p,
                              LookupInterface **value)
        {
          return ctx->resource_manager()->Lookup<LookupInterface, false>(
              p.container(), p.name(), value);
        }
        Status GetResourceHashTable(StringPiece input_name, OpKernelContext *ctx,
                                    LookupInterface **table)
        {
          const Tensor *handle_tensor;
          TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
          const ResourceHandle &handle = handle_tensor->scalar<ResourceHandle>()();
          return this->LookupResource(ctx, handle, table);
        }
        Status GetTable(OpKernelContext *ctx, LookupInterface **table)
        {
          if (expected_input_0_ == DT_RESOURCE)
          {
            return this->GetResourceHashTable("table_handle", ctx, table);
          }
          else
          {
            return GetReferenceLookupTable("table_handle", ctx, table);
          }
        }

        const DataType expected_input_0_;
      };

      class HashTableFindOp : public HashTableOpKernel
      {
      public:
        using HashTableOpKernel::HashTableOpKernel;

        void Compute(OpKernelContext *ctx) override
        {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                            table->value_dtype()};
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

      REGISTER_KERNEL_BUILDER(Name("TFRA>RedisTableFind").Device(DEVICE_CPU),
                              HashTableFindOp);

      // Table insert op.
      class HashTableInsertOp : public HashTableOpKernel
      {
      public:
        using HashTableOpKernel::HashTableOpKernel;

        void Compute(OpKernelContext *ctx) override
        {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                            table->value_dtype()};
          OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

          const Tensor &keys = ctx->input(1);
          const Tensor &values = ctx->input(2);
          OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForInsert(keys, values));

          int64 memory_used_before = 0;
          if (ctx->track_allocations())
          {
            memory_used_before = table->MemoryUsed();
          }
          OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
          if (ctx->track_allocations())
          {
            ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                                     memory_used_before);
          }
        }
      };

      REGISTER_KERNEL_BUILDER(Name("TFRA>RedisTableInsert").Device(DEVICE_CPU),
                              HashTableInsertOp);

      // Table remove op.
      class HashTableRemoveOp : public HashTableOpKernel
      {
      public:
        using HashTableOpKernel::HashTableOpKernel;

        void Compute(OpKernelContext *ctx) override
        {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype()};
          OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

          const Tensor &key = ctx->input(1);
          OP_REQUIRES_OK(ctx, table->CheckKeyTensorForRemove(key));

          int64 memory_used_before = 0;
          if (ctx->track_allocations())
          {
            memory_used_before = table->MemoryUsed();
          }
          OP_REQUIRES_OK(ctx, table->Remove(ctx, key));
          if (ctx->track_allocations())
          {
            ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                                     memory_used_before);
          }
        }
      };

      REGISTER_KERNEL_BUILDER(Name("TFRA>RedisTableRemove").Device(DEVICE_CPU),
                              HashTableRemoveOp);

      // Op that returns the size of the given table.
      class HashTableSizeOp : public HashTableOpKernel
      {
      public:
        using HashTableOpKernel::HashTableOpKernel;

        void Compute(OpKernelContext *ctx) override
        {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          Tensor *out;
          OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
          out->flat<int64>().setConstant(table->size());
        }
      };

      REGISTER_KERNEL_BUILDER(Name("TFRA>RedisTableSize").Device(DEVICE_CPU),
                              HashTableSizeOp);

      // Op that outputs tensors of all keys and all values.
      class HashTableExportOp : public HashTableOpKernel
      {
      public:
        using HashTableOpKernel::HashTableOpKernel;

        void Compute(OpKernelContext *ctx) override
        {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
        }
      };

      REGISTER_KERNEL_BUILDER(Name("TFRA>RedisTableExport").Device(DEVICE_CPU),
                              HashTableExportOp);

      // Clear the table and insert data.
      class HashTableImportOp : public HashTableOpKernel
      {
      public:
        using HashTableOpKernel::HashTableOpKernel;

        void Compute(OpKernelContext *ctx) override
        {
          LookupInterface *table;
          OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
          core::ScopedUnref unref_me(table);

          DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                            table->value_dtype()};
          OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

          const Tensor &keys = ctx->input(1);
          const Tensor &values = ctx->input(2);
          OP_REQUIRES_OK(ctx, table->CheckKeyAndValueTensorsForImport(keys, values));

          int memory_used_before = 0;
          if (ctx->track_allocations())
          {
            memory_used_before = table->MemoryUsed();
          }
          OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
          if (ctx->track_allocations())
          {
            ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                                     memory_used_before);
          }
        }
      };

      REGISTER_KERNEL_BUILDER(Name("TFRA>RedisTableImport").Device(DEVICE_CPU),
                              HashTableImportOp);

// Register the RedisTableOfTensors op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                              \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("TFRA>RedisTableOfTensors")                                       \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<key_dtype>("key_dtype")                            \
          .TypeConstraint<value_dtype>("value_dtype"),                       \
      HashTableOp<redis_lookup::RedisTableOfTensors<key_dtype, value_dtype>, \
                  key_dtype, value_dtype>)

      REGISTER_KERNEL(int32, double);
      REGISTER_KERNEL(int32, float);
      REGISTER_KERNEL(int32, int32);
      REGISTER_KERNEL(int64, double);
      REGISTER_KERNEL(int64, float);
      REGISTER_KERNEL(int64, int32);
      REGISTER_KERNEL(int64, int64);
      REGISTER_KERNEL(int64, tstring);
      REGISTER_KERNEL(int64, int8);
      REGISTER_KERNEL(int64, Eigen::half);
      REGISTER_KERNEL(tstring, bool);
      REGISTER_KERNEL(tstring, double);
      REGISTER_KERNEL(tstring, float);
      REGISTER_KERNEL(tstring, int32);
      REGISTER_KERNEL(tstring, int64);
      REGISTER_KERNEL(tstring, int8);
      REGISTER_KERNEL(tstring, Eigen::half);

#undef REGISTER_KERNEL

#undef SWITCH_REDIS_MODE
#undef SWITCH_REDIS_MODE_noCluster

    } // namespace redis_lookup
  }   // namespace recommenders_addons
} // namespace tensorflow
