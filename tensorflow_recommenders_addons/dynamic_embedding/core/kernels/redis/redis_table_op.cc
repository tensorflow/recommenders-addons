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

extern "C"
{
#include <hiredis/sds.h>
}

#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/util/work_sharder.h"

#include "redis_impl/redis_connection_pool.hpp"
#include "redis_impl/redis_cluster_connection_pool.hpp"
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
// constexpr long long multi_redis_cmd_max_argc = 1024 * 1024; 
constexpr long long multi_redis_cmd_max_argc = 1024 * 8; // For better parallelism performance

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
        std::string embedding_name;
        std::string keys_prefix_name;
        std::vector<std::string> keys_prefix_name_slices;
        std::array<unsigned char, 16> keys_prefix_name_md5;

        std::shared_ptr<RedisVirtualWrapper> _table_instance;

        std::vector<ThreadContext> threads_Find;
        std::vector<ThreadContext> threads_Insert;

      public:
        Redis_Connection_Params redis_connection_params;

      private:
        void launchFind_parallel(OpKernelContext *context, std::vector<std::string> &keys_prefix_name_slices,
                        const Tensor &keys, Tensor *values, const Tensor &default_value,
                        const int64 &total, const int64 &value_dim0, 
                        std::vector<ThreadContext> &threads_Find)
        {
          const bool is_full_default = (total == value_dim0);

          const int64 max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

          threads_Find.reserve(max_parallelism);
          
          std::atomic_uint thread_id_a(0);
          auto shard = [this, &total, &keys_prefix_name_slices, &keys, &values, &default_value, &is_full_default, &threads_Find, &thread_id_a]
          (int64 begin, int64 end)
          {
            const int64 max_i = std::min(total, end);
            unsigned thread_id = thread_id_a.load(std::memory_order_relaxed);
            thread_id_a.store(thread_id+1, std::memory_order_consume); 

            auto reply = _table_instance->MGET_COMMAND(keys, threads_Find[thread_id], begin, max_i, keys_prefix_name_slices);
            
            assert(reply->size == redis_connection_params.storage_slice); // #define REDIS_REPLY_ARRAY 2

            _table_instance->MGET_to_Tensor(values, default_value, is_full_default, threads_Find[thread_id], reply, begin, max_i);
          };
          int64 slices_size = std::min( total, multi_redis_cmd_max_argc - 1 );
          auto &worker_threads = *context->device()->tensorflow_cpu_worker_threads();
          Shard(max_parallelism, worker_threads.workers, total, slices_size,
                shard);

        }

        void launchFind(OpKernelContext *context, std::vector<std::string> &keys_prefix_name_slices,
                        const Tensor &keys, Tensor *values, const Tensor &default_value,
                        const int64 &total, const int64 &value_dim0, 
                        std::vector<ThreadContext> &threads_Find)
        {
          const bool is_full_default = (total == value_dim0);

          threads_Find.reserve(1);
          
          auto reply = _table_instance->MGET_COMMAND(keys, threads_Find[0], 0, total, keys_prefix_name_slices);
          
          assert(reply->size == redis_connection_params.storage_slice); // #define REDIS_REPLY_ARRAY 2

          _table_instance->MGET_to_Tensor(values, default_value, is_full_default, threads_Find[0], reply, 0, total);     
        }

        void launchInsert_parallel(OpKernelContext *context, std::vector<std::string> &keys_prefix_name_slices,
                          const Tensor &keys, const Tensor &values, const int64 &total, 
                          std::vector<ThreadContext> &threads_Find)
        {
          const int64 max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

          threads_Find.reserve(max_parallelism);
          
          std::atomic_uint thread_id_a(0);
          auto shard = [this, &total, &keys_prefix_name_slices, &keys, &values, &threads_Find, &thread_id_a]
          (int64 begin, int64 end)
          {
            const int64 max_i = std::min(total, end);
            unsigned thread_id = thread_id_a.load(std::memory_order_relaxed);
            thread_id_a.store(thread_id+1, std::memory_order_consume); 

            _table_instance->MSET_COMMAND(keys, values, threads_Find[thread_id], begin, max_i, keys_prefix_name_slices);
          };
          int64 slices_size = std::min( total, multi_redis_cmd_max_argc - 1 );
          auto &worker_threads = *context->device()->tensorflow_cpu_worker_threads();
          Shard(max_parallelism, worker_threads.workers, total, slices_size,
                shard);

        }

        void launchInsert(OpKernelContext *context, std::vector<std::string> &keys_prefix_name_slices,
                          const Tensor &keys, const Tensor &values, const int64 &total, 
                          std::vector<ThreadContext> &threads_Find)
        {
          threads_Find.reserve(1);

          _table_instance->MSET_COMMAND(keys, values, threads_Find[0], 0, total, keys_prefix_name_slices);
        }

      public:
        RedisTableOfTensors(OpKernelContext *ctx, OpKernel *kernel)
        {
          int64 env_var = 0;
          int64 init_size = 0;
          std::string tmp_embedding_name;

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

          OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "embedding_name", &tmp_embedding_name));
          embedding_name = static_cast<std::string>(strtok(const_cast<char *>(tmp_embedding_name.data()), ":"));

          if(redis_connection_params.using_MD5_prefix_name)
          {
            const std::string tmp_keys_prefix_name = redis_connection_params.model_tag+":"+embedding_name;
            keys_prefix_name_md5 = ::sw::redis::redis_connection::MD5(tmp_keys_prefix_name);

            std::string md5_string;
            char *md5_view_in_redis = sdscatrepr(sdsempty(), reinterpret_cast<char*>(keys_prefix_name_md5.data()), 16);
            char tmp[3];
            for( int i = 0; i < 16; ++i )
            {   
              memset( tmp, 0x00, sizeof( tmp ) );
              sprintf( tmp, "%02X", keys_prefix_name_md5[i] );
              md5_string += tmp;
            }
            LOG(INFO) << "Init table tensor, now prefix name for keys namespace is " << keys_prefix_name << \
              ". The MD5 of prefix name for keys is " << md5_string << \
              ". And Its characters view in redis namespace is " << md5_view_in_redis << \
              ". This MD5 is used to store keys for distinguishing between different model and table names" << std::endl;

            const std::string tmp_char2string(reinterpret_cast<char*>(keys_prefix_name_md5.data()),16);
            keys_prefix_name = tmp_char2string;
          }
          else
          {
            keys_prefix_name = redis_connection_params.model_tag+":"+embedding_name;
          }
          const unsigned storage_slice = redis_connection_params.storage_slice;
          keys_prefix_name_slices.reserve(storage_slice);
          for (unsigned i = 0; i < storage_slice; ++i)
          {
            keys_prefix_name_slices[i] = keys_prefix_name + std::to_string(storage_slice);
          }

          switch (redis_connection_params.connection_mode)
          {
          case ClusterMode:
          {
            _table_instance = RedisWrapper<RedisCluster, K, V>::get_instance();
            _table_instance->set_params(redis_connection_params);
            _table_instance->conn();
            break;
          }
          case SentinelMode:
          {
            _table_instance = RedisWrapper<RedisCluster, K, V>::get_instance();
            _table_instance->set_params(redis_connection_params);
            _table_instance->conn();
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
          _table_instance.reset();
        }

        size_t size() const override
        {
          std::string info_result;
          // SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, info_result =, info, "memory");
          auto tmp1 = strtok(const_cast<char *>(info_result.data()), "\n");
          tmp1 = strtok(NULL, "\n");
          tmp1 = strtok(tmp1, ":");
          tmp1 = strtok(NULL, ":");
          return std::stoi(tmp1);
        }

        Status Find(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
                    const Tensor &default_value) override
        {
          const static int64 value_dim = value_shape_.dim_size(0);
          int64 total = keys.dim_size(0);
          // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
          if( total < (multi_redis_cmd_max_argc - 1) )
          {
            launchFind(ctx, keys_prefix_name_slices, keys, values, default_value, total, value_dim, threads_Find);
          }
          else
          {
            // redis commmand args > multi_redis_cmd_max_argc
            launchFind_parallel(ctx, keys_prefix_name_slices, keys, values, default_value, total, value_dim, threads_Find); 
          }

          return Status::OK();
        }

        Status DoInsert(bool clear, OpKernelContext *ctx, const Tensor &keys,
                        const Tensor &values)
        {
          // int64 value_dim = value_shape_.dim_size(0);
          int64 total = keys.dim_size(0);

          if (clear)
          {
            // SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, , flushall, false);
          }
          if( total < (multi_redis_cmd_max_argc - 1) )
          {
            launchInsert(ctx, keys_prefix_name_slices, keys, values, total, threads_Find);      
          }
          else
          {
            launchInsert_parallel(ctx, keys_prefix_name_slices, keys, values, total, threads_Find); // redis commmand args > multi_redis_cmd_max_argc
          }
          

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
            // SWITCH_REDIS_MODE(redis_connection_params.connection_mode, , del, keys.SubSlice(i).tensor_data().data());
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

          // SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, size =, dbsize);

          TF_RETURN_IF_ERROR(
              ctx->allocate_output("keys", TensorShape({size}), &keys));
          TF_RETURN_IF_ERROR(ctx->allocate_output(
              "values", TensorShape({size, value_dim}), &values));

          std::vector<std::string> redis_keys(size);
          // SWITCH_REDIS_MODE_noCluster(redis_connection_params.connection_mode, , keys, "*", std::back_inserter(redis_keys));

          std::vector<std::string> redis_vals(size);
          // SWITCH_REDIS_MODE(redis_connection_params.connection_mode, , mget, redis_keys.begin(), redis_keys.end(), std::back_inserter(redis_vals));

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

    } // namespace redis_lookup
  }   // namespace recommenders_addons
} // namespace tensorflow
