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

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <csignal>
#include <type_traits>
#include <utility>

extern "C" {
#include <hiredis/sds.h>
}
#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

#include "redis_impl/redis_cluster_connection_pool.hpp"
#include "redis_impl/redis_connection_pool.hpp"
#include "redis_table_op.h"

// constexpr int kv_pairs_default_size = 1024*8;
/*
In the code, Redis limits the size of arguments that command can set to
1024*1024. For example, mset can only set 524287 {(1024*1024-2)/2} keys every
times. The source code is shown in the following link:
https://github.com/redis/redis/blob/be6ce8a92a9acbecfaaa6c57a45037fc1018fefe/src/networking.c#L1851
*/
// constexpr long long multi_redis_cmd_max_argc = 1024 * 1024;
constexpr long long multi_redis_cmd_max_argc =
    1024 * 8; // For better parallelism performance

using sw::redis::OptionalString;
using sw::redis::Redis;
using sw::redis::RedisCluster;

namespace tensorflow {
namespace recommenders_addons {

namespace redis_table {

using namespace redis_connection;

template <class K, class V>
class RedisTableOfTensors final : public LookupInterface {
private:
  TensorShape value_shape_;
  size_t runtime_dim_;
  // size_t init_size_;
  std::string embedding_name;
  std::string keys_prefix_name;
  std::vector<std::string> keys_prefix_name_slices;
  std::array<unsigned char, 16> keys_prefix_name_md5;

  std::shared_ptr<RedisVirtualWrapper> _table_instance;

  std::vector<ThreadContext> threads_Find;
  std::vector<ThreadContext> threads_Insert;
  std::vector<ThreadContext> threads_Delete;

  std::vector<aiocb> IMPORT_content;
  std::vector<int> IMPORT_fds;
  std::vector<unsigned long> IMPORT_fds_sizes;
  std::vector<aiocb> EXPORT_content;
  std::vector<int> EXPORT_fds;

public:
  Redis_Connection_Params redis_connection_params;

private:
  void launchFind_parallel(OpKernelContext *context,
                           std::vector<std::string> &keys_prefix_name_slices,
                           const Tensor &keys, Tensor *values,
                           const Tensor &default_value, const int64 &total,
                           const int64 &value_dim0,
                           std::vector<ThreadContext> &threads_Find) {
    const bool is_full_default = (total == value_dim0);

    const int64 max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    if (max_parallelism > static_cast<int64>(threads_Find.size()))
      threads_Find.resize(max_parallelism);

    std::atomic_uint thread_id_a(0);
    auto shard = [this, &total, &keys_prefix_name_slices, &keys, &values,
                  &default_value, &is_full_default, &threads_Find,
                  &thread_id_a](int64 begin, int64 end) {
      const int64 max_i = std::min(total, end);
      unsigned thread_id = thread_id_a.load(std::memory_order_relaxed);
      thread_id_a.store(thread_id + 1, std::memory_order_consume);

      auto reply = _table_instance->MgetCommand(
          keys, threads_Find[thread_id], begin, max_i, keys_prefix_name_slices);

      assert(
          reply.size() ==
          redis_connection_params.storage_slice); // #define REDIS_REPLY_ARRAY 2

      _table_instance->MgetToTensor(
          values, default_value, is_full_default, threads_Find[thread_id],
          reply, begin, max_i, values->NumElements() / values->dim_size(0));
    };
    int64 slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *context->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchFind(OpKernelContext *context,
                  std::vector<std::string> &keys_prefix_name_slices,
                  const Tensor &keys, Tensor *values,
                  const Tensor &default_value, const int64 &total,
                  const int64 &value_dim0,
                  std::vector<ThreadContext> &threads_Find) {
    const bool is_full_default = (total == value_dim0);

    if (1 > threads_Find.size())
      threads_Find.resize(1);

    auto reply = _table_instance->MgetCommand(keys, threads_Find[0], 0, total,
                                               keys_prefix_name_slices);

    assert(
        reply.size() ==
        redis_connection_params.storage_slice); // #define REDIS_REPLY_ARRAY 2

    _table_instance->MgetToTensor(
        values, default_value, is_full_default, threads_Find[0], reply, 0,
        total, values->NumElements() / values->dim_size(0));
  }

  void launchInsert_parallel(OpKernelContext *context,
                             std::vector<std::string> &keys_prefix_name_slices,
                             const Tensor &keys, const Tensor &values,
                             const int64 &total,
                             std::vector<ThreadContext> &threads_Insert) {
    const int64 max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    if (max_parallelism > static_cast<int64>(threads_Insert.size()))
      threads_Insert.resize(max_parallelism);

    std::atomic_uint thread_id_a(0);
    auto shard = [this, &total, &keys_prefix_name_slices, &keys, &values,
                  &threads_Insert, &thread_id_a](int64 begin, int64 end) {
      const int64 max_i = std::min(total, end);
      unsigned thread_id = thread_id_a.load(std::memory_order_relaxed);
      thread_id_a.store(thread_id + 1, std::memory_order_consume);

      _table_instance->MsetCommand(keys, values, threads_Insert[thread_id],
                                    begin, max_i, keys_prefix_name_slices);
    };
    int64 slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *context->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchInsert(OpKernelContext *context,
                    std::vector<std::string> &keys_prefix_name_slices,
                    const Tensor &keys, const Tensor &values,
                    const int64 &total,
                    std::vector<ThreadContext> &threads_Insert) {
    if (1 > threads_Insert.size())
      threads_Insert.resize(1);

    _table_instance->MsetCommand(keys, values, threads_Insert[0], 0, total,
                                  keys_prefix_name_slices);
  }

  void launchDelete_parallel(OpKernelContext *context,
                             std::vector<std::string> &keys_prefix_name_slices,
                             const Tensor &keys, const int64 &total,
                             std::vector<ThreadContext> &threads_Delete) {
    const int64 max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    if (max_parallelism > static_cast<int64>(threads_Delete.size()))
      threads_Delete.resize(max_parallelism);

    std::atomic_uint thread_id_a(0);
    auto shard = [this, &total, &keys_prefix_name_slices, &keys,
                  &threads_Delete, &thread_id_a](int64 begin, int64 end) {
      const int64 max_i = std::min(total, end);
      unsigned thread_id = thread_id_a.load(std::memory_order_relaxed);
      thread_id_a.store(thread_id + 1, std::memory_order_consume);

      _table_instance->DelCommand(keys, threads_Delete[thread_id], begin,
                                   max_i, keys_prefix_name_slices);
    };
    int64 slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *context->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchDelete(OpKernelContext *context,
                    std::vector<std::string> &keys_prefix_name_slices,
                    const Tensor &keys, const int64 &total,
                    std::vector<ThreadContext> &threads_Delete) {
    if (1 > threads_Delete.size())
      threads_Delete.resize(1);
    _table_instance->DelCommand(keys, threads_Delete[0], 0, total,
                                 keys_prefix_name_slices);
  }

public:
  RedisTableOfTensors(OpKernelContext *ctx, OpKernel *kernel) {
    // int64 env_var = 0;
    // int64 init_size = 0;
    // std::string tmp_embedding_name;

    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));

    // //The init_size and embedding vector shape are useless for the
    // initialization of Redis.
    //
    // OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "init_size", &init_size));
    // init_size_ = static_cast<size_t>(init_size);
    // if (init_size_ == 0)
    // {
    //   Status status = ReadInt64FromEnvVar("TF_HASHTABLE_INIT_SIZE",
    //                                       kv_pairs_default_size, // 8192 KV
    //                                       pairs by default &env_var);
    //   if (!status.ok())
    //   {
    //     LOG(ERROR) << "Error parsing TF_HASHTABLE_INIT_SIZE: " << status;
    //   }
    //   init_size_ = env_var;
    // }
    //

    int revn_status;
    revn_status = ReadInt32FromEnvVar(
        "redis_connect_timeout", redis_connection_params.redis_connect_timeout,
        &redis_connection_params.redis_connect_timeout);
    if (revn_status != 0)
      LOG(INFO) << "ReadInt32FromEnvVar failed with redis_connect_timeout."
                << std::endl;
    revn_status = ReadInt32FromEnvVar(
        "redis_socket_timeout", redis_connection_params.redis_socket_timeout,
        &redis_connection_params.redis_socket_timeout);
    if (revn_status != 0)
      LOG(INFO) << "ReadInt32FromEnvVar failed with redis_socket_timeout."
                << std::endl;
    revn_status = ReadInt32FromEnvVar(
        "redis_conn_pool_size", redis_connection_params.redis_conn_pool_size,
        &redis_connection_params.redis_conn_pool_size);
    if (revn_status != 0)
      LOG(INFO) << "ReadInt32FromEnvVar failed with redis_conn_pool_size."
                << std::endl;
    revn_status = ReadInt32FromEnvVar(
        "redis_wait_timeout", redis_connection_params.redis_wait_timeout,
        &redis_connection_params.redis_wait_timeout);
    if (revn_status != 0)
      LOG(INFO) << "ReadInt32FromEnvVar failed with redis_wait_timeout."
                << std::endl;
    revn_status =
        ReadInt32FromEnvVar("redis_connection_lifetime",
                            redis_connection_params.redis_connection_lifetime,
                            &redis_connection_params.redis_connection_lifetime);
    if (revn_status != 0)
      LOG(INFO) << "ReadInt32FromEnvVar failed with redis_connection_lifetime."
                << std::endl;
    revn_status = ReadInt32FromEnvVar(
        "redis_sentinel_connect_timeout",
        redis_connection_params.redis_sentinel_connect_timeout,
        &redis_connection_params.redis_sentinel_connect_timeout);
    if (revn_status != 0)
      LOG(INFO)
          << "ReadInt32FromEnvVar failed with redis_sentinel_connect_timeout."
          << std::endl;
    revn_status =
        ReadInt32FromEnvVar("sentinel_socket_timeout",
                            redis_connection_params.sentinel_socket_timeout,
                            &redis_connection_params.sentinel_socket_timeout);
    if (revn_status != 0)
      LOG(INFO) << "ReadInt32FromEnvVar failed with sentinel_socket_timeout."
                << std::endl;

    runtime_dim_ = value_shape_.dim_size(0);

    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "embedding_name", &embedding_name));
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "redis_connection_mode",
                               &redis_connection_params.redis_connection_mode));
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "redis_master_name",
                               &redis_connection_params.redis_master_name));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_host_ip",
                                    &redis_connection_params.redis_host_ip));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_host_port",
                                    &redis_connection_params.redis_host_port));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_password",
                                    &redis_connection_params.redis_password));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_db",
                                    &redis_connection_params.redis_db));
    int tem_storage_slice = 1;
    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "storage_slice", &tem_storage_slice));
    redis_connection_params.storage_slice =
        *(reinterpret_cast<unsigned *>(&tem_storage_slice));
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "using_MD5_prefix_name",
                               &redis_connection_params.using_MD5_prefix_name));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "model_tag",
                                    &redis_connection_params.model_tag));
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "using_model_lib",
                                    &redis_connection_params.using_model_lib));
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "model_lib_abs_dir",
                               &redis_connection_params.model_lib_abs_dir));

    if (redis_connection_params.using_MD5_prefix_name) {
      const std::string &&tmp_keys_prefix_name =
          redis_connection_params.model_tag + ":" + embedding_name;
      keys_prefix_name_md5 = MD5(tmp_keys_prefix_name);

      std::string md5_string;
      char *md5_view_in_redis =
          sdscatrepr(sdsempty(),
                     reinterpret_cast<char *>(keys_prefix_name_md5.data()), 16);
      char tmp[3];
      for (int i = 0; i < 16; ++i) {
        memset(tmp, 0x00, sizeof(tmp));
        sprintf(tmp, "%02X", keys_prefix_name_md5[i]);
        md5_string += tmp;
      }
      LOG(INFO) << "Init table tensor, now prefix name for keys namespace is "
                << keys_prefix_name << ". The MD5 of prefix name for keys is "
                << md5_string
                << ". And Its characters view in redis namespace is "
                << md5_view_in_redis
                << ". This MD5 is used to store keys for distinguishing "
                   "between different model and table names"
                << std::endl;

      keys_prefix_name = std::string(
          reinterpret_cast<char *>(keys_prefix_name_md5.data()), 16);
    } else {
      keys_prefix_name =
          redis_connection_params.model_tag + ":" + embedding_name;
    }

    const unsigned &storage_slice = redis_connection_params.storage_slice;
    keys_prefix_name_slices.clear();
    keys_prefix_name_slices.reserve(storage_slice);
    for (unsigned i = 0; i < storage_slice; ++i) {
      keys_prefix_name_slices.push_back(keys_prefix_name + std::to_string(i));
    }

    // creat redis instance
    switch (redis_connection_params.redis_connection_mode) {
    case ClusterMode: {
      _table_instance = RedisWrapper<RedisCluster, K, V>::get_instance();
      _table_instance->set_params(redis_connection_params);
      _table_instance->Conn();
      break;
    }
    case SentinelMode: {
      _table_instance = RedisWrapper<Redis, K, V>::get_instance();
      _table_instance->set_params(redis_connection_params);
      _table_instance->Conn();
      break;
    }
    case StreamMode: {
      LOG(ERROR) << "Sorry! redis_connection_mode="
                 << redis_connection_params.redis_connection_mode
                 << " The Stream connection mode is still being TODO."
                 << std::endl;
      throw(std::invalid_argument(
          std::to_string(redis_connection_params.redis_connection_mode) +
          " is illegal redis_connection_mode."));
      break;
    }
    default: {
      LOG(ERROR) << "There are only three Redis connection modes, which "
                    "Cluster=0/Sentinel=1/Stream=2."
                 << std::endl;
      throw(std::invalid_argument(
          std::to_string(redis_connection_params.redis_connection_mode) +
          " is illegal redis_connection_mode."));
      break;
    }
    }

    int &&check_slices_num_return =
        _table_instance->CheckSlicesNum(keys_prefix_name);
    if (check_slices_num_return == -1) {
      LOG(ERROR)
          << "The embedding table prefix name " << keys_prefix_name
          << " has already been saved in the Redis Servers. "
          << "And its number of slices is not equal to the number you putted "
             "in the setting. "
          << "Please change the storage_slice in redis_connection_params.";
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument(
                      "storage_slice must be set properly equaling to the "
                      "slices number in the Redis, got prefix storage_slice ",
                      redis_connection_params.storage_slice));
      throw(
          std::invalid_argument("Exit without setting correct slice number."));
    }

    /*
      When there is not a corresponding table existing in Redis service and
      using_model_lib==True, try to restore from a Redis binary dump files
      which paths are directory '[model_lib_abs_dir]/[model_tag]/[name].rdb'.
    */
    if (redis_connection_params.using_model_lib) {
      if (check_slices_num_return == 0) {
        ImportValuesFromFiles(ctx);
      } else if (check_slices_num_return == 1) {
        LOG(INFO) << "There is already a corresponding table "
                  << keys_prefix_name << " existing in Redis service"
                  << std::endl;
      }
    }
  }

  ~RedisTableOfTensors() {
    _table_instance.reset();
    for (auto in_aiocb_obj : IMPORT_content) {
      free((void *)in_aiocb_obj.aio_buf);
    }
    for (auto ex_aiocb_obj : EXPORT_content) {
      free((void *)ex_aiocb_obj.aio_buf);
    }
  }

  size_t size() const override {
    return _table_instance->TableSizeInSlots(keys_prefix_name_slices);
  }

  Status Find(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
              const Tensor &default_value) override {
    const static int64 value_dim = value_shape_.dim_size(0);
    int64 total = keys.dim_size(0);

    if (total < (multi_redis_cmd_max_argc - 1)) {
      launchFind(ctx, keys_prefix_name_slices, keys, values, default_value,
                 total, value_dim, threads_Find);
    } else {
      // redis commmand args > multi_redis_cmd_max_argc
      launchFind_parallel(ctx, keys_prefix_name_slices, keys, values,
                          default_value, total, value_dim, threads_Find);
    }

    return Status::OK();
  }

  Status DoInsert(bool clear, OpKernelContext *ctx, const Tensor &keys,
                  const Tensor &values) {
    // int64 value_dim = value_shape_.dim_size(0);
    int64 total = keys.dim_size(0);

    if (clear) {
      _table_instance->RemoveHkeysInSlots(keys_prefix_name_slices);
    }
    if (total < (multi_redis_cmd_max_argc - 1)) {
      launchInsert(ctx, keys_prefix_name_slices, keys, values, total,
                   threads_Insert);
    } else {
      launchInsert_parallel(
          ctx, keys_prefix_name_slices, keys, values, total,
          threads_Insert); // redis commmand args > multi_redis_cmd_max_argc
    }

    return Status::OK();
  }

  Status Insert(OpKernelContext *ctx, const Tensor &keys,
                const Tensor &values) override {
    return DoInsert(false, ctx, keys, values);
  }

  Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
    int64 total = keys.dim_size(0);
    if (total < (multi_redis_cmd_max_argc - 1)) {
      launchDelete(ctx, keys_prefix_name_slices, keys, total, threads_Delete);
    } else {
      // redis commmand args > multi_redis_cmd_max_argc
      launchDelete_parallel(ctx, keys_prefix_name_slices, keys, total,
                            threads_Delete);
    }

    return Status::OK();
  }

  Status Clear(OpKernelContext *ctx) {
    _table_instance->RemoveHkeysInSlots(keys_prefix_name_slices);
    return Status::OK();
  }

  Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                      const Tensor &values) override {
    if (redis_connection_params.using_model_lib) {
      return ImportValuesFromFiles(ctx);
    } else {
      return DoInsert(true, ctx, keys, values);
    }
  }

  Status ImportValuesFromFiles(OpKernelContext *ctx) {
    std::string file_path, folder_dir;
    const unsigned &storage_slice = redis_connection_params.storage_slice;

    IMPORT_content.resize(storage_slice);
    IMPORT_fds.reserve(storage_slice);
    IMPORT_fds.clear();
    IMPORT_fds_sizes.reserve(storage_slice);
    IMPORT_fds_sizes.clear();

    folder_dir = check_dir(redis_connection_params.model_lib_abs_dir +
                           redis_connection_params.model_tag + "/");
    for (unsigned i = 0; i < storage_slice; ++i) {
      file_path = folder_dir + keys_prefix_name_slices[i] + ".rdb";
      if (access(file_path.c_str(), 0) == -1) {
        LOG(WARNING) << "file " << file_path
                     << " doesn't exist. Using the table that already exist in "
                        "the Redis or creating a new one"
                     << std::endl;
      } else {
        IMPORT_fds.push_back(open(file_path.c_str(), O_RDONLY));
        IMPORT_fds_sizes.push_back(get_file_size(file_path));
      }
    }

    if (IMPORT_fds.size() > 0) {
      LOG(INFO) << "Try to restore the table " << keys_prefix_name
                << " to Redis service from "
                << folder_dir + keys_prefix_name + "[*].rdb" << std::endl;

      _table_instance->RestoreFromDisk(keys_prefix_name_slices,
                                         IMPORT_content, IMPORT_fds,
                                         IMPORT_fds_sizes);
      for (auto fd : IMPORT_fds)
        close(fd);
    }

    return Status::OK();
  }

  Status ExportValues(OpKernelContext *ctx) override {
    if (redis_connection_params.using_model_lib) {
      return ExportValuesToFiles(ctx);
    } else {
      int64 value_dim = value_shape_.dim_size(0);
      int64 total_size = 0;

      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          keys_replies =
              _table_instance->GetKeysInHkeys(keys_prefix_name_slices);
      for (size_t i = 0; i < keys_replies.size(); ++i) {
        total_size += keys_replies[i]->elements;
      }

      Tensor *keys;
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "keys", TensorShape({reinterpret_cast<int64>(total_size)}), &keys));

      K *pk_raw = reinterpret_cast<K *>(keys->data());

      // float percent = 0;
      // std::cout << "Write key tensor in order, now up to ......";
      redisReply *temp_reply;
      for (size_t i = 0; i < keys_replies.size(); ++i) {
        for (size_t j = 0; j < keys_replies[i]->elements; ++j) {
          // std::cout.width(3);
          // std::cout << percent << "%";
          temp_reply = keys_replies[i]->element[j];
          if (temp_reply->type ==
              REDIS_REPLY_STRING) { // #define REDIS_REPLY_STRING 1
            ReplyMemcpyToKeyTensor<K>(
                pk_raw, temp_reply->str,
                temp_reply->len); // Direct access to Tensor data in TensorFlow
          }
          // percent = round(100 * static_cast<float>((i+1) * (j+1)) /
          // static_cast<float>(total_size)); std::cout << "\b\b\b\b";
          ++pk_raw;
        }
      }

      Tensor *values;
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "values", TensorShape({total_size, value_dim}), &values));

      if (1 > threads_Find.size())
        threads_Find.resize(1);

      auto reply = _table_instance->MgetCommand(
          *keys, threads_Find[0], 0, total_size, keys_prefix_name_slices);

      assert(
          reply.size() ==
          redis_connection_params.storage_slice); // #define REDIS_REPLY_ARRAY 2

      _table_instance->MgetToTensor(values, *values, true, threads_Find[0],
                                      reply, 0, total_size, value_dim);

      return Status::OK();
    }
  }

  Status ExportValuesToFiles(OpKernelContext *ctx) {
    std::string file_path, folder_dir;
    const unsigned &storage_slice = redis_connection_params.storage_slice;
    int tem_fd;

    EXPORT_content.resize(storage_slice);
    EXPORT_fds.reserve(storage_slice);
    EXPORT_fds.clear();

    folder_dir = check_dir(redis_connection_params.model_lib_abs_dir +
                           redis_connection_params.model_tag + "/");
    for (unsigned i = 0; i < storage_slice; ++i) {
      file_path = folder_dir + keys_prefix_name_slices[i] + ".rdb";
      tem_fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0777);
      if (tem_fd < 0) {
        LOG(ERROR) << "File " + file_path +
                          " can't be created. Maybe it's already exist!"
                   << std::endl;
        time_t totalseconds = time(NULL);
        struct tm *st = localtime(&totalseconds);
        char tmp_time_str[20];
        sprintf(tmp_time_str, "%04d-%02d-%02d-%02d:%02d:%02d",
                st->tm_year + 1900, st->tm_mon + 1, st->tm_mday, st->tm_hour,
                st->tm_min, st->tm_sec);
        file_path = file_path + "." + tmp_time_str;
        tem_fd = open(file_path.c_str(), O_WRONLY | O_CREAT, 0777);
        if (tem_fd > 0) {
          LOG(WARNING) << "Now the data will be dumped to the file "
                       << file_path << " for instead." << std::endl;
          EXPORT_fds.push_back(tem_fd);
        } else {
          LOG(ERROR) << "Can't not the file " << file_path
                     << " for instead. Something bad happens" << std::endl;
        }
      } else {
        EXPORT_fds.push_back(tem_fd);
      }
    }

    if (EXPORT_fds.size() > 0) {
      LOG(INFO) << "Try to dump the table " << keys_prefix_name
                << " from Redis service to "
                << folder_dir + keys_prefix_name + "[*].rdb" << std::endl;

      _table_instance->DumpToDisk(keys_prefix_name_slices, EXPORT_content,
                                    EXPORT_fds);
      // for (auto fd : EXPORT_fds) // for now the writting may be not finished
      //   close(fd);
    }

    return Status::OK();
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return value_shape_; }

  int64 MemoryUsed() const override {
    int64 ret = 0;
    ret = (int64)(size() * (sizeof(K) + sizeof(V)));
    return sizeof(RedisTableOfTensors) + ret;
  }
};

class HashTableOpKernel : public OpKernel {
public:
  explicit HashTableOpKernel(OpKernelConstruction *ctx)
      : OpKernel(ctx),
        expected_input_0_(ctx->input_type(0) == DT_RESOURCE ? DT_RESOURCE
                                                            : DT_STRING_REF) {}

protected:
  Status LookupResource(OpKernelContext *ctx, const ResourceHandle &p,
                        LookupInterface **value) {
    return ctx->resource_manager()->Lookup<LookupInterface, false>(
        p.container(), p.name(), value);
  }
  Status GetResourceHashTable(StringPiece input_name, OpKernelContext *ctx,
                              LookupInterface **table) {
    const Tensor *handle_tensor;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
    const ResourceHandle &handle = handle_tensor->scalar<ResourceHandle>()();
    return this->LookupResource(ctx, handle, table);
  }
  Status GetTable(OpKernelContext *ctx, LookupInterface **table) {
    if (expected_input_0_ == DT_RESOURCE) {
      return this->GetResourceHashTable("table_handle", ctx, table);
    } else {
      return GetReferenceLookupTable("table_handle", ctx, table);
    }
  }

  const DataType expected_input_0_;
};

class HashTableFindOp : public HashTableOpKernel {
public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
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

// Table insert op.
class HashTableInsertOp : public HashTableOpKernel {
public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
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
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->Insert(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// Table remove op.
class HashTableRemoveOp : public HashTableOpKernel {
public:
  using HashTableOpKernel::HashTableOpKernel;

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
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// Table clear op.
template <class K, class V> class HashTableClearOp : public HashTableOpKernel {
public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    redis_table::RedisTableOfTensors<K, V> *redis_table =
        dynamic_cast<redis_table::RedisTableOfTensors<K, V> *>(table);

    int64 memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, redis_table->Clear(ctx));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

// Op that returns the size of the given table.
class HashTableSizeOp : public HashTableOpKernel {
public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    Tensor *out;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("size", TensorShape({}), &out));
    out->flat<int64>().setConstant(table->size());
  }
};

// Op that outputs tensors of all keys and all values.
class HashTableExportOp : public HashTableOpKernel {
public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    OP_REQUIRES_OK(ctx, table->ExportValues(ctx));
  }
};

// Clear the table and insert data.
class HashTableImportOp : public HashTableOpKernel {
public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
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
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx, table->ImportValues(ctx, keys, values));
    if (ctx->track_allocations()) {
      ctx->record_persistent_memory_allocation(table->MemoryUsed() -
                                               memory_used_before);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(RedisTableFind)).Device(DEVICE_CPU),
                        HashTableFindOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RedisTableInsert)).Device(DEVICE_CPU),
    HashTableInsertOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RedisTableRemove)).Device(DEVICE_CPU),
    HashTableRemoveOp);
REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(RedisTableSize)).Device(DEVICE_CPU),
                        HashTableSizeOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RedisTableExport)).Device(DEVICE_CPU),
    HashTableExportOp);
REGISTER_KERNEL_BUILDER(
    Name(PREFIX_OP_NAME(RedisTableImport)).Device(DEVICE_CPU),
    HashTableImportOp);

// Register the CuckooMutableHashTableOfTensors op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                                \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name(PREFIX_OP_NAME(RedisTableOfTensors))                                \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<key_dtype>("key_dtype")                              \
          .TypeConstraint<value_dtype>("value_dtype"),                         \
      HashTableOp<redis_table::RedisTableOfTensors<key_dtype, value_dtype>,    \
                  key_dtype, value_dtype>);                                    \
  REGISTER_KERNEL_BUILDER(Name(PREFIX_OP_NAME(RedisTableClear))                \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<key_dtype>("key_dtype")          \
                              .TypeConstraint<value_dtype>("value_dtype"),     \
                          redis_table::HashTableClearOp<key_dtype, value_dtype>)

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

} // namespace redis_table
} // namespace recommenders_addons
} // namespace tensorflow
