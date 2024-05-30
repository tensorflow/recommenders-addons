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

#include "redis_table_op.h"

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <csignal>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

#include "redis_impl/json.h"
#include "redis_impl/redis_cluster_connection_pool.hpp"
#include "redis_impl/redis_connection_pool.hpp"
#include "redis_impl/redis_table_op_util.hpp"
#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/types.h"
#include "tensorflow_recommenders_addons/dynamic_embedding/core/utils/utils.h"

// constexpr int kv_pairs_default_size = 1024*8;
/*
In the code, Redis limits the size of arguments that command can set to
1024*1024. For example, mset can only set 524287 {(1024*1024-2)/2} keys every
times. The source code is shown in the following link:
https://github.com/redis/redis/blob/be6ce8a92a9acbecfaaa6c57a45037fc1018fefe/src/networking.c#L1851
*/
// constexpr int64_t multi_redis_cmd_max_argc = 1024 * 1024;
static int64_t multi_redis_cmd_max_argc =
    128 * 8;  // For better parallelism performance

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
  int64_t runtime_value_dim_;
  // size_t init_size_;
  std::string redis_config_abs_dir;
  std::string embedding_name;
  std::string keys_prefix_name;
  std::string keys_prefix_name_import;
  std::vector<std::string> keys_prefix_name_slices;
  std::vector<std::string> keys_prefix_name_slices_import;

  std::shared_ptr<RedisBaseWrapper<K, V>> _table_instance = nullptr;

  std::vector<ThreadContext *> threads_Find;
  std::vector<ThreadContext *> threads_Insert;
  std::vector<ThreadContext *> threads_Delete;
  std::mutex threads_Find_mutex;
  std::mutex threads_Insert_mutex;
  std::mutex threads_Accum_mutex;
  std::mutex threads_Delete_mutex;

  std::vector<aiocb> IMPORT_content;
  std::vector<int> IMPORT_fds;
  std::vector<unsigned long> IMPORT_fds_sizes;
  std::vector<aiocb> EXPORT_content;
  std::vector<int> EXPORT_fds;

 public:
  Redis_Connection_Params redis_connection_params;

 private:
  void launchFind_parallel(OpKernelContext *ctx,
                           std::vector<std::string> &keys_prefix_name_slices,
                           const K *keys, V *values, const V *default_value,
                           const int64_t &total,
                           const int64_t &Velems_per_flat2_dim0,
                           const bool is_full_default,
                           std::vector<ThreadContext *> &threads_Find) {
    const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    auto shard = [this, &ctx, &total, &keys_prefix_name_slices, &keys, &values,
                  &default_value, &is_full_default, &Velems_per_flat2_dim0,
                  &threads_Find](int64_t begin, int64_t end) {
      const int64_t max_i = std::min(total, end);

      OP_REQUIRES_OK(ctx,
                     launchFindCore<K, V>(
                         _table_instance, keys_prefix_name_slices, keys, values,
                         default_value, is_full_default, Velems_per_flat2_dim0,
                         threads_Find, threads_Find_mutex, begin, max_i));
    };
    int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchFind(OpKernelContext *ctx,
                  std::vector<std::string> &keys_prefix_name_slices,
                  const K *keys, V *values, const V *default_value,
                  const int64_t &total, const int64_t &Velems_per_flat2_dim0,
                  const bool is_full_default,
                  std::vector<ThreadContext *> &threads_Find) {
    OP_REQUIRES_OK(
        ctx, launchFindCore<K, V>(_table_instance, keys_prefix_name_slices,
                                  keys, values, default_value, is_full_default,
                                  Velems_per_flat2_dim0, threads_Find,
                                  threads_Find_mutex, 0, total));
  }

  void launchFindWithExists_parallel(
      OpKernelContext *ctx, std::vector<std::string> &keys_prefix_name_slices,
      const K *keys, V *values, const V *default_value, bool *exists,
      const int64_t &total, const int64_t &Velems_per_flat2_dim0,
      const bool is_full_default, std::vector<ThreadContext *> &threads_Find) {
    const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    auto shard = [this, &ctx, &total, &keys_prefix_name_slices, &keys, &values,
                  &default_value, &exists, &is_full_default,
                  &Velems_per_flat2_dim0,
                  &threads_Find](int64_t begin, int64_t end) {
      const int64_t max_i = std::min(total, end);

      OP_REQUIRES_OK(ctx, launchFindWithExistsCore<K, V>(
                              _table_instance, keys_prefix_name_slices, keys,
                              values, default_value, exists, is_full_default,
                              Velems_per_flat2_dim0, threads_Find,
                              threads_Find_mutex, begin, max_i));
    };
    int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchFindWithExists(OpKernelContext *ctx,
                            std::vector<std::string> &keys_prefix_name_slices,
                            const K *keys, V *values, const V *default_value,
                            bool *exists, const int64_t &total,
                            const int64_t &Velems_per_flat2_dim0,
                            const bool is_full_default,
                            std::vector<ThreadContext *> &threads_Find) {
    OP_REQUIRES_OK(
        ctx, launchFindWithExistsCore<K, V>(
                 _table_instance, keys_prefix_name_slices, keys, values,
                 default_value, exists, is_full_default, Velems_per_flat2_dim0,
                 threads_Find, threads_Find_mutex, 0, total));
  }

  void launchInsert_parallel(OpKernelContext *ctx,
                             std::vector<std::string> &keys_prefix_name_slices,
                             const K *keys, const V *values,
                             const int64_t &total,
                             const int64_t &Velems_per_flat2_dim0,
                             std::vector<ThreadContext *> &threads_Insert) {
    const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    auto shard = [this, &ctx, &total, &keys_prefix_name_slices, &keys, &values,
                  &Velems_per_flat2_dim0,
                  &threads_Insert](int64_t begin, int64_t end) {
      const int64_t max_i = std::min(total, end);

      OP_REQUIRES_OK(ctx, launchInsertCore<K, V>(
                              _table_instance, keys_prefix_name_slices, keys,
                              values, Velems_per_flat2_dim0, threads_Insert,
                              threads_Insert_mutex, begin, max_i));
    };
    int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchInsert(OpKernelContext *ctx,
                    std::vector<std::string> &keys_prefix_name_slices,
                    const K *keys, const V *values, const int64_t &total,
                    const int64_t &Velems_per_flat2_dim0,
                    std::vector<ThreadContext *> &threads_Insert) {
    OP_REQUIRES_OK(ctx, launchInsertCore<K, V>(
                            _table_instance, keys_prefix_name_slices, keys,
                            values, Velems_per_flat2_dim0, threads_Insert,
                            threads_Insert_mutex, 0, total));
  }

  void launchAccum_parallel(OpKernelContext *ctx,
                            std::vector<std::string> &keys_prefix_name_slices,
                            const K *keys, const V *values_or_delta,
                            const bool *exists, const int64_t &total,
                            const int64_t &Velems_per_flat2_dim0,
                            std::string &values_dtype_str,
                            std::vector<ThreadContext *> &threads_Insert) {
    const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    auto shard = [this, &ctx, &total, &keys_prefix_name_slices, &keys,
                  &values_or_delta, &exists, &Velems_per_flat2_dim0,
                  &values_dtype_str,
                  &threads_Insert](int64_t begin, int64_t end) {
      const int64_t max_i = std::min(total, end);

      OP_REQUIRES_OK(ctx, launchAccumCore<K, V>(
                              _table_instance, keys_prefix_name_slices, keys,
                              values_or_delta, exists, Velems_per_flat2_dim0,
                              values_dtype_str, threads_Insert,
                              threads_Accum_mutex, begin, max_i));
    };
    int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchAccum(OpKernelContext *ctx,
                   std::vector<std::string> &keys_prefix_name_slices,
                   const K *keys, const V *values_or_delta, const bool *exists,
                   const int64_t &total, const int64_t &Velems_per_flat2_dim0,
                   std::string &values_dtype_str,
                   std::vector<ThreadContext *> &threads_Insert) {
    OP_REQUIRES_OK(ctx, launchAccumCore<K, V>(
                            _table_instance, keys_prefix_name_slices, keys,
                            values_or_delta, exists, Velems_per_flat2_dim0,
                            values_dtype_str, threads_Insert,
                            threads_Insert_mutex, 0, total));
  }

  void launchDelete_parallel(OpKernelContext *ctx,
                             std::vector<std::string> &keys_prefix_name_slices,
                             const K *keys, const int64_t &total,
                             std::vector<ThreadContext *> &threads_Delete) {
    const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    auto shard = [this, &ctx, &total, &keys_prefix_name_slices, &keys,
                  &threads_Delete](int64_t begin, int64_t end) {
      const int64_t max_i = std::min(total, end);

      OP_REQUIRES_OK(
          ctx, launchDeleteCore<K, V>(_table_instance, keys_prefix_name_slices,
                                      keys, threads_Delete,
                                      threads_Delete_mutex, begin, max_i));
    };
    int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  void launchDelete(OpKernelContext *ctx,
                    std::vector<std::string> &keys_prefix_name_slices,
                    const K *keys, const int64_t &total,
                    std::vector<ThreadContext *> &threads_Delete) {
    OP_REQUIRES_OK(ctx, launchDeleteCore<K, V>(
                            _table_instance, keys_prefix_name_slices, keys,
                            threads_Delete, threads_Delete_mutex, 0, total));
  }

 public:
  RedisTableOfTensors(OpKernelContext *ctx, OpKernel *kernel) {
    OP_REQUIRES_OK(ctx,
                   GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(value_shape_),
        errors::InvalidArgument("Default value must be a vector, got shape ",
                                value_shape_.DebugString()));

    OP_REQUIRES_OK(
        ctx, GetNodeAttr(kernel->def(), "embedding_name", &embedding_name));

    std::string redis_config_abs_dir_tem;
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_config_abs_dir",
                                    &redis_config_abs_dir_tem));

    std::string redis_config_abs_dir_env;
    OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_config_abs_dir_env",
                                    &redis_config_abs_dir_env));
    Status spec_config_status = ReadStringFromEnvVar(
        redis_config_abs_dir_env, "NotFound", &redis_config_abs_dir);
    if (redis_config_abs_dir != "NotFound") {
      if (get_file_size(redis_config_abs_dir) > 0) {
        LOG(INFO)
            << "Read TFRA Redis config file path from the environment variable "
            << redis_config_abs_dir_env << " successfully. Config file path is "
            << redis_config_abs_dir;
      } else {
        ctx->CtxFailure(
            errors::NotFound("Can not find the file " + redis_config_abs_dir +
                             ". Please check carefully again if the file exist "
                             "or unset the evironment viriable " +
                             redis_config_abs_dir_env));
      }
    } else {
      LOG(WARNING) << "Fails to read the TFRA Redis config file path from the "
                      "environment variable "
                   << redis_config_abs_dir_env
                   << " which read from OP attribute redis_config_abs_dir_env "
                      "firstly, now try to read config file path from global "
                      "environment variable TFRA_REDIS_CONFIG_PATH.";
      Status global_config_status = ReadStringFromEnvVar(
          "TFRA_REDIS_CONFIG_PATH", "NotFound", &redis_config_abs_dir);
      if (redis_config_abs_dir != "NotFound" &&
          get_file_size(redis_config_abs_dir) <= 0) {
        ctx->CtxFailure(errors::NotFound(
            "Can not find the file " + redis_config_abs_dir +
            ". Please check carefully again if the file exist or unset the "
            "evironment viriable TFRA_REDIS_CONFIG_PATH"));
      } else if (redis_config_abs_dir == "NotFound") {
        LOG(WARNING)
            << "Fails to read the TFRA Redis config file path from the "
               "global environment variable TFRA_REDIS_CONFIG_PATH firstly, "
               "config file path is "
            << redis_config_abs_dir
            << ". now the config file path which assigned OP attribute "
               "redis_config_abs_dir "
            << redis_config_abs_dir_tem << " is used.";
        redis_config_abs_dir = redis_config_abs_dir_tem;
      } else {
        LOG(INFO) << "TFRA Redis config file path is " << redis_config_abs_dir;
      }
    }

    if (get_file_size(redis_config_abs_dir) > 0) {
      OP_REQUIRES_OK(ctx, ParseJsonConfig(&redis_config_abs_dir,
                                          &redis_connection_params));
    } else {
      ctx->CtxFailure(errors::NotFound("Unable to find config file with path: ",
                                       redis_config_abs_dir));
    }

    const int64_t &&default_value_width = value_shape_.dim_size(0);
    const int64_t &&default_value_total = value_shape_.num_elements();
    if (default_value_width == default_value_total) {
      runtime_value_dim_ = default_value_width;
    } else {
      LOG(WARNING) << "The num_elements in default_value dosen't equal to the "
                      "dim_size(0) in default_value. Using the num_elements as "
                      "output tensor dim two now.";
      runtime_value_dim_ = default_value_total;
    }

    std::vector<std::pair<unsigned, unsigned>> cluster_slots;

    // creat redis instance
    switch (redis_connection_params.redis_connection_mode) {
      case ClusterMode: {
        multi_redis_cmd_max_argc = redis_connection_params.keys_sending_size *
                                   redis_connection_params.storage_slice;
        _table_instance = RedisWrapper<RedisCluster, K, V>::get_instance();
        OP_REQUIRES_OK(ctx,
                       _table_instance->set_params(redis_connection_params));
        if (redis_connection_params.using_hash_storage_slice) {
          OP_REQUIRES_OK(ctx, _table_instance->set_K_bucket_num_handle(
                                  KBucketNumCRC32Handle));
        } else {
          OP_REQUIRES_OK(ctx, _table_instance->set_K_bucket_num_handle(
                                  KBucketNumCommonHandle<K>));
        }
        OP_REQUIRES_OK(ctx, _table_instance->Conn());
        if (redis_connection_params.redis_hash_tags_hypodispersion == false)
          cluster_slots = _table_instance->ClusterNodesSlots(false);
        break;
      }
      case SentinelMode: {
        multi_redis_cmd_max_argc =
            redis_connection_params.keys_sending_size * 1;
        _table_instance = RedisWrapper<Redis, K, V>::get_instance();
        OP_REQUIRES_OK(ctx,
                       _table_instance->set_params(redis_connection_params));
        if (redis_connection_params.using_hash_storage_slice) {
          OP_REQUIRES_OK(ctx, _table_instance->set_K_bucket_num_handle(
                                  KBucketNumCRC32Handle));
        } else {
          OP_REQUIRES_OK(ctx, _table_instance->set_K_bucket_num_handle(
                                  KBucketNumCommonHandle<K>));
        }
        OP_REQUIRES_OK(ctx, _table_instance->Conn());
        break;
      }
      case StandaloneMode: {
        multi_redis_cmd_max_argc =
            redis_connection_params.keys_sending_size * 1;
        _table_instance = RedisWrapper<Redis, K, V>::get_instance(false);
        OP_REQUIRES_OK(ctx,
                       _table_instance->set_params(redis_connection_params));
        if (redis_connection_params.using_hash_storage_slice) {
          OP_REQUIRES_OK(ctx, _table_instance->set_K_bucket_num_handle(
                                  KBucketNumCRC32Handle));
        } else {
          OP_REQUIRES_OK(ctx, _table_instance->set_K_bucket_num_handle(
                                  KBucketNumCommonHandle<K>));
        }
        OP_REQUIRES_OK(ctx, _table_instance->Conn());
        break;
      }
      default: {
        LOG(ERROR) << "There are only three Redis connection modes, which "
                      "Cluster=0/Sentinel=1/Standalone=2.";
        ctx->CtxFailure(errors::InvalidArgument(
            std::to_string(redis_connection_params.redis_connection_mode) +
            " is illegal redis_connection_mode."));
        break;
      }
    }

    CreateKeysPrefixNameHandle(cluster_slots, &redis_connection_params,
                               embedding_name, keys_prefix_name,
                               keys_prefix_name_import, keys_prefix_name_slices,
                               keys_prefix_name_slices_import);

    // Rehash buckets
    auto keys_prefix_name_slices_import_sort = keys_prefix_name_slices_import;
    std::sort(keys_prefix_name_slices_import_sort.begin(),
              keys_prefix_name_slices_import_sort.end());
    auto keys_prefix_name_slices_sort = keys_prefix_name_slices;
    std::sort(keys_prefix_name_slices_sort.begin(),
              keys_prefix_name_slices_sort.end());
    if (redis_connection_params.model_tag_import ==
            redis_connection_params.model_tag_runtime &&
        (keys_prefix_name_slices_import_sort != keys_prefix_name_slices_sort ||
         redis_connection_params.storage_slice_import !=
             static_cast<int>(redis_connection_params.storage_slice))) {
      auto keys_prefix_name_slices_redis =
          _table_instance->GetKeyBucketsAndOptimizerParamsWithName(
              keys_prefix_name_import, true);
      auto keys_prefix_name_slices_redis_sort = keys_prefix_name_slices_redis;
      std::sort(keys_prefix_name_slices_redis_sort.begin(),
                keys_prefix_name_slices_redis_sort.end());
      LOG(INFO) << "Arrange the new Redis hash tags to the table "
                << keys_prefix_name_import
                << ". And remove the old one. Remember changing config file "
                   "next time!";
      if (keys_prefix_name_slices_redis.size() ==
          redis_connection_params.storage_slice) {
        if (keys_prefix_name_slices_redis_sort ==
            keys_prefix_name_slices_import_sort) {
          OP_REQUIRES_OK(ctx, _table_instance->DuplicateInRedis(
                                  keys_prefix_name_slices_import,
                                  keys_prefix_name_slices));
          for (auto keys_prefix_name_slice_import :
               keys_prefix_name_slices_import) {
            OP_REQUIRES_OK(ctx, _table_instance->RemoveHkeysInBuckets(
                                    keys_prefix_name_slice_import));
          }
        }
      } else {
        if (keys_prefix_name_slices_redis_sort !=
            keys_prefix_name_slices_import_sort) {
          std::stringstream warning_print;
          for (auto keys_prefix_name_slice_import :
               keys_prefix_name_slices_import) {
            warning_print << keys_prefix_name_slice_import << " , ";
          }
          warning_print << "; And the keys_prefix_name_slices_redis are: ";
          for (auto keys_prefix_name_slice_redis :
               keys_prefix_name_slices_redis) {
            warning_print << keys_prefix_name_slice_redis << " , ";
          }
          warning_print << " . Now try to replace imported "
                           "keys_prefix_name_slices with those in Redis. "
                        << std::endl;
          LOG(WARNING) << "Hashtag in Redis for " << keys_prefix_name_import
                       << " is not equal to the imported one. Imported "
                          "keys_prefix_name_slices are: "
                       << warning_print.str();
          keys_prefix_name_slices_import.swap(keys_prefix_name_slices_redis);
        }
        LOG(WARNING)
            << "The embedding table prefix name " << keys_prefix_name_import
            << " has already been saved in the Redis Servers. "
            << "And its number of slices is not equal to the number you putted "
               "in the setting. ";
        LOG(INFO) << "Try to recreate the embedding table "
                  << keys_prefix_name_import << " into the bucket number "
                  << redis_connection_params.storage_slice << " for"
                  << keys_prefix_name;
        OP_REQUIRES_OK(ctx, ReCreateTableBuckets(ctx, keys_prefix_name_import));
      }

      if (_table_instance->CheckSlicesNum(keys_prefix_name) != 1) {
        LOG(WARNING)
            << "CheckSlicesNum fails after many operations. Please check your "
               "Redis service manually. If you are in the first few steps of "
               "your training, you can ignore this warning.";
      }
    }

    // remove expiring time of buckets
    OP_REQUIRES_OK(ctx, _table_instance->SetPersistBuckets(keys_prefix_name));

    // allocate the memory of threads helper
    for (size_t i = 0; i < hardware_concurrency_; ++i) {
      threads_Find.emplace_back(new ThreadContext());
      threads_Insert.emplace_back(new ThreadContext());
      threads_Delete.emplace_back(new ThreadContext());
    }
  }

  ~RedisTableOfTensors() {
    if (_table_instance != nullptr && _table_instance->isRedisConnect == true) {
      auto statu = _table_instance->SetExpireBuckets(keys_prefix_name);
      if (statu != TFOkStatus) {
        LOG(ERROR) << "Redis instance SetExpireBuckets failed.";
      }
    }

    for (auto &in_aiocb_obj : IMPORT_content) {
      if (in_aiocb_obj.aio_buf) {
        free((void *)in_aiocb_obj.aio_buf);
      }
    }
    for (auto &ex_aiocb_obj : EXPORT_content) {
      if (ex_aiocb_obj.aio_buf) {
        free((void *)ex_aiocb_obj.aio_buf);
      }
    }
    for (auto &threads_Find_i : threads_Find) {
      if (threads_Find_i->thread_occupied.load(std::memory_order_consume) ==
          false) {
        threads_Find_i->HandleRelease();
      }
    }
    for (auto &threads_Insert_i : threads_Insert) {
      if (threads_Insert_i->thread_occupied.load(std::memory_order_consume) ==
          false) {
        threads_Insert_i->HandleRelease();
      }
    }
    for (auto &threads_Delete_i : threads_Delete) {
      if (threads_Delete_i->thread_occupied.load(std::memory_order_consume) ==
          false) {
        threads_Delete_i->HandleRelease();
      }
    }

    if (_table_instance != nullptr) {
      _table_instance.reset();
    }
  }

  Status ReCreateTableBuckets(OpKernelContext *ctx,
                              const std::string &keys_prefix_name_from) {
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> hscan_reply;
    const redisReply *kvs_reply;
    std::vector<std::string> keys_prefix_name_slices_in_redis =
        _table_instance->GetKeyBucketsAndOptimizerParamsWithName(
            keys_prefix_name_from, false);
    Tensor keys_temp;
    const K *pk_raw;
    Tensor values_temp;
    const V *pv_raw;
    int64_t slice_keys_size = 0;
    long long cursor = 0;

    redisReply *temp_reply;
    for (size_t i = 0; i < keys_prefix_name_slices_in_redis.size(); ++i) {
      slice_keys_size = _table_instance->TableSizeInBucket(
          keys_prefix_name_slices_in_redis[i]);
      // fill Tensor keys_temp
      try {
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<K>::v(),
                                              TensorShape({slice_keys_size}),
                                              &keys_temp));
        TF_RETURN_IF_ERROR(ctx->allocate_temp(
            DataTypeToEnum<V>::v(),
            TensorShape({slice_keys_size, runtime_value_dim_}), &values_temp));
        pk_raw = reinterpret_cast<const K *>(keys_temp.tensor_data().data());
        pv_raw = reinterpret_cast<const V *>(values_temp.tensor_data().data());
        cursor = 0;
        while (true) {
          hscan_reply.reset();
          hscan_reply = std::move(_table_instance->HscanGetKeysValsInBucket(
              keys_prefix_name_slices_in_redis[i], &cursor,
              multi_redis_cmd_max_argc));
          if (hscan_reply == nullptr) {
            return errors::Unknown(
                "Unknown errors happen when HscanGetKeysValsInBucket in "
                "ReCreateTableBuckets");
          }
          if (hscan_reply->type == REDIS_REPLY_ARRAY &&
              hscan_reply->elements > 1) {
            kvs_reply = hscan_reply->element[1];
            // fill Tensor keys and values
            if (kvs_reply->elements < 2 && cursor == 0) {
              // Find nothing in Redis
              break;
            }
            if constexpr (!std::is_same<V, tstring>::value) {
              if (kvs_reply->element[1]->len !=
                  runtime_value_dim_ * sizeof(V)) {
                return errors::InvalidArgument(
                    "Embedding dim in Redis server is not equal to the OP "
                    "runtime "
                    "dim.");
              }
            }
            for (size_t j = 0; j < kvs_reply->elements; ++j) {
              temp_reply = kvs_reply->element[j];
              if (temp_reply->type ==
                  REDIS_REPLY_STRING) {  // #define REDIS_REPLY_STRING 1
                ReplyMemcpyToKeyTensor<K>(
                    pk_raw, temp_reply->str,
                    temp_reply
                        ->len);  // Direct access to Tensor data in TensorFlow
              }
              ++pk_raw;

              ++j;
              temp_reply = kvs_reply->element[j];
              if (temp_reply->type ==
                  REDIS_REPLY_STRING) {  // #define REDIS_REPLY_STRING 1
                ReplyMemcpyToValTensor<V>(
                    pv_raw, temp_reply->str,
                    runtime_value_dim_);  // Direct access to Tensor data in
                                          // TensorFlow
              }
              pv_raw += runtime_value_dim_;
            }
          }

          LOG(INFO) << "The cursor of scanning "
                    << keys_prefix_name_slices_in_redis[i]
                    << " in ReCreateTableBuckets is " << cursor << " now.";
          if (cursor == 0) {
            break;
          }
        }
      } catch (const std::exception &err) {
        LOG(ERROR) << "Some errors happened when try to copy Redis old buckets "
                      "data reply into tensor for preparing to insert"
                   << " -- " << err.what();
        return errors::Unknown(err.what());
      }
      try {
        // insert KV pair into new Redis with new storage_slice
        launchInsert(ctx, keys_prefix_name_slices, pk_raw, pv_raw,
                     slice_keys_size, runtime_value_dim_, threads_Insert);
      } catch (const std::exception &err) {
        LOG(ERROR)
            << "Some errors happened when try to insert new buckets into Redis"
            << " -- " << err.what();
        return errors::Unknown(err.what());
      }
    }
    auto statu = TFOkStatus;
    for (auto keys_prefix_name_slice_in_redis :
         keys_prefix_name_slices_in_redis) {
      LOG(INFO) << "Now try to delet old bucket "
                << keys_prefix_name_slice_in_redis;
      auto iter = std::find(keys_prefix_name_slices.begin(),
                            keys_prefix_name_slices.end(),
                            keys_prefix_name_slice_in_redis);
      if (iter == keys_prefix_name_slices.end()) {
        statu = _table_instance->RemoveHkeysInBuckets(
            keys_prefix_name_slice_in_redis);
        if (statu != TFOkStatus) {
          return statu;
        }
      }
    }
    return TFOkStatus;
  }

  size_t size() const override {
    size_t size = 0;
    const unsigned &storage_slice = redis_connection_params.storage_slice;
    for (unsigned i = 0; i != storage_slice; ++i) {
      size += _table_instance->TableSizeInBucket(keys_prefix_name_slices[i]);
    }
    return size;
  }

  Status Find(OpKernelContext *ctx, const Tensor &keys, Tensor *values,
              const Tensor &default_value) override {
    int64_t total = keys.NumElements();
    if (total > 0) {
      const int64_t Velems_per_flat2_dim0 = values->NumElements() / total;
      const bool is_full_default =
          (values->NumElements() == default_value.NumElements());
      if (total < (multi_redis_cmd_max_argc - 1)) {
        launchFind(ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
                   (V *)values->tensor_data().data(),
                   (V *)default_value.tensor_data().data(), total,
                   Velems_per_flat2_dim0, is_full_default, threads_Find);
      } else {
        // redis commmand args > multi_redis_cmd_max_argc
        launchFind_parallel(
            ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
            (V *)values->tensor_data().data(),
            (V *)default_value.tensor_data().data(), total,
            Velems_per_flat2_dim0, is_full_default, threads_Find);
      }
    }

    return TFOkStatus;
  }

  Status FindWithExists(OpKernelContext *ctx, const Tensor &keys,
                        Tensor *values, const Tensor &default_value,
                        Tensor &exists) {
    int64_t total = keys.NumElements();
    if (total > 0) {
      const int64_t Velems_per_flat2_dim0 = values->NumElements() / total;
      const bool is_full_default =
          (values->NumElements() == default_value.NumElements());
      if (total < (multi_redis_cmd_max_argc - 1)) {
        launchFindWithExists(
            ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
            (V *)values->tensor_data().data(),
            (V *)default_value.tensor_data().data(),
            (bool *)exists.tensor_data().data(), total, Velems_per_flat2_dim0,
            is_full_default, threads_Find);
      } else {
        // redis commmand args > multi_redis_cmd_max_argc
        launchFindWithExists_parallel(
            ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
            (V *)values->tensor_data().data(),
            (V *)default_value.tensor_data().data(),
            (bool *)exists.tensor_data().data(), total, Velems_per_flat2_dim0,
            is_full_default, threads_Find);
      }
    }
    return TFOkStatus;
  }

  Status DoInsert(bool clear, OpKernelContext *ctx, const K *keys,
                  const V *values, const int64_t total,
                  const int64_t Velems_per_flat2_dim0) {
    auto statu = TFOkStatus;
    if (clear) {
      for (auto keys_prefix_name_slice : keys_prefix_name_slices) {
        statu = _table_instance->RemoveHkeysInBuckets(keys_prefix_name_slice);
        if (statu != TFOkStatus) {
          return statu;
        }
      }
    }
    if (total < (multi_redis_cmd_max_argc - 1)) {
      launchInsert(ctx, keys_prefix_name_slices, keys, values, total,
                   Velems_per_flat2_dim0, threads_Insert);
    } else {
      launchInsert_parallel(
          ctx, keys_prefix_name_slices, keys, values, total,
          Velems_per_flat2_dim0,
          threads_Insert);  // redis commmand args > multi_redis_cmd_max_argc
    }
    return TFOkStatus;
  }

  Status Insert(OpKernelContext *ctx, const Tensor &keys,
                const Tensor &values) override {
    const int64_t total = keys.NumElements();
    if (total > 0) {
      const int64_t Velems_per_flat2_dim0 = values.NumElements() / total;
      return DoInsert(false, ctx, (K *)keys.tensor_data().data(),
                      (V *)values.tensor_data().data(), total,
                      Velems_per_flat2_dim0);
    } else {
      LOG(INFO) << "Redis Backend Insert nothing for empty input keys tensor.";
      return TFOkStatus;
    }
  }

  Status Accum(OpKernelContext *ctx, const Tensor &keys,
               const Tensor &values_or_delta, const Tensor &exists) {
    int64_t total = keys.NumElements();
    const int64_t Velems_per_flat2_dim0 =
        values_or_delta.NumElements() / keys.NumElements();
    auto values_dtype_str = DataTypeString(values_or_delta.dtype());

    if (total < (multi_redis_cmd_max_argc - 1)) {
      launchAccum(ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
                  (V *)values_or_delta.tensor_data().data(),
                  (bool *)exists.tensor_data().data(), total,
                  Velems_per_flat2_dim0, values_dtype_str, threads_Insert);
    } else {
      launchAccum_parallel(
          ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
          (V *)values_or_delta.tensor_data().data(),
          (bool *)exists.tensor_data().data(), total, Velems_per_flat2_dim0,
          values_dtype_str,
          threads_Insert);  // redis commmand args > multi_redis_cmd_max_argc
    }

    return TFOkStatus;
  }

  Status Remove(OpKernelContext *ctx, const Tensor &keys) override {
    int64_t total = keys.NumElements();
    if (total > 0) {
      if (total < (multi_redis_cmd_max_argc - 1)) {
        launchDelete(ctx, keys_prefix_name_slices,
                     (K *)keys.tensor_data().data(), total, threads_Delete);
      } else {
        // redis commmand args > multi_redis_cmd_max_argc
        launchDelete_parallel(ctx, keys_prefix_name_slices,
                              (K *)keys.tensor_data().data(), total,
                              threads_Delete);
      }
    }
    return TFOkStatus;
  }

  Status Clear(OpKernelContext *ctx) {
    auto statu = TFOkStatus;
    for (auto keys_prefix_name_slice : keys_prefix_name_slices) {
      statu = _table_instance->RemoveHkeysInBuckets(keys_prefix_name_slice);
      if (statu != TFOkStatus) {
        return statu;
      }
    }
    return TFOkStatus;
  }

  Status ImportValues(OpKernelContext *ctx, const Tensor &keys,
                      const Tensor &values) override {
    if (redis_connection_params.table_store_mode == 1) {
      // When there is not a corresponding table existing in Redis service and
      // table_store_mode==1, try to restore from a Redis binary dump files
      // which paths are directory
      // '[model_lib_abs_dir]/[model_tag]/[name].rdb'.
      return ImportValuesFromFiles(ctx);
    } else {
      if (keys.NumElements() > 0 &&
          redis_connection_params.table_store_mode == 0) {
        return Insert(ctx, keys, values);
      } else {
        LOG(INFO) << "Import nothing from the TensorFlow saved model to Redis "
                     "service for "
                  << keys_prefix_name_import;
        if (redis_connection_params.model_tag_import !=
            redis_connection_params.model_tag_runtime) {
          if (_table_instance->CheckSlicesNum(keys_prefix_name_import) == 1 &&
              _table_instance->CheckSlicesNum(keys_prefix_name) != 1) {
            LOG(INFO) << "Because model_tag_import is not equal to "
                         "model_tag_runtime. Now begin to DuplicateInRedis, "
                         "remember changing config file next time!";
            return _table_instance->DuplicateInRedis(
                keys_prefix_name_slices_import, keys_prefix_name_slices);
          }
        }
        return TFOkStatus;
      }
    }
  }

  Status ImportValuesFromFiles(OpKernelContext *ctx) {
    std::string file_path, folder_dir;
    const unsigned &storage_slice = redis_connection_params.storage_slice;

    IMPORT_content.resize(storage_slice);
    IMPORT_fds.clear();
    IMPORT_fds.reserve(storage_slice);
    IMPORT_fds_sizes.clear();
    IMPORT_fds_sizes.reserve(storage_slice);

    folder_dir = check_dir(redis_connection_params.model_lib_abs_dir);
    folder_dir =
        check_dir(folder_dir + redis_connection_params.model_tag_import);

    auto statu = TFOkStatus;

    for (unsigned i = 0; i < storage_slice; ++i) {
      file_path = folder_dir + keys_prefix_name_slices_import[i] + ".rdb";
      if (access(file_path.c_str(), 0) == -1) {
        LOG(WARNING) << "file " << file_path
                     << " doesn't exist. Using the table that already exist in "
                        "the Redis or creating a new one";
      } else {
        IMPORT_fds.push_back(open(file_path.c_str(), O_RDONLY));
        IMPORT_fds_sizes.push_back(get_file_size(file_path));
      }
    }

    if (IMPORT_fds.size() > 0) {
      LOG(INFO) << "Try to restore the table " << keys_prefix_name
                << " to Redis service from "
                << folder_dir + keys_prefix_name_slices_import[0] +
                       ".rdb and its companions";

      statu = _table_instance->RestoreFromDisk(keys_prefix_name_slices,
                                               IMPORT_content, IMPORT_fds,
                                               IMPORT_fds_sizes);
      if (statu != TFOkStatus) {
        return statu;
      }
      for (auto &fd : IMPORT_fds) close(fd);
    }

    return TFOkStatus;
  }

  Status ExportValues(OpKernelContext *ctx) override {
    if (redis_connection_params.table_store_mode == 0) {
      return ExportValuesToTensor(ctx);
    } else if (redis_connection_params.table_store_mode == 1) {
      return ExportValuesToFiles(ctx);
    } else if (redis_connection_params.table_store_mode == 2) {
      Tensor *keys;
      TF_RETURN_IF_ERROR(ctx->allocate_output("keys", TensorShape({0}), &keys));
      Tensor *values;
      TF_RETURN_IF_ERROR(ctx->allocate_output(
          "values", TensorShape({0, runtime_value_dim_}), &values));
      return TFOkStatus;
    }
#if TF_VERSION_INTEGER >= 2130  // 2.13.0
    return Status(absl::StatusCode::kInvalidArgument,
                  "invalid redis_connection_params.table_store_mode.");
#else
    return Status(error::INVALID_ARGUMENT,
                  "invalid redis_connection_params.table_store_mode.");
#endif
  }

  Status ExportValuesToFiles(OpKernelContext *ctx) {
    std::string file_path, folder_dir;
    const unsigned &storage_slice = redis_connection_params.storage_slice;
    int tem_fd;
    auto statu = TFOkStatus;

    EXPORT_content.resize(storage_slice);
    EXPORT_fds.clear();
    EXPORT_fds.reserve(storage_slice);

    folder_dir = check_dir(redis_connection_params.model_lib_abs_dir);
    folder_dir =
        check_dir(folder_dir + redis_connection_params.model_tag_runtime);

    for (unsigned i = 0; i < storage_slice; ++i) {
      file_path = folder_dir + keys_prefix_name_slices[i] + ".rdb";
      if (access(file_path.c_str(), 0) != -1) {
        LOG(WARNING) << "File " + file_path + " has already existed!";
        time_t totalseconds = time(NULL);
        struct tm *st = localtime(&totalseconds);
        char tmp_time_str[20];
        sprintf(tmp_time_str, "%04d-%02d-%02d-%02d:%02d:%02d",
                (st->tm_year + 1900) % 10000u, (st->tm_mon + 1) % 100u,
                (st->tm_mday) % 100u, (st->tm_hour) % 100u, (st->tm_min) % 100u,
                (st->tm_sec) % 100u);
        std::string new_file_path = file_path + "." + tmp_time_str;
        LOG(WARNING) << "Rename the file " + file_path + " into " +
                            new_file_path + " with local time!";
        rename(file_path.c_str(), new_file_path.c_str());
        tem_fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0777);
        if (tem_fd > 0) {
          EXPORT_fds.push_back(tem_fd);
        } else {
          LOG(ERROR) << "Can not create the file " << file_path
                     << " for instead. Something bad happens";
        }
      } else {
        tem_fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0777);
        EXPORT_fds.push_back(tem_fd);
      }
    }

    if (EXPORT_fds.size() > 0) {
      LOG(INFO) << "Try to dump the table " << keys_prefix_name
                << " from Redis service to "
                << folder_dir + keys_prefix_name + "[*].rdb";

      statu = _table_instance->DumpToDisk(keys_prefix_name_slices,
                                          EXPORT_content, EXPORT_fds);
      if (statu != TFOkStatus) {
        return statu;
      }
      // for (auto &fd : EXPORT_fds) // for now the writting may be not
      // finished
      //   close(fd);
    }

    Tensor *keys;
    TF_RETURN_IF_ERROR(ctx->allocate_output("keys", TensorShape({1}), &keys));

    Tensor *values;
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({1, runtime_value_dim_}), &values));

    return TFOkStatus;
  }

  Status ExportValuesToTensor(OpKernelContext *ctx) {
    int64_t total_size = 0;
    long long cursor = 0;
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> hscan_reply;
    const redisReply *kvs_reply;

    for (size_t i = 0; i < keys_prefix_name_slices.size(); ++i) {
      total_size +=
          _table_instance->TableSizeInBucket(keys_prefix_name_slices[i]);
    }

    Tensor *keys;
    TF_RETURN_IF_ERROR(
        ctx->allocate_output("keys", TensorShape({total_size}), &keys));

    Tensor *values;
    TF_RETURN_IF_ERROR(ctx->allocate_output(
        "values", TensorShape({total_size, runtime_value_dim_}), &values));

    if (total_size == 0) {
      LOG(WARNING) << "There is no embedding table called " << keys_prefix_name
                   << " existing in the Redis service. "
                   << "Exporting values to Tensor failed.";
      return TFOkStatus;
    }

    redisReply const *temp_reply;
    const K *pk_raw = reinterpret_cast<const K *>(keys->tensor_data().data());
    const V *pv_raw = reinterpret_cast<const V *>(values->tensor_data().data());
    for (size_t i = 0; i < keys_prefix_name_slices.size(); ++i) {
      cursor = 0;
      while (true) {
        hscan_reply.reset();
        hscan_reply = _table_instance->HscanGetKeysValsInBucket(
            keys_prefix_name_slices[i], &cursor, multi_redis_cmd_max_argc);
        if (hscan_reply == nullptr) {
          return errors::Unknown(
              "Unknown errors happen when HscanGetKeysValsInBucket in "
              "ExportValuesToTensor");
        }
        kvs_reply = hscan_reply->element[1];
        // fill Tensor keys and values
        if (kvs_reply->elements < 2 && cursor == 0) {
          // Find nothing in Redis
          break;
        }
        if constexpr (!std::is_same<V, tstring>::value) {
          if (kvs_reply->element[1]->len != runtime_value_dim_ * sizeof(V)) {
            return errors::InvalidArgument(
                "Embedding dim in Redis server is not equal to the OP runtime "
                "dim.");
          }
        }
        for (size_t j = 0; j < kvs_reply->elements; ++j) {
          temp_reply = kvs_reply->element[j];
          if (temp_reply->type ==
              REDIS_REPLY_STRING) {  // #define REDIS_REPLY_STRING 1
            ReplyMemcpyToKeyTensor<K>(
                pk_raw, temp_reply->str,
                temp_reply->len);  // Direct access to Tensor data in TensorFlow
          }
          ++pk_raw;

          ++j;
          temp_reply = kvs_reply->element[j];
          if (temp_reply->type ==
              REDIS_REPLY_STRING) {  // #define REDIS_REPLY_STRING 1
            ReplyMemcpyToValTensor<V>(
                pv_raw, temp_reply->str,
                runtime_value_dim_);  // Direct access to Tensor data in
                                      // TensorFlow
          }
          pv_raw += runtime_value_dim_;
        }

        LOG(INFO) << "The cursor of scanning " << keys_prefix_name_slices[i]
                  << " in ExportValuesToTensor is " << cursor << " now.";
        if (cursor == 0) {
          break;
        }
      }
    }

    return TFOkStatus;
  }

  Status SaveToFileSystemImpl(FileSystem *fs, const string &filepath,
                              const size_t buffer_size,
                              const bool append_to_file) {
    int64_t total_size = 0;
    long long cursor = 0;
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> hscan_reply;
    const redisReply *kvs_reply;

    for (size_t i = 0; i < keys_prefix_name_slices.size(); ++i) {
      total_size +=
          _table_instance->TableSizeInBucket(keys_prefix_name_slices[i]);
    }

    // construct file system relative object
    std::unique_ptr<WritableFile> key_writer;
    std::unique_ptr<WritableFile> value_writer;
    const string key_filepath(filepath + "-keys");
    const string value_filepath(filepath + "-values");
    string key_tmpfilepath(filepath + "-keys.tmp");
    string value_tmpfilepath(filepath + "-values.tmp");
    bool has_atomic_move = false;
    auto has_atomic_move_ret = fs->HasAtomicMove(filepath, &has_atomic_move);
    bool need_tmp_file =
        (has_atomic_move == false) || (has_atomic_move_ret != TFOkStatus);
    if (!need_tmp_file) {
      key_tmpfilepath = key_filepath;
      value_tmpfilepath = value_filepath;
    }
    TF_RETURN_IF_ERROR(
        fs->RecursivelyCreateDir(std::string(fs->Dirname(filepath))));
    if (append_to_file) {
      TF_RETURN_IF_ERROR(fs->NewAppendableFile(key_tmpfilepath, &key_writer));
      TF_RETURN_IF_ERROR(
          fs->NewAppendableFile(value_tmpfilepath, &value_writer));
    } else {
      TF_RETURN_IF_ERROR(fs->NewWritableFile(key_tmpfilepath, &key_writer));
      TF_RETURN_IF_ERROR(fs->NewWritableFile(value_tmpfilepath, &value_writer));
    }

    if (total_size == 0) {
      LOG(WARNING) << "There is no embedding table called " << keys_prefix_name
                   << " existing in the Redis service. "
                   << "Saving values to file system failed.";
      return TFOkStatus;
    }

    // buffer for write to file system
    const size_t value_len = sizeof(V) * runtime_value_dim_;
    const size_t key_buffer_byte_size = buffer_size * sizeof(K);
    const size_t value_buffer_byte_size = buffer_size * value_len;
    std::vector<char> key_buffer_vector(key_buffer_byte_size);
    std::vector<char> value_buffer_vector(value_buffer_byte_size);

    size_t total_saved = 0;

    redisReply const *temp_reply;
    const K *pk_raw = reinterpret_cast<const K *>(key_buffer_vector.data());
    const V *pv_raw = reinterpret_cast<const V *>(value_buffer_vector.data());
    for (size_t i = 0; i < keys_prefix_name_slices.size(); ++i) {
      cursor = 0;
      while (true) {
        hscan_reply.reset();
        hscan_reply = _table_instance->HscanGetKeysValsInBucket(
            keys_prefix_name_slices[i], &cursor, multi_redis_cmd_max_argc);
        if (hscan_reply == nullptr) {
          return errors::Unknown(
              "Unknown errors happen when HscanGetKeysValsInBucket in "
              "SaveToFileSystemImpl");
        }
        kvs_reply = hscan_reply->element[1];
        // fill Tensor keys and values
        if (kvs_reply->elements < 2 && cursor == 0) {
          // Find nothing in Redis
          break;
        }
        if constexpr (!std::is_same<V, tstring>::value) {
          if (kvs_reply->element[1]->len != runtime_value_dim_ * sizeof(V)) {
            return errors::InvalidArgument(
                "Embedding dim in Redis server is not equal to the OP runtime "
                "dim.");
          }
        }
        for (size_t j = 0; j < kvs_reply->elements; ++j) {
          temp_reply = kvs_reply->element[j];
          if (temp_reply->type ==
              REDIS_REPLY_STRING) {  // #define REDIS_REPLY_STRING 1
            ReplyMemcpyToKeyTensor<K>(
                pk_raw, temp_reply->str,
                temp_reply->len);  // Direct access to Tensor data in TensorFlow
          }
          ++pk_raw;

          ++j;
          temp_reply = kvs_reply->element[j];
          if (temp_reply->type ==
              REDIS_REPLY_STRING) {  // #define REDIS_REPLY_STRING 1
            ReplyMemcpyToValTensor<V>(
                pv_raw, temp_reply->str,
                runtime_value_dim_);  // Direct access to Tensor data in
                                      // TensorFlow
          }
          pv_raw += runtime_value_dim_;

          if (((char *)pk_raw - key_buffer_vector.data()) >=
              static_cast<int64_t>(key_buffer_byte_size)) {
            pk_raw = reinterpret_cast<const K *>(key_buffer_vector.data());
            TF_RETURN_IF_ERROR(key_writer->Append(
                StringPiece((char *)pk_raw, key_buffer_byte_size)));
            pv_raw = reinterpret_cast<const V *>(value_buffer_vector.data());
            TF_RETURN_IF_ERROR(value_writer->Append(
                StringPiece((char *)pv_raw, value_buffer_byte_size)));
          }
          ++total_saved;
        }

        LOG(INFO) << "The cursor of scanning " << keys_prefix_name_slices[i]
                  << " in SaveToFileSystem is " << cursor << " now.";
        if (cursor == 0) {
          break;
        }
      }
    }

    if (((char *)pk_raw - key_buffer_vector.data()) &&
        ((char *)pv_raw - value_buffer_vector.data())) {
      TF_RETURN_IF_ERROR(key_writer->Append(
          StringPiece(key_buffer_vector.data(),
                      (char *)pk_raw - key_buffer_vector.data())));
      TF_RETURN_IF_ERROR(value_writer->Append(
          StringPiece(value_buffer_vector.data(),
                      (char *)pv_raw - value_buffer_vector.data())));
    }

    TF_RETURN_IF_ERROR(key_writer->Flush());
    TF_RETURN_IF_ERROR(value_writer->Flush());
    TF_RETURN_IF_ERROR(key_writer->Sync());
    TF_RETURN_IF_ERROR(value_writer->Sync());

    LOG(INFO) << "Finish saving " << total_saved << " keys and values to "
              << key_filepath << " and " << value_filepath << " in total.";

    if (need_tmp_file) {
      TF_RETURN_IF_ERROR(fs->FileExists(key_tmpfilepath));
      TF_RETURN_IF_ERROR(fs->RenameFile(key_tmpfilepath, key_filepath));
      TF_RETURN_IF_ERROR(fs->FileExists(value_tmpfilepath));
      TF_RETURN_IF_ERROR(fs->RenameFile(value_tmpfilepath, value_filepath));
    }

    return TFOkStatus;
  }

  Status SaveToFileSystem(OpKernelContext *ctx, const string &dirpath,
                          const string &file_name, const size_t buffer_size,
                          bool append_to_file) {
    string filepath = io::JoinPath(dirpath, file_name);
    FileSystem *fs;
    const auto env = ctx->env();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        env->GetFileSystemForFile(filepath, &fs),
        "Please make sure you have already imported tensorflow_io before using "
        "TFRA file system operation.");
    return SaveToFileSystemImpl(fs, filepath, buffer_size, append_to_file);
  }

  Status LoadFromFileSystemImpl(OpKernelContext *ctx, FileSystem *fs,
                                const string &filepath,
                                const size_t buffer_size) {
    const string key_filepath = filepath + "-keys";
    TF_RETURN_IF_ERROR(fs->FileExists(key_filepath));
    std::unique_ptr<RandomAccessFile> key_file;
    TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(key_filepath, &key_file));
    std::unique_ptr<io::RandomAccessInputStream> key_input_stream(
        new io::RandomAccessInputStream(key_file.get()));
    size_t key_buffer_byte_size = buffer_size * sizeof(K);
    io::BufferedInputStream key_reader(key_input_stream.get(),
                                       key_buffer_byte_size * 2);

    const string value_filepath = filepath + "-values";
    TF_RETURN_IF_ERROR(fs->FileExists(key_filepath));
    std::unique_ptr<RandomAccessFile> value_file;
    TF_RETURN_IF_ERROR(fs->NewRandomAccessFile(value_filepath, &value_file));
    std::unique_ptr<io::RandomAccessInputStream> value_input_stream(
        new io::RandomAccessInputStream(value_file.get()));
    const size_t value_len = sizeof(V) * runtime_value_dim_;
    size_t value_buffer_byte_size = buffer_size * value_len;
    io::BufferedInputStream value_reader(value_input_stream.get(),
                                         value_buffer_byte_size * 2);

    uint64 key_file_size = 0;
    TF_RETURN_IF_ERROR(fs->GetFileSize(key_filepath, &key_file_size));
    const size_t key_size = key_file_size / sizeof(K);

    uint64 value_file_size = 0;
    TF_RETURN_IF_ERROR(fs->GetFileSize(value_filepath, &value_file_size));
    const size_t value_size = value_file_size / value_len;

    if (key_size != value_size) {
      return errors::Unavailable(
          "the keys number in file " + key_filepath +
          " is not equal to the value vectors number in file " +
          value_filepath + ".");
    }

    tstring key_buffer;
    key_buffer.resize(key_buffer_byte_size);
    tstring value_buffer;
    value_buffer.resize(value_buffer_byte_size);

    size_t key_file_offset = 0;
    int64_t remainder = key_file_size - key_file_offset;
    size_t nkeys = 0;
    size_t key_read_byte = 0;
    size_t value_read_byte = 0;
    while (remainder > 0) {
      if (remainder > static_cast<int64_t>(key_buffer_byte_size)) {
        key_read_byte = key_buffer_byte_size;
        nkeys = buffer_size;
        value_read_byte = value_buffer_byte_size;
      } else {
        key_read_byte = remainder;
        nkeys = key_read_byte / sizeof(K);
        value_read_byte = nkeys * value_len;
      }
      TF_RETURN_IF_ERROR(key_reader.ReadNBytes(key_read_byte, &key_buffer));
      TF_RETURN_IF_ERROR(
          value_reader.ReadNBytes(value_read_byte, &value_buffer));
      TF_RETURN_IF_ERROR(DoInsert(false, ctx, (K *)key_buffer.data(),
                                  (V *)value_buffer.data(), nkeys,
                                  runtime_value_dim_));
      key_file_offset += key_read_byte;
      remainder = key_file_size - key_file_offset;
    }

    LOG(INFO) << "Finish loading " << key_size << " keys and values from "
              << key_filepath << " and " << value_filepath << " in total.";

    return TFOkStatus;
  }

  Status LoadFromFileSystem(OpKernelContext *ctx, const string &dirpath,
                            const string &file_name, const size_t buffer_size,
                            bool load_entire_dir) {
    string filepath = io::JoinPath(dirpath, file_name);
    FileSystem *fs;
    const auto env = ctx->env();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        env->GetFileSystemForFile(filepath, &fs),
        "Please make sure you have already imported tensorflow_io before using "
        "TFRA file system operation.");
    if (load_entire_dir) {
      string separator = "_mht_";
      int separator_pos = file_name.rfind(separator);
      string file_pattern =
          io::JoinPath(dirpath,
                       file_name.substr(0, separator_pos + separator.size())) +
          "*";
      std::vector<string> all_filepath;
      TF_RETURN_IF_ERROR(fs->GetMatchingPaths(file_pattern, &all_filepath));
      // delete -keys/-values postfix
      for (auto it = all_filepath.begin(); it != all_filepath.end(); ++it) {
        int kv_separator_pos = it->rfind("-");
        *it = it->substr(0, kv_separator_pos);
      }
      // remove duplicate elements
      sort(all_filepath.begin(), all_filepath.end());
      all_filepath.erase(unique(all_filepath.begin(), all_filepath.end()),
                         all_filepath.end());
      for (auto &fp : all_filepath) {
        TF_RETURN_IF_ERROR(LoadFromFileSystemImpl(ctx, fs, fp, buffer_size));
      }
    } else {
      return LoadFromFileSystemImpl(ctx, fs, filepath, buffer_size);
    }
    return TFOkStatus;
  }

  DataType key_dtype() const override { return DataTypeToEnum<K>::v(); }

  DataType value_dtype() const override { return DataTypeToEnum<V>::v(); }

  TensorShape key_shape() const final { return TensorShape(); }

  TensorShape value_shape() const override { return value_shape_; }

  tensorflow::int64 MemoryUsed() const override {
    tensorflow::int64 ret = 0;
    ret = (tensorflow::int64)(size() * (sizeof(K) + sizeof(V)));
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

  Status GetTableHandle(StringPiece input_name, OpKernelContext *ctx,
                        string *container, string *table_handle) {
    {
      mutex *mu;
      TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
      mutex_lock l(*mu);
      Tensor tensor;
      TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
      if (tensor.NumElements() != 2) {
        return errors::InvalidArgument(
            "Lookup table handle must be scalar, but had shape: ",
            tensor.shape().DebugString());
      }
      auto h = tensor.flat<tstring>();
      *container = h(0);
      *table_handle = h(1);
    }
    return TFOkStatus;
  }

  Status GetResourceHashTable(StringPiece input_name, OpKernelContext *ctx,
                              LookupInterface **table) {
    const Tensor *handle_tensor;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &handle_tensor));
    const ResourceHandle &handle = handle_tensor->scalar<ResourceHandle>()();
    return this->LookupResource(ctx, handle, table);
  }

  Status GetReferenceLookupTable(StringPiece input_name, OpKernelContext *ctx,
                                 LookupInterface **table) {
    string container;
    string table_handle;
    TF_RETURN_IF_ERROR(
        this->GetTableHandle(input_name, ctx, &container, &table_handle));
    return ctx->resource_manager()->Lookup(container, table_handle, table);
  }

  Status GetTable(OpKernelContext *ctx, LookupInterface **table) {
    if (expected_input_0_ == DT_RESOURCE) {
      return this->GetResourceHashTable("table_handle", ctx, table);
    } else {
      return this->GetReferenceLookupTable("table_handle", ctx, table);
    }
  }

  const DataType expected_input_0_;
};

// Table find op .
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

// Table find op with return exists tensor.
template <class K, class V>
class HashTableFindWithExistsOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    redis_table::RedisTableOfTensors<K, V> *redis_table =
        dynamic_cast<redis_table::RedisTableOfTensors<K, V> *>(table);

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype()};
    DataTypeVector expected_outputs = {table->value_dtype(), DT_BOOL};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, expected_outputs));

    const Tensor &key = ctx->input(1);
    const Tensor &default_value = ctx->input(2);

    TensorShape output_shape = key.shape();
    output_shape.RemoveLastDims(table->key_shape().dims());
    output_shape.AppendShape(table->value_shape());

    Tensor *values;
    Tensor *exists;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("values", output_shape, &values));
    OP_REQUIRES_OK(ctx, ctx->allocate_output("exists", key.shape(), &exists));

    OP_REQUIRES_OK(ctx, redis_table->FindWithExists(ctx, key, values,
                                                    default_value, *exists));
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

    int64_t memory_used_before = 0;
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

// Table accum op.
template <class K, class V>
class HashTableAccumOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    RedisTableOfTensors<K, V> *redis_table = (RedisTableOfTensors<K, V> *)table;

    DataTypeVector expected_inputs = {expected_input_0_, table->key_dtype(),
                                      table->value_dtype(),
                                      DataTypeToEnum<bool>::v()};
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));

    const Tensor &keys = ctx->input(1);
    const Tensor &values_or_deltas = ctx->input(2);
    const Tensor &exists = ctx->input(3);
    OP_REQUIRES(ctx, (values_or_deltas.dtype() != DataTypeToEnum<tstring>::v()),
                errors::InvalidArgument(
                    "AccumOP is not supporting tstring value type!"));
    OP_REQUIRES_OK(
        ctx, table->CheckKeyAndValueTensorsForInsert(keys, values_or_deltas));

    int64_t memory_used_before = 0;
    if (ctx->track_allocations()) {
      memory_used_before = table->MemoryUsed();
    }
    OP_REQUIRES_OK(ctx,
                   redis_table->Accum(ctx, keys, values_or_deltas, exists));
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

    int64_t memory_used_before = 0;
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
template <class K, class V>
class HashTableClearOp : public HashTableOpKernel {
 public:
  using HashTableOpKernel::HashTableOpKernel;

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    redis_table::RedisTableOfTensors<K, V> *redis_table =
        dynamic_cast<redis_table::RedisTableOfTensors<K, V> *>(table);

    int64_t memory_used_before = 0;
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
    out->flat<int64_t>().setConstant(table->size());
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

// Op that export all keys and values to FileSystem.
template <class K, class V>
class HashTableSaveToFileSystemOp : public HashTableOpKernel {
 public:
  explicit HashTableSaveToFileSystemOp(OpKernelConstruction *ctx)
      : HashTableOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dirpath_env", &dirpath_env_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("append_to_file", &append_to_file_));
    int64 signed_buffer_size = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &signed_buffer_size));
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    string dirpath;
    TF_CHECK_OK(ReadStringFromEnvVar(dirpath_env_, "NotFound", &dirpath));
    if (dirpath != "NotFound") {
      LOG(INFO) << "Read TFRA key/value file directory path from the "
                   "environment variable "
                << dirpath_env_ << " successfully. Saving directory path is "
                << dirpath;
    } else {
      const Tensor &dir_tensor = ctx->input(1);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dir_tensor.shape()),
                  errors::InvalidArgument("directory path must be scalar."));
      dirpath = string(dir_tensor.scalar<tstring>()().data());
    }

    const Tensor &fname_tensor = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fname_tensor.shape()),
                errors::InvalidArgument("file name must be scalar."));
    string file_name = string(fname_tensor.scalar<tstring>()().data());

    RedisTableOfTensors<K, V> *redis_table = (RedisTableOfTensors<K, V> *)table;
    OP_REQUIRES_OK(
        ctx, redis_table->SaveToFileSystem(ctx, dirpath, file_name,
                                           buffer_size_, append_to_file_));
  }

 private:
  string dirpath_env_;
  bool append_to_file_;
  size_t buffer_size_;
};

// Insert data.
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

// Insert data from FileSystem.
template <class K, class V>
class HashTableLoadFromFileSystemOp : public HashTableOpKernel {
 public:
  explicit HashTableLoadFromFileSystemOp(OpKernelConstruction *ctx)
      : HashTableOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dirpath_env", &dirpath_env_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_entire_dir", &load_entire_dir_));
    int64 signed_buffer_size = 0;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &signed_buffer_size));
    buffer_size_ = static_cast<size_t>(signed_buffer_size);
  }

  void Compute(OpKernelContext *ctx) override {
    LookupInterface *table;
    OP_REQUIRES_OK(ctx, GetTable(ctx, &table));
    core::ScopedUnref unref_me(table);

    string dirpath;
    TF_CHECK_OK(ReadStringFromEnvVar(dirpath_env_, "NotFound", &dirpath));
    if (dirpath != "NotFound") {
      LOG(INFO) << "Read TFRA key/value file directory path from the "
                   "environment variable "
                << dirpath_env_ << " successfully. Saving directory path is "
                << dirpath;
    } else {
      const Tensor &dir_tensor = ctx->input(1);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(dir_tensor.shape()),
                  errors::InvalidArgument("directory path must be scalar."));
      dirpath = string(dir_tensor.scalar<tstring>()().data());
    }

    const Tensor &fname_tensor = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fname_tensor.shape()),
                errors::InvalidArgument("file name must be scalar."));
    string file_name = string(fname_tensor.scalar<tstring>()().data());

    RedisTableOfTensors<K, V> *redis_table = (RedisTableOfTensors<K, V> *)table;
    OP_REQUIRES_OK(
        ctx, redis_table->LoadFromFileSystem(ctx, dirpath, file_name,
                                             buffer_size_, load_entire_dir_));
  }

 private:
  string dirpath_env_;
  bool load_entire_dir_;
  size_t buffer_size_;
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

// Register the custom op.
#define REGISTER_KERNEL(key_dtype, value_dtype)                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(RedisTableOfTensors))                             \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      HashTableOp<redis_table::RedisTableOfTensors<key_dtype, value_dtype>, \
                  key_dtype, value_dtype>);                                 \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(RedisTableClear))                                 \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      redis_table::HashTableClearOp<key_dtype, value_dtype>);               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(RedisTableAccum))                                 \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      redis_table::HashTableAccumOp<key_dtype, value_dtype>);               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(RedisTableFindWithExists))                        \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("Tin")                                 \
          .TypeConstraint<value_dtype>("Tout"),                             \
      redis_table::HashTableFindWithExistsOp<key_dtype, value_dtype>);      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(RedisTableSaveToFileSystem))                      \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      redis_table::HashTableSaveToFileSystemOp<key_dtype, value_dtype>);    \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name(PREFIX_OP_NAME(RedisTableLoadFromFileSystem))                    \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<key_dtype>("key_dtype")                           \
          .TypeConstraint<value_dtype>("value_dtype"),                      \
      redis_table::HashTableLoadFromFileSystemOp<key_dtype, value_dtype>);

REGISTER_KERNEL(int32, double);
REGISTER_KERNEL(int32, float);
REGISTER_KERNEL(int32, int32);
REGISTER_KERNEL(int32, bfloat16);
REGISTER_KERNEL(int64_t, double);
REGISTER_KERNEL(int64_t, float);
REGISTER_KERNEL(int64_t, int32);
REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int64_t, tstring);
REGISTER_KERNEL(int64_t, int8);
REGISTER_KERNEL(int64_t, Eigen::half);
REGISTER_KERNEL(int64_t, bfloat16);
REGISTER_KERNEL(tstring, bool);
REGISTER_KERNEL(tstring, double);
REGISTER_KERNEL(tstring, float);
REGISTER_KERNEL(tstring, int32);
REGISTER_KERNEL(tstring, int64_t);
REGISTER_KERNEL(tstring, int8);
REGISTER_KERNEL(tstring, Eigen::half);
REGISTER_KERNEL(tstring, bfloat16);

#undef REGISTER_KERNEL

}  // namespace redis_table
}  // namespace recommenders_addons
}  // namespace tensorflow
