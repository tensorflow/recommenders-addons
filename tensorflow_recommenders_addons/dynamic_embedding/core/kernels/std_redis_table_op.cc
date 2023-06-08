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

// #include "redis_table_op.h"

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
#include <atomic>

// #include "lookup_table_interface.h"
#include <Eigen/Core>
#include "lookup_table_op_registry.h"

#include "uniform_redis_impl/json.h"
#include "uniform_redis_impl/redis_cluster_connection_pool.hpp"
#include "uniform_redis_impl/redis_connection_pool.hpp"
#include "uniform_redis_impl/redis_table_op_util.hpp"

// #include "tensorflow/core/kernels/lookup_table_op.h"
// #include "tensorflow/core/lib/io/buffered_inputstream.h"
// #include "tensorflow/core/lib/io/random_inputstream.h"
// #include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/platform/file_system.h"
// #include "tensorflow/core/platform/path.h"
// #include "tensorflow/core/util/work_sharder.h"
// #include "tensorflow/core/util/env_var.h"

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
using namespace lookup_table;

template <class K, class V>
class RedisLookupTable final : public TFRALookupTableInterface<K, V> {
 private:
  std::vector<int> key_shape_;
  std::vector<int> value_shape_;
  std::string embedding_name_;
  int64_t runtime_value_dim_;

  std::string keys_prefix_name;
  std::string keys_prefix_name_import;
  std::vector<std::string> keys_prefix_name_slices_;
  std::vector<std::string> keys_prefix_name_slices_import_;

  size_t dump_index_ = 0;
  long long dump_cursor_ = 0;

  std::shared_ptr<RedisBaseWrapper<K, V>> table_instance_ = nullptr;

  std::vector<ThreadContext *> threads_Find_;
  std::vector<ThreadContext *> threads_Insert_;
  std::vector<ThreadContext *> threads_Delete_;
  std::mutex threads_Find_mutex_;
  std::mutex threads_Insert_mutex_;
  std::mutex threads_Accum_mutex_;
  std::mutex threads_Delete_mutex_;

  std::vector<aiocb> IMPORT_content;
  std::vector<int> IMPORT_fds_;
  std::vector<unsigned long> IMPORT_fds_sizes;
  std::vector<aiocb> EXPORT_content;
  std::vector<int> EXPORT_fds_;

 public:
  Redis_Connection_Params redis_connection_params_;

 private:
 // todo(chenjinglin) parallel remake
  TFRA_Status launchFind_parallel(std::vector<std::string> &keys_prefix_name_slices,
                           const K *keys, V *values, const V *default_value,
                           const int64_t &total,
                           const int64_t &Velems_per_flat2_dim0,
                           const bool is_full_default,
                           std::vector<ThreadContext *> &threads_Find) {
    // const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    // auto shard = [this, &total, &keys_prefix_name_slices, &keys, &values,
    //               &default_value, &is_full_default, &Velems_per_flat2_dim0,
    //               &threads_Find](int64_t begin, int64_t end) {
    //   const int64_t max_i = std::min(total, end);

    return launchFindCore<K, V>(
                        table_instance_, keys_prefix_name_slices, keys, values,
                        default_value, is_full_default, Velems_per_flat2_dim0,
                        threads_Find, threads_Find_mutex_, 0, total);
    // };
    // int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    // auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    // Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  TFRA_Status launchFind(std::vector<std::string> &keys_prefix_name_slices,
                  const K *keys, V *values, const V *default_value,
                  const int64_t &total, const int64_t &Velems_per_flat2_dim0,
                  const bool is_full_default,
                  std::vector<ThreadContext *> &threads_Find) {
    return launchFindCore<K, V>(table_instance_, keys_prefix_name_slices,
                                  keys, values, default_value, is_full_default,
                                  Velems_per_flat2_dim0, threads_Find,
                                  threads_Find_mutex_, 0, total);
  }

  // todo paralle remake
  TFRA_Status launchFindWithExists_parallel(std::vector<std::string> &keys_prefix_name_slices,
      const K *keys, V *values, const V *default_value, bool *exists,
      const int64_t &total, const int64_t &Velems_per_flat2_dim0,
      const bool is_full_default, std::vector<ThreadContext *> &threads_Find) {
    // const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    // auto shard = [this, &total, &keys_prefix_name_slices, &keys, &values,
    //               &default_value, &exists, &is_full_default,
    //               &Velems_per_flat2_dim0,
    //               &threads_Find](int64_t begin, int64_t end) {
    //   const int64_t max_i = std::min(total, end);

      return launchFindWithExistsCore<K, V>(
                              table_instance_, keys_prefix_name_slices, keys,
                              values, default_value, exists, is_full_default,
                              Velems_per_flat2_dim0, threads_Find,
                              threads_Find_mutex_, 0, total);
    // };
    // int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    // auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    // Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  TFRA_Status launchFindWithExists(std::vector<std::string> &keys_prefix_name_slices,
                            const K *keys, V *values, const V *default_value,
                            bool *exists, const int64_t &total,
                            const int64_t &Velems_per_flat2_dim0,
                            const bool is_full_default,
                            std::vector<ThreadContext *> &threads_Find) {
    return launchFindWithExistsCore<K, V>(
                 table_instance_, keys_prefix_name_slices, keys, values,
                 default_value, exists, is_full_default, Velems_per_flat2_dim0,
                 threads_Find, threads_Find_mutex_, 0, total);
  }

  // todo(chenjinglin) parallelremake
  TFRA_Status launchInsert_parallel(std::vector<std::string> &keys_prefix_name_slices,
                             const K *keys, const V *values,
                             const int64_t &key_num,
                             const int64_t &value_dim,
                             std::vector<ThreadContext *> &threads_Insert) {
    // const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    // auto shard = [this, &total, &keys_prefix_name_slices, &keys, &values,
    //               &Velems_per_flat2_dim0,
    //               &threads_Insert](int64_t begin, int64_t end) {
    //   const int64_t max_i = std::min(total, end);

      return launchInsertCore<K, V>(
                          table_instance_, keys_prefix_name_slices, keys,
                          values, value_dim, threads_Insert,
                          threads_Insert_mutex_, 0, key_num);
    // };
    // int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    // auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    // Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  TFRA_Status launchInsert(std::vector<std::string> &keys_prefix_name_slices,
                    const K *keys, const V *values, const int64_t &key_num,
                    const int64_t &value_dim,
                    std::vector<ThreadContext *> &threads_Insert) {
    return launchInsertCore<K, V>(
                      table_instance_, keys_prefix_name_slices, keys,
                      values, value_dim, threads_Insert,
                      threads_Insert_mutex_, 0, key_num);
}

  // todo(chenjinglin) parallel remake
  TFRA_Status launchAccum_parallel(std::vector<std::string> &keys_prefix_name_slices,
                            const K *keys, const V *values_or_delta,
                            const bool *exists, const int64_t &total,
                            const int64_t &value_dim,
                            std::string &values_dtype_str,
                            std::vector<ThreadContext *> &threads_Insert) {
    // const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    // auto shard = [this, &total, &keys_prefix_name_slices, &keys,
    //               &values_or_delta, &exists, &Velems_per_flat2_dim0,
    //               &values_dtype_str,
    //               &threads_Insert](int64_t begin, int64_t end) {
    //   const int64_t max_i = std::min(total, end);

      return launchAccumCore<K, V>(
                          table_instance_, keys_prefix_name_slices, keys,
                          values_or_delta, exists, value_dim,
                          values_dtype_str, threads_Insert,
                          threads_Accum_mutex_, 0, total);
    // };
    // int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    // auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    // Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  TFRA_Status launchAccum(std::vector<std::string> &keys_prefix_name_slices,
                   const K *keys, const V *values_or_delta, const bool *exists,
                   const int64_t &total, const int64_t &value_dim,
                   std::string &values_dtype_str,
                   std::vector<ThreadContext *> &threads_Insert) {
    return launchAccumCore<K, V>(
                        table_instance_, keys_prefix_name_slices, keys,
                        values_or_delta, exists, value_dim,
                        values_dtype_str, threads_Insert,
                        threads_Insert_mutex_, 0, total);
  }

  TFRA_Status launchDelete_parallel(std::vector<std::string> &keys_prefix_name_slices,
                             const K *keys, const int64_t &total,
                             std::vector<ThreadContext *> &threads_Delete) {
    // const int64_t max_parallelism = (total / multi_redis_cmd_max_argc) + 1;

    // auto shard = [this, &total, &keys_prefix_name_slices, &keys,
    //               &threads_Delete](int64_t begin, int64_t end) {
    //   const int64_t max_i = std::min(total, end);

    return launchDeleteCore<K, V>(table_instance_, keys_prefix_name_slices,
                                            keys, threads_Delete,
                                            threads_Delete_mutex_, 0, total);
    // };
    // int64_t slices_size = std::min(total, multi_redis_cmd_max_argc - 1);
    // auto &worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
    // Shard(max_parallelism, worker_threads.workers, total, slices_size, shard);
  }

  TFRA_Status launchDelete(std::vector<std::string> &keys_prefix_name_slices,
                    const K *keys, const int64_t &total,
                    std::vector<ThreadContext *> &threads_Delete) {
    return launchDeleteCore<K, V>(
                            table_instance_, keys_prefix_name_slices, keys,
                            threads_Delete, threads_Delete_mutex_, 0, total);
  }

 public:
  // RedisLookupTable(OpKernelContext *ctx, OpKernel *kernel) {
  //   OP_REQUIRES_OK(ctx,
  //                  GetNodeAttr(kernel->def(), "value_shape", &value_shape_));
  //   OP_REQUIRES(
  //       ctx, TensorShapeUtils::IsVector(value_shape_),
  //       errors::InvalidArgument("Default value must be a vector, got shape ",
  //                               value_shape_.DebugString()));

  //   OP_REQUIRES_OK(
  //       ctx, GetNodeAttr(kernel->def(), "embedding_name", &embedding_name));

  //   std::string redis_config_abs_dir_tem;
  //   OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_config_abs_dir",
  //                                   &redis_config_abs_dir_tem));

  //   std::string redis_config_abs_dir_env;
  //   OP_REQUIRES_OK(ctx, GetNodeAttr(kernel->def(), "redis_config_abs_dir_env",
  //                                   &redis_config_abs_dir_env));
  //   Status spec_config_status = ReadStringFromEnvVar(
  //       redis_config_abs_dir_env, "NotFound", &redis_config_abs_dir);
  //   if (redis_config_abs_dir != "NotFound") {
  //     if (get_file_size(redis_config_abs_dir) > 0) {
  //       LOG(INFO)
  //           << "Read TFRA Redis config file path from the environment variable "
  //           << redis_config_abs_dir_env << " successfully. Config file path is "
  //           << redis_config_abs_dir;
  //     } else {
  //       ctx->CtxFailure(
  //           errors::NotFound("Can not find the file " + redis_config_abs_dir +
  //                            ". Please check carefully again if the file exist "
  //                            "or unset the evironment viriable " +
  //                            redis_config_abs_dir_env));
  //     }
  //   } else {
  //     LOG(WARNING) << "Fails to read the TFRA Redis config file path from the "
  //                     "environment variable "
  //                  << redis_config_abs_dir_env
  //                  << " which read from OP attribute redis_config_abs_dir_env "
  //                     "firstly, now try to read config file path from global "
  //                     "environment variable TFRA_REDIS_CONFIG_PATH.";
  //     Status global_config_status = ReadStringFromEnvVar(
  //         "TFRA_REDIS_CONFIG_PATH", "NotFound", &redis_config_abs_dir);
  //     if (redis_config_abs_dir != "NotFound" &&
  //         get_file_size(redis_config_abs_dir) <= 0) {
  //       ctx->CtxFailure(errors::NotFound(
  //           "Can not find the file " + redis_config_abs_dir +
  //           ". Please check carefully again if the file exist or unset the "
  //           "evironment viriable TFRA_REDIS_CONFIG_PATH"));
  //     } else if (redis_config_abs_dir == "NotFound") {
  //       LOG(WARNING)
  //           << "Fails to read the TFRA Redis config file path from the "
  //              "global environment variable TFRA_REDIS_CONFIG_PATH firstly, "
  //              "config file path is "
  //           << redis_config_abs_dir
  //           << ". now the config file path which assigned OP attribute "
  //              "redis_config_abs_dir "
  //           << redis_config_abs_dir_tem << " is used.";
  //       redis_config_abs_dir = redis_config_abs_dir_tem;
  //     } else {
  //       LOG(INFO) << "TFRA Redis config file path is " << redis_config_abs_dir;
  //     }
  //   }

  //   if (get_file_size(redis_config_abs_dir) > 0) {
  //     OP_REQUIRES_OK(ctx, ParseJsonConfig(&redis_config_abs_dir,
  //                                         &redis_connection_params_));
  //   } else {
  //     ctx->CtxFailure(errors::NotFound("Unable to find config file with path: ",
  //                                      redis_config_abs_dir));
  //   }

  //   const int64_t &&default_value_width = value_shape_.dim_size(0);
  //   const int64_t &&default_value_total = value_shape_.num_elements();
  //   if (default_value_width == default_value_total) {
  //     runtime_value_dim_ = default_value_width;
  //   } else {
  //     LOG(WARNING) << "The num_elements in default_value dosen't equal to the "
  //                     "dim_size(0) in default_value. Using the num_elements as "
  //                     "output tensor dim two now.";
  //     runtime_value_dim_ = default_value_total;
  //   }

  //   std::vector<std::pair<unsigned, unsigned>> cluster_slots;

  //   // creat redis instance
  //   switch (redis_connection_params_.redis_connection_mode) {
  //     case ClusterMode: {
  //       multi_redis_cmd_max_argc = redis_connection_params_.keys_sending_size *
  //                                  redis_connection_params_.storage_slice;
  //       table_instance_ = RedisWrapper<RedisCluster, K, V>::get_instance();
  //       OP_REQUIRES_OK(ctx,
  //                      table_instance_->set_params(redis_connection_params_));
  //       if (redis_connection_params_.using_hash_storage_slice) {
  //         OP_REQUIRES_OK(ctx, table_instance_->set_K_bucket_num_handle(
  //                                 KBucketNumCRC32Handle));
  //       } else {
  //         OP_REQUIRES_OK(ctx, table_instance_->set_K_bucket_num_handle(
  //                                 KBucketNumCommonHandle<K>));
  //       }
  //       OP_REQUIRES_OK(ctx, table_instance_->Conn());
  //       if (redis_connection_params_.redis_hash_tags_hypodispersion == false)
  //         cluster_slots = table_instance_->ClusterNodesSlots(false);
  //       break;
  //     }
  //     case SentinelMode: {
  //       multi_redis_cmd_max_argc =
  //           redis_connection_params_.keys_sending_size * 1;
  //       table_instance_ = RedisWrapper<Redis, K, V>::get_instance();
  //       OP_REQUIRES_OK(ctx,
  //                      table_instance_->set_params(redis_connection_params_));
  //       if (redis_connection_params_.using_hash_storage_slice) {
  //         OP_REQUIRES_OK(ctx, table_instance_->set_K_bucket_num_handle(
  //                                 KBucketNumCRC32Handle));
  //       } else {
  //         OP_REQUIRES_OK(ctx, table_instance_->set_K_bucket_num_handle(
  //                                 KBucketNumCommonHandle<K>));
  //       }
  //       OP_REQUIRES_OK(ctx, table_instance_->Conn());
  //       break;
  //     }
  //     case StandaloneMode: {
  //       multi_redis_cmd_max_argc =
  //           redis_connection_params_.keys_sending_size * 1;
  //       table_instance_ = RedisWrapper<Redis, K, V>::get_instance(false);
  //       OP_REQUIRES_OK(ctx,
  //                      table_instance_->set_params(redis_connection_params_));
  //       if (redis_connection_params_.using_hash_storage_slice) {
  //         OP_REQUIRES_OK(ctx, table_instance_->set_K_bucket_num_handle(
  //                                 KBucketNumCRC32Handle));
  //       } else {
  //         OP_REQUIRES_OK(ctx, table_instance_->set_K_bucket_num_handle(
  //                                 KBucketNumCommonHandle<K>));
  //       }
  //       OP_REQUIRES_OK(ctx, table_instance_->Conn());
  //       break;
  //     }
  //     default: {
  //       LOG(ERROR) << "There are only three Redis connection modes, which "
  //                     "Cluster=0/Sentinel=1/Standalone=2.";
  //       ctx->CtxFailure(errors::InvalidArgument(
  //           std::to_string(redis_connection_params_.redis_connection_mode) +
  //           " is illegal redis_connection_mode."));
  //       break;
  //     }
  //   }

  //   CreateKeysPrefixNameHandle(cluster_slots, &redis_connection_params_,
  //                              embedding_name, keys_prefix_name,
  //                              keys_prefix_name_import, keys_prefix_name_slices_,
  //                              keys_prefix_name_slices_import);

  //   // Rehash buckets
  //   auto keys_prefix_name_slices_import_sort = keys_prefix_name_slices_import;
  //   std::sort(keys_prefix_name_slices_import_sort.begin(),
  //             keys_prefix_name_slices_import_sort.end());
  //   auto keys_prefix_name_slices_sort = keys_prefix_name_slices_;
  //   std::sort(keys_prefix_name_slices_sort.begin(),
  //             keys_prefix_name_slices_sort.end());
  //   if (redis_connection_params_.model_tag_import ==
  //           redis_connection_params_.model_tag_runtime &&
  //       (keys_prefix_name_slices_import_sort != keys_prefix_name_slices_sort ||
  //        redis_connection_params_.storage_slice_import !=
  //            static_cast<int>(redis_connection_params_.storage_slice))) {
  //     auto keys_prefix_name_slices_redis =
  //         table_instance_->GetKeyBucketsAndOptimizerParamsWithName(
  //             keys_prefix_name_import, true);
  //     auto keys_prefix_name_slices_redis_sort = keys_prefix_name_slices_redis;
  //     std::sort(keys_prefix_name_slices_redis_sort.begin(),
  //               keys_prefix_name_slices_redis_sort.end());
  //     LOG(INFO) << "Arrange the new Redis hash tags to the table "
  //               << keys_prefix_name_import
  //               << ". And remove the old one. Remember changing config file "
  //                  "next time!";
  //     if (keys_prefix_name_slices_redis.size() ==
  //         redis_connection_params_.storage_slice) {
  //       if (keys_prefix_name_slices_redis_sort ==
  //           keys_prefix_name_slices_import_sort) {
  //         OP_REQUIRES_OK(ctx, table_instance_->DuplicateInRedis(
  //                                 keys_prefix_name_slices_import,
  //                                 keys_prefix_name_slices_));
  //         for (auto keys_prefix_name_slice_import :
  //              keys_prefix_name_slices_import) {
  //           OP_REQUIRES_OK(ctx, table_instance_->RemoveHkeysInBuckets(
  //                                   keys_prefix_name_slice_import));
  //         }
  //       }
  //     } else {
  //       if (keys_prefix_name_slices_redis_sort !=
  //           keys_prefix_name_slices_import_sort) {
  //         std::stringstream warning_print;
  //         for (auto keys_prefix_name_slice_import :
  //              keys_prefix_name_slices_import) {
  //           warning_print << keys_prefix_name_slice_import << " , ";
  //         }
  //         warning_print << "; And the keys_prefix_name_slices_redis are: ";
  //         for (auto keys_prefix_name_slice_redis :
  //              keys_prefix_name_slices_redis) {
  //           warning_print << keys_prefix_name_slice_redis << " , ";
  //         }
  //         warning_print << " . Now try to replace imported "
  //                          "keys_prefix_name_slices with those in Redis. "
  //                       << std::endl;
  //         LOG(WARNING) << "Hashtag in Redis for " << keys_prefix_name_import
  //                      << " is not equal to the imported one. Imported "
  //                         "keys_prefix_name_slices are: "
  //                      << warning_print.str();
  //         keys_prefix_name_slices_import.swap(keys_prefix_name_slices_redis);
  //       }
  //       LOG(WARNING)
  //           << "The embedding table prefix name " << keys_prefix_name_import
  //           << " has already been saved in the Redis Servers. "
  //           << "And its number of slices is not equal to the number you putted "
  //              "in the setting. ";
  //       LOG(INFO) << "Try to recreate the embedding table "
  //                 << keys_prefix_name_import << " into the bucket number "
  //                 << redis_connection_params_.storage_slice << " for"
  //                 << keys_prefix_name;
  //       OP_REQUIRES_OK(ctx, ReCreateTableBuckets(ctx, keys_prefix_name_import));
  //     }

  //     if (table_instance_->CheckSlicesNum(keys_prefix_name) != 1) {
  //       LOG(WARNING)
  //           << "CheckSlicesNum fails after many operations. Please check your "
  //              "Redis service manually. If you are in the first few steps of "
  //              "your training, you can ignore this warning.";
  //     }
  //   }

  //   // remove expiring time of buckets
  //   OP_REQUIRES_OK(ctx, table_instance_->SetPersistBuckets(keys_prefix_name));

  //   // allocate the memory of threads helper
  //   for (size_t i = 0; i < hardware_concurrency_; ++i) {
  //     threads_Find.emplace_back(new ThreadContext());
  //     threads_Insert.emplace_back(new ThreadContext());
  //     threads_Delete.emplace_back(new ThreadContext());
  //   }
  // }

  RedisLookupTable(const KVTableInfo& info) {
    this->Init(info);
  }

  TFRA_Status Init(const KVTableInfo& info) {
    TFRA_Status status = TFRA_Status::OK();
    value_shape_ = info.value_shape_;
    embedding_name_ = info.embedding_name_;

    std::string redis_config_abs_dir_tem;
    std::string redis_config_abs_dir_env;
    std::string redis_config_abs_dir;

    // if (info.redis_config_) {
    //   redis_config_abs_dir_tem = info.redis_config_.redis_config_abs_dir_;
    //   redis_config_abs_dir_env  = info.redis_config_.redis_config_abs_dir_env_;
    // }

    redis_config_abs_dir_tem = info.redis_config_.redis_config_abs_dir_;
    redis_config_abs_dir_env  = info.redis_config_.redis_config_abs_dir_env_; 

    ReadStringFromEnvVar(redis_config_abs_dir_env, "NotFound", &redis_config_abs_dir);
    if (redis_config_abs_dir != "NotFound") {
      if (get_file_size(redis_config_abs_dir) <= 0) {
      //   LOG(INFO)
      //       << "Read TFRA Redis config file path from the environment variable "
      //       << redis_config_abs_dir_env << " successfully. Config file path is "
      //       << redis_config_abs_dir;
      // } else {
        return TFRA_Status(StatusCode::NOT_FOUND, "Can not find the file " + redis_config_abs_dir +
                             ". Please check carefully again if the file exist " +
                             "or unset the evironment viriable " +
                             redis_config_abs_dir_env);
      }
    } else {
      // LOG(WARNING) << "Fails to read the TFRA Redis config file path from the "
      //                 "environment variable "
      //              << redis_config_abs_dir_env
      //              << " which read from OP attribute redis_config_abs_dir_env "
      //                 "firstly, now try to read config file path from global "
      //                 "environment variable TFRA_REDIS_CONFIG_PATH.";
      ReadStringFromEnvVar("TFRA_REDIS_CONFIG_PATH", "NotFound", &redis_config_abs_dir);
      if (redis_config_abs_dir != "NotFound" && get_file_size(redis_config_abs_dir) <= 0) {
        return TFRA_Status(StatusCode::NOT_FOUND, "Can not find the file " + redis_config_abs_dir +
            ". Please check carefully again if the file exist or unset the " +
            "evironment viriable TFRA_REDIS_CONFIG_PATH");
      } else if ("NotFound" == redis_config_abs_dir) {
        // LOG(WARNING)
        //     << "Fails to read the TFRA Redis config file path from the "
        //        "global environment variable TFRA_REDIS_CONFIG_PATH firstly, "
        //        "config file path is "
        //     << redis_config_abs_dir
        //     << ". now the config file path which assigned OP attribute "
        //        "redis_config_abs_dir "
        //     << redis_config_abs_dir_tem << " is used.";
        redis_config_abs_dir = redis_config_abs_dir_tem;
      }
      // else {
      //   LOG(INFO) << "TFRA Redis config file path is " << redis_config_abs_dir;
      // }
    }

    if (get_file_size(redis_config_abs_dir) > 0) {
      // OP_REQUIRES_OK(ctx, ParseJsonConfig(&redis_config_abs_dir,
      //                                     &redis_connection_params_));
      status = ParseJsonConfig(&redis_config_abs_dir, &redis_connection_params_);
      if (!status.ok()) {
        // log
        return status;
      }
    } else {
      return TFRA_Status(StatusCode::NOT_FOUND, 
                         "Unable to find config file with path: " + redis_config_abs_dir);
    }

    // const int64_t &&default_value_width = value_shape_.dim_size(0);
    // const int64_t &&default_value_total = value_shape_.num_elements();

    const int64_t &&default_value_width = value_shape_[0];
    int64_t default_value_total = 0;
    for (auto value_with : value_shape_) {
      default_value_total += value_with;
    }
    // const int64_t &&default_value_total = value_shape_.size();
    if (default_value_width == default_value_total) {
      runtime_value_dim_ = default_value_width;
    } else {
      // LOG(WARNING) << "The num_elements in default_value dosen't equal to the "
      //                 "dim_size(0) in default_value. Using the num_elements as "
      //                 "output tensor dim two now.";
      runtime_value_dim_ = default_value_total;
    }

    std::vector<std::pair<unsigned, unsigned>> cluster_slots;

    // create redis instance
    switch (redis_connection_params_.redis_connection_mode) {
      case ClusterMode: {
        multi_redis_cmd_max_argc = redis_connection_params_.keys_sending_size *
                                   redis_connection_params_.storage_slice;
        table_instance_ = RedisWrapper<RedisCluster, K, V>::get_instance();
        TFRA_REQUIRES_OK(table_instance_->set_params(redis_connection_params_));
        if (redis_connection_params_.using_hash_storage_slice) {
          TFRA_REQUIRES_OK(table_instance_->set_K_bucket_num_handle(KBucketNumCRC32Handle));
        } else {
          TFRA_REQUIRES_OK(table_instance_->set_K_bucket_num_handle(KBucketNumCommonHandle<K>));
        }
        TFRA_REQUIRES_OK(table_instance_->Conn());
        if (redis_connection_params_.redis_hash_tags_hypodispersion == false)
          cluster_slots = table_instance_->ClusterNodesSlots(false);
        break;
      }
      case SentinelMode: {
        multi_redis_cmd_max_argc =
            redis_connection_params_.keys_sending_size * 1;
        table_instance_ = RedisWrapper<Redis, K, V>::get_instance();
        TFRA_REQUIRES_OK(table_instance_->set_params(redis_connection_params_));
        if (redis_connection_params_.using_hash_storage_slice) {
          TFRA_REQUIRES_OK(table_instance_->set_K_bucket_num_handle(KBucketNumCRC32Handle));
        } else {
          TFRA_REQUIRES_OK(table_instance_->set_K_bucket_num_handle(KBucketNumCommonHandle<K>));
        }
        TFRA_REQUIRES_OK(table_instance_->Conn());
        break;
      }
      case StandaloneMode: {
        multi_redis_cmd_max_argc =
            redis_connection_params_.keys_sending_size * 1;
        table_instance_ = RedisWrapper<Redis, K, V>::get_instance(false);
        TFRA_REQUIRES_OK(table_instance_->set_params(redis_connection_params_));
        if (redis_connection_params_.using_hash_storage_slice) {
          TFRA_REQUIRES_OK(table_instance_->set_K_bucket_num_handle(KBucketNumCRC32Handle));
        } else {
          TFRA_REQUIRES_OK(table_instance_->set_K_bucket_num_handle(KBucketNumCommonHandle<K>));
        }
        TFRA_REQUIRES_OK(table_instance_->Conn());
        break;
      }
      default: {
        // LOG(ERROR) << "There are only three Redis connection modes, which "
        //               "Cluster=0/Sentinel=1/Standalone=2.";
        return TFRA_Status(StatusCode::INVALID_ARGUMENT, 
                            std::to_string(redis_connection_params_.redis_connection_mode) +
                            " is illegal redis_connection_mode.");
        break;
      }
    }

    CreateKeysPrefixNameHandle(cluster_slots, &redis_connection_params_,
                               embedding_name_, keys_prefix_name,
                               keys_prefix_name_import, keys_prefix_name_slices_,
                               keys_prefix_name_slices_import_);

    // Rehash buckets
    auto keys_prefix_name_slices_import_sort = keys_prefix_name_slices_import_;
    std::sort(keys_prefix_name_slices_import_sort.begin(),
              keys_prefix_name_slices_import_sort.end());
    auto keys_prefix_name_slices_sort = keys_prefix_name_slices_;
    std::sort(keys_prefix_name_slices_sort.begin(),
              keys_prefix_name_slices_sort.end());
    if (redis_connection_params_.model_tag_import ==
            redis_connection_params_.model_tag_runtime &&
        (keys_prefix_name_slices_import_sort != keys_prefix_name_slices_sort ||
         redis_connection_params_.storage_slice_import !=
             static_cast<int>(redis_connection_params_.storage_slice))) {
      auto keys_prefix_name_slices_redis =
          table_instance_->GetKeyBucketsAndOptimizerParamsWithName(
              keys_prefix_name_import, true);
      auto keys_prefix_name_slices_redis_sort = keys_prefix_name_slices_redis;
      std::sort(keys_prefix_name_slices_redis_sort.begin(),
                keys_prefix_name_slices_redis_sort.end());
      // LOG(INFO) << "Arrange the new Redis hash tags to the table "
      //           << keys_prefix_name_import
      //           << ". And remove the old one. Remember changing config file "
      //              "next time!";
      if (keys_prefix_name_slices_redis.size() ==
          redis_connection_params_.storage_slice) {
        if (keys_prefix_name_slices_redis_sort ==
            keys_prefix_name_slices_import_sort) {
          TFRA_REQUIRES_OK(table_instance_->DuplicateInRedis(
                                  keys_prefix_name_slices_import_,
                                  keys_prefix_name_slices_));
          for (auto keys_prefix_name_slice_import : keys_prefix_name_slices_import_) {
            TFRA_REQUIRES_OK(table_instance_->RemoveHkeysInBuckets(
                                    keys_prefix_name_slice_import));
          }
        }
      } else {
        if (keys_prefix_name_slices_redis_sort !=
            keys_prefix_name_slices_import_sort) {
          std::stringstream warning_print;
          for (auto keys_prefix_name_slice_import : keys_prefix_name_slices_import_) {
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
          // LOG(WARNING) << "Hashtag in Redis for " << keys_prefix_name_import
          //              << " is not equal to the imported one. Imported "
          //                 "keys_prefix_name_slices are: "
          //              << warning_print.str();
          keys_prefix_name_slices_import_.swap(keys_prefix_name_slices_redis);
        }
        // LOG(WARNING)
        //     << "The embedding table prefix name " << keys_prefix_name_import
        //     << " has already been saved in the Redis Servers. "
        //     << "And its number of slices is not equal to the number you putted "
        //        "in the setting. ";
        // LOG(INFO) << "Try to recreate the embedding table "
        //           << keys_prefix_name_import << " into the bucket number "
        //           << redis_connection_params_.storage_slice << " for"
        //           << keys_prefix_name;
        TFRA_REQUIRES_OK(ReCreateTableBuckets(keys_prefix_name_import));
      }

      if (table_instance_->CheckSlicesNum(keys_prefix_name) != 1) {
        // LOG(WARNING)
        //     << "CheckSlicesNum fails after many operations. Please check your "
        //        "Redis service manually. If you are in the first few steps of "
        //        "your training, you can ignore this warning.";
      }
    }

    // remove expiring time of buckets
    TFRA_REQUIRES_OK(table_instance_->SetPersistBuckets(keys_prefix_name));

    // allocate the memory of threads helper
    for (size_t i = 0; i < hardware_concurrency_; ++i) {
      threads_Find_.emplace_back(new ThreadContext());
      threads_Insert_.emplace_back(new ThreadContext());
      threads_Delete_.emplace_back(new ThreadContext());
    }

    return status;
  }

  ~RedisLookupTable() {
    if (table_instance_ != nullptr && table_instance_->isRedisConnect == true) {
      table_instance_->SetExpireBuckets(keys_prefix_name);
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
    for (auto &threads_Find_i : threads_Find_) {
      if (threads_Find_i->thread_occupied.load(std::memory_order_consume) ==
          false) {
        threads_Find_i->HandleRelease();
      }
    }
    for (auto &threads_Insert_i : threads_Insert_) {
      if (threads_Insert_i->thread_occupied.load(std::memory_order_consume) ==
          false) {
        threads_Insert_i->HandleRelease();
      }
    }
    for (auto &threads_Delete_i : threads_Delete_) {
      if (threads_Delete_i->thread_occupied.load(std::memory_order_consume) ==
          false) {
        threads_Delete_i->HandleRelease();
      }
    }

    if (table_instance_ != nullptr) {
      table_instance_.reset();
    }
  }

  TFRA_Status ReCreateTableBuckets(const std::string &keys_prefix_name_from) {
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> hscan_reply;
    const redisReply *kvs_reply;
    std::vector<std::string> keys_prefix_name_slices_in_redis =
        table_instance_->GetKeyBucketsAndOptimizerParamsWithName(
            keys_prefix_name_from, false);
    // Tensor keys_temp;
    // Tensor values_temp;
    std::unique_ptr<K> keys_temp_ptr(nullptr);
    std::unique_ptr<V> values_temp_ptr(nullptr);
    const K *pk_raw;
    const V *pv_raw;
    int64_t slice_keys_size = 0;
    long long cursor = 0;

    redisReply *temp_reply;
    for (size_t i = 0; i < keys_prefix_name_slices_in_redis.size(); ++i) {
      slice_keys_size = table_instance_->TableSizeInBucket(
          keys_prefix_name_slices_in_redis[i]);
      // fill Tensor keys_temp
      try {
        keys_temp_ptr.reset(new K[slice_keys_size]);
        values_temp_ptr.reset(new V[slice_keys_size * runtime_value_dim_]);
        pk_raw = keys_temp_ptr.get();
        pv_raw = values_temp_ptr.get();
        // TF_RETURN_IF_ERROR(ctx->allocate_temp(TFRA_DataTypeToEnum<K>::v(),
        //                                       TensorShape({slice_keys_size}),
        //                                       &keys_temp));
        // TF_RETURN_IF_ERROR(ctx->allocate_temp(
        //     TFRA_DataTypeToEnum<V>::v(),
        //     TensorShape({slice_keys_size, runtime_value_dim_}), &values_temp));
        // pk_raw = reinterpret_cast<const K *>(keys_temp.tensor_data().data());
        // pv_raw = reinterpret_cast<const V *>(values_temp.tensor_data().data());
        cursor = 0;
        while (true) {
          hscan_reply.reset();
          hscan_reply = std::move(table_instance_->HscanGetKeysValsInBucket(
              keys_prefix_name_slices_in_redis[i], &cursor,
              multi_redis_cmd_max_argc));
          if (hscan_reply == nullptr) {
            return TFRA_Status(StatusCode::UNKNOWN,
                "Unknown errors happen when HscanGetKeysValsInBucket in ReCreateTableBuckets");
          }
          if (hscan_reply->type == REDIS_REPLY_ARRAY &&
              hscan_reply->elements > 1) {
            kvs_reply = hscan_reply->element[1];
            // fill Tensor keys and values
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

          // LOG(INFO) << "The cursor of scanning "
          //           << keys_prefix_name_slices_in_redis[i]
          //           << " in ReCreateTableBuckets is " << cursor << " now.";
          if (cursor == 0) {
            break;
          }
        }
      } catch (const std::exception &err) {
        // LOG(ERROR) << "Some errors happened when try to copy Redis old buckets "
        //               "data reply into tensor for preparing to insert"
        //            << " -- " << err.what();
        return TFRA_Status(StatusCode::UNKNOWN,
                "Unknown errors happen when HscanGetKeysValsInBucket in ReCreateTableBuckets");
      }
      try {
        // insert KV pair into new Redis with new storage_slice
        launchInsert(keys_prefix_name_slices_, pk_raw, pv_raw,
                     slice_keys_size, runtime_value_dim_, threads_Insert_);
      } catch (const std::exception &err) {
        // LOG(ERROR)
        //     << "Some errors happened when try to insert new buckets into Redis"
        //     << " -- " << err.what();
        return TFRA_Status(StatusCode::UNKNOWN, err.what());
      }
    }
    for (auto keys_prefix_name_slice_in_redis :
         keys_prefix_name_slices_in_redis) {
      // LOG(INFO) << "Now try to delet old bucket "
      //           << keys_prefix_name_slice_in_redis;
      auto iter = std::find(keys_prefix_name_slices_.begin(),
                            keys_prefix_name_slices_.end(),
                            keys_prefix_name_slice_in_redis);
      if (iter == keys_prefix_name_slices_.end()) {
        TFRA_REQUIRES_OK(table_instance_->RemoveHkeysInBuckets(keys_prefix_name_slice_in_redis));
      }
    }
    return TFRA_Status::OK();
  }

  size_t size() const override {
    size_t size = 0;
    const unsigned &storage_slice = redis_connection_params_.storage_slice;
    for (unsigned i = 0; i != storage_slice && i < keys_prefix_name_slices_.size(); ++i) {
      size += table_instance_->TableSizeInBucket(keys_prefix_name_slices_[i]);
    }
    return size;
  }

  int64_t dim() const override {
    return runtime_value_dim_;
  }

  TFRA_Status Find(const K* keys, int64_t key_num,
                   V* values, int64_t value_dim,
                   const V* default_values, int64_t default_value_num) override {
    // int64_t total = keys.NumElements();
    // if (total > 0) {
    //   const int64_t Velems_per_flat2_dim0 = values->NumElements() / total;
    //   const bool is_full_default =
    //       (values->NumElements() == default_value.NumElements());
    //   if (total < (multi_redis_cmd_max_argc - 1)) {
    //     launchFind(ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
    //                (V *)values->tensor_data().data(),
    //                (V *)default_value.tensor_data().data(), total,
    //                Velems_per_flat2_dim0, is_full_default, threads_Find);
    //   } else {
    //     // redis commmand args > multi_redis_cmd_max_argc
    //     launchFind_parallel(
    //         ctx, keys_prefix_name_slices, (K *)keys.tensor_data().data(),
    //         (V *)values->tensor_data().data(),
    //         (V *)default_value.tensor_data().data(), total,
    //         Velems_per_flat2_dim0, is_full_default, threads_Find);
    //   }
    // }

    if (key_num > 0) {
      // const int64_t Velems_per_flat2_dim0 = values_num_elemenst / keys_num_elements;
      const int64_t Velems_per_flat2_dim0 = value_dim;

      // const bool is_full_default =
      //     (values_num_elemenst == default_values_num_elements);
      const bool is_full_default = default_value_num == value_dim * key_num;
      if (key_num < (multi_redis_cmd_max_argc - 1)) {
        launchFind(keys_prefix_name_slices_, keys, values, default_values, key_num,
                   Velems_per_flat2_dim0, is_full_default, threads_Find_);
      } else {
        // redis commmand args > multi_redis_cmd_max_argc
        launchFind_parallel(keys_prefix_name_slices_, keys, values, default_values,
                            key_num, Velems_per_flat2_dim0, is_full_default, threads_Find_);
      }
    }

    return TFRA_Status::OK();
  }

  TFRA_Status FindWithExists(const K* keys, int64_t key_num,
                             V* values, int64_t value_dim, 
                             const V* default_values, int64_t default_value_num,
                             bool* exists) {
    if (key_num > 0) {
      // const int64_t Velems_per_flat2_dim0 = values_num_elements / keys_num_elements;
      const int64_t Velems_per_flat2_dim0 = value_dim;

      // const bool is_full_default =
      //     (values_num_elements == default_values_num_elements);
      const bool is_full_default = default_value_num == key_num * value_dim;
      if (key_num < (multi_redis_cmd_max_argc - 1)) {
        launchFindWithExists(keys_prefix_name_slices_, keys, values, default_values,
                             exists, key_num, Velems_per_flat2_dim0, is_full_default, threads_Find_);
      } else {
        // redis commmand args > multi_redis_cmd_max_argc
        launchFindWithExists_parallel(keys_prefix_name_slices_, keys, values, default_values,
                exists, key_num, Velems_per_flat2_dim0, is_full_default, threads_Find_);
      }
    }
    return TFRA_Status::OK();
  }

  TFRA_Status DoInsert(bool clear, const K* keys, const V* values, const int64_t key_num,
                      const int64_t value_dim) {
    if (clear) {
      for (auto keys_prefix_name_slice : keys_prefix_name_slices_) {
        TFRA_REQUIRES_OK(table_instance_->RemoveHkeysInBuckets(keys_prefix_name_slice));
      }
    }
    if (key_num < (multi_redis_cmd_max_argc - 1)) {
      launchInsert(keys_prefix_name_slices_, keys, values, key_num,
                   value_dim, threads_Insert_);
    } else {
      launchInsert_parallel(keys_prefix_name_slices_, keys, values, key_num, value_dim, 
                            threads_Insert_);  // redis commmand args > multi_redis_cmd_max_argc
    }
    return TFRA_Status::OK();
  }

  TFRA_Status Insert(const K* keys, int64_t key_num,
                     const V* values, int64_t value_dim) override {
    if (key_num > 0) {
      // const int64_t Velems_per_flat2_dim0 = values_num_elements / keys_num_elements;
      const int64_t Velems_per_flat2_dim0 = value_dim;

      return DoInsert(false, keys, values, key_num, Velems_per_flat2_dim0);
    } else {
      // LOG(INFO) << "Redis Backend Insert nothing for empty input keys tensor.";
      return TFRA_Status::OK();
    }
  }

  TFRA_Status Accum(const K* keys, int64_t key_num,
                    const V* values_or_delta, int64_t vod_num,
                    const bool* exists) {
    // const int64_t Velems_per_flat2_dim0 =
    //     values_or_delta_num_elements / keys_num_elements;
    const int64_t Velems_per_flat2_dim0 = vod_num / key_num;

    auto values_dtype_str = TFRA_DataTypeString(value_dtype());

    if (key_num < (multi_redis_cmd_max_argc - 1)) {
      launchAccum(keys_prefix_name_slices_, keys, values_or_delta, exists,
                  key_num, Velems_per_flat2_dim0, values_dtype_str, threads_Insert_);
    } else {
      launchAccum_parallel(keys_prefix_name_slices_, keys, values_or_delta, exists, 
                          key_num, Velems_per_flat2_dim0, values_dtype_str,
                          threads_Insert_);  // redis commmand args > multi_redis_cmd_max_argc
    }

    return TFRA_Status::OK();
  }

  TFRA_Status Remove(const K* keys, int64_t key_num) override {
    if (key_num > 0) {
      if (key_num < (multi_redis_cmd_max_argc - 1)) {
        launchDelete(keys_prefix_name_slices_, keys, key_num, threads_Delete_);
      } else {
        // redis commmand args > multi_redis_cmd_max_argc
        launchDelete_parallel(keys_prefix_name_slices_, keys, key_num, threads_Delete_);
      }
    }
    return TFRA_Status::OK();
  }

  TFRA_Status Clear() {
    for (auto keys_prefix_name_slice : keys_prefix_name_slices_) {
      TFRA_REQUIRES_OK(table_instance_->RemoveHkeysInBuckets(keys_prefix_name_slice));
    }
    return TFRA_Status::OK();
  }

  TFRA_Status ImportValues(const K* keys, int64_t key_num,
                           const V* values, int64_t value_dim) override {
    if (redis_connection_params_.table_store_mode == 1) {
      // When there is not a corresponding table existing in Redis service and
      // table_store_mode==1, try to restore from a Redis binary dump files
      // which paths are directory
      // '[model_lib_abs_dir]/[model_tag]/[name].rdb'.
      return ImportValuesFromFiles();
    } else {
      if (key_num > 0 && redis_connection_params_.table_store_mode == 0) {
        return Insert(keys, key_num, values, value_dim);
      } else {
        // LOG(INFO) << "Import nothing from the TensorFlow saved model to Redis "
        //              "service for "
        //           << keys_prefix_name_import;
        if (redis_connection_params_.model_tag_import !=
            redis_connection_params_.model_tag_runtime) {
          if (table_instance_->CheckSlicesNum(keys_prefix_name_import) == 1 &&
              table_instance_->CheckSlicesNum(keys_prefix_name) != 1) {
            // LOG(INFO) << "Because model_tag_import is not equal to "
            //              "model_tag_runtime. Now begin to DuplicateInRedis, "
            //              "remember changing config file next time!";
            return table_instance_->DuplicateInRedis(
                keys_prefix_name_slices_import_, keys_prefix_name_slices_);
          }
        }
        return TFRA_Status::OK();
      }
    }
  }

  TFRA_Status ImportValuesFromFiles() {
    std::string file_path, folder_dir;
    const unsigned &storage_slice = redis_connection_params_.storage_slice;

    IMPORT_content.resize(storage_slice);
    IMPORT_fds_.clear();
    IMPORT_fds_.reserve(storage_slice);
    IMPORT_fds_sizes.clear();
    IMPORT_fds_sizes.reserve(storage_slice);

    folder_dir = check_dir(redis_connection_params_.model_lib_abs_dir);
    folder_dir =
        check_dir(folder_dir + redis_connection_params_.model_tag_import);

    for (unsigned i = 0; i < storage_slice; ++i) {
      file_path = folder_dir + keys_prefix_name_slices_import_[i] + ".rdb";
      if (access(file_path.c_str(), 0) == -1) {
        // LOG(WARNING) << "file " << file_path
        //              << " doesn't exist. Using the table that already exist in "
        //                 "the Redis or creating a new one";
      } else {
        IMPORT_fds_.push_back(open(file_path.c_str(), O_RDONLY));
        IMPORT_fds_sizes.push_back(get_file_size(file_path));
      }
    }

    if (IMPORT_fds_.size() > 0) {
      // LOG(INFO) << "Try to restore the table " << keys_prefix_name
      //           << " to Redis service from "
      //           << folder_dir + keys_prefix_name_slices_import_[0] +
      //                  ".rdb and its companions";

      TFRA_REQUIRES_OK(table_instance_->RestoreFromDisk(keys_prefix_name_slices_,
                                               IMPORT_content, IMPORT_fds_,
                                               IMPORT_fds_sizes));
      for (auto &fd : IMPORT_fds_) close(fd);
    }

    return TFRA_Status::OK();
  }

  TFRA_Status ExportValues(K* const keys, V* const values) override {
    if (redis_connection_params_.table_store_mode == 0) {
      return ExportValuesToTensor(keys, values);
    } else if (redis_connection_params_.table_store_mode == 1) {
      return ExportValuesToFiles();
    } else if (redis_connection_params_.table_store_mode == 2) {
      // Tensor *keys;
      // TF_RETURN_IF_ERROR(ctx->allocate_output("keys", TensorShape({1}), &keys));
      // Tensor *values;
      // TF_RETURN_IF_ERROR(ctx->allocate_output(
      //     "values", TensorShape({1, runtime_value_dim_}), &values));
      return TFRA_Status::OK();
    }
    return TFRA_Status(StatusCode::INVALID_ARGUMENT,
                  "invalid redis_connection_params_.table_store_mode.");
  }

  TFRA_Status ExportValuesToFiles() {
    std::string file_path, folder_dir;
    const unsigned &storage_slice = redis_connection_params_.storage_slice;
    int tem_fd;

    EXPORT_content.resize(storage_slice);
    EXPORT_fds_.clear();
    EXPORT_fds_.reserve(storage_slice);

    folder_dir = check_dir(redis_connection_params_.model_lib_abs_dir);
    folder_dir = check_dir(folder_dir + redis_connection_params_.model_tag_runtime);

    for (unsigned i = 0; i < storage_slice && i < keys_prefix_name_slices_.size(); ++i) {
      file_path = folder_dir + keys_prefix_name_slices_[i] + ".rdb";
      if (access(file_path.c_str(), 0) != -1) {
        // LOG(WARNING) << "File " + file_path + " has already existed!";
        time_t totalseconds = time(NULL);
        struct tm *st = localtime(&totalseconds);
        char tmp_time_str[20];
        sprintf(tmp_time_str, "%04d-%02d-%02d-%02d:%02d:%02d",
                (st->tm_year + 1900) % 10000u, (st->tm_mon + 1) % 100u,
                (st->tm_mday) % 100u, (st->tm_hour) % 100u, (st->tm_min) % 100u,
                (st->tm_sec) % 100u);
        std::string new_file_path = file_path + "." + tmp_time_str;
        // LOG(WARNING) << "Rename the file " + file_path + " into " +
        //                     new_file_path + " with local time!";
        rename(file_path.c_str(), new_file_path.c_str());
        tem_fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0777);
        if (tem_fd > 0) {
          EXPORT_fds_.push_back(tem_fd);
        } else {
          // LOG(ERROR) << "Can not create the file " << file_path
          //            << " for instead. Something bad happens";
        }
      } else {
        tem_fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0777);
        EXPORT_fds_.push_back(tem_fd);
      }
    }

    if (EXPORT_fds_.size() > 0) {
      // LOG(INFO) << "Try to dump the table " << keys_prefix_name
      //           << " from Redis service to "
      //           << folder_dir + keys_prefix_name + "[*].rdb";

      TFRA_REQUIRES_OK(table_instance_->DumpToDisk(keys_prefix_name_slices_,
                                          EXPORT_content, EXPORT_fds_));
      // for (auto &fd : EXPORT_fds_) // for now the writting may be not
      // finished
      //   close(fd);
    }

    // TODO chenjinglin
    // Tensor *keys;
    // TF_RETURN_IF_ERROR(ctx->allocate_output("keys", TensorShape({1}), &keys));

    // Tensor *values;
    // TF_RETURN_IF_ERROR(ctx->allocate_output(
    //     "values", TensorShape({1, runtime_value_dim_}), &values));

    return TFRA_Status::OK();
  }

  TFRA_Status ExportValuesToTensor(K* const keys, V* const values) {
    int64_t total_size = 0;
    long long cursor = 0;
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> hscan_reply;
    const redisReply *kvs_reply;

    for (size_t i = 0; i < keys_prefix_name_slices_.size(); ++i) {
      total_size +=
          table_instance_->TableSizeInBucket(keys_prefix_name_slices_[i]);
    }

    // Tensor *keys;
    // TF_RETURN_IF_ERROR(
    //     ctx->allocate_output("keys", TensorShape({total_size}), &keys));

    // Tensor *values;
    // TF_RETURN_IF_ERROR(ctx->allocate_output(
    //     "values", TensorShape({total_size, runtime_value_dim_}), &values));

    if (total_size == 0) {
      // LOG(WARNING) << "There is no embedding table called " << keys_prefix_name
      //              << " existing in the Redis service. "
      //              << "Exporting values to Tensor failed.";
      return TFRA_Status::OK();
    }

    redisReply const *temp_reply;
    // const K *pk_raw = reinterpret_cast<const K *>(keys->tensor_data().data());
    // const V *pv_raw = reinterpret_cast<const V *>(values->tensor_data().data());
    const K *pk_raw = keys;
    const V *pv_raw = values;
    for (size_t i = 0; i < keys_prefix_name_slices_.size(); ++i) {
      cursor = 0;
      while (true) {
        hscan_reply.reset();
        hscan_reply = table_instance_->HscanGetKeysValsInBucket(
            keys_prefix_name_slices_[i], &cursor, multi_redis_cmd_max_argc);
        if (hscan_reply == nullptr) {
          // return errors::Unknown(
          //     "Unknown errors happen when HscanGetKeysValsInBucket in "
          //     "ExportValuesToTensor");
          return TFRA_Status(StatusCode::UNKNOWN,
                             "Unknown errors happen when HscanGetKeysValsInBucket in ExportValuesToTensor");
        }
        kvs_reply = hscan_reply->element[1];
        // fill Tensor keys and values
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

        // LOG(INFO) << "The cursor of scanning " << keys_prefix_name_slices_[i]
        //           << " in ExportValuesToTensor is " << cursor << " now.";
        if (cursor == 0) {
          break;
        }
      }
    }

    return TFRA_Status::OK();
  }

  TFRA_Status Dump(K* const key_buffer, V* const value_buffer, size_t search_offset, 
                   size_t buffer_size, size_t* dumped_counter) override {
    // int64_t total_size = 0;
    *dumped_counter = 0;
    K* pk_raw = key_buffer;
    V* pv_raw = value_buffer;
  
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> hscan_reply;
    const redisReply *kvs_reply;
    
    if (dump_index_ >= keys_prefix_name_slices_.size()) {
      dump_index_ = 0;
    }

    hscan_reply.reset();
    hscan_reply = table_instance_->HscanGetKeysValsInBucket(
            keys_prefix_name_slices_[dump_index_], &dump_cursor_, buffer_size);
    if (hscan_reply == nullptr) {
      return TFRA_Status(StatusCode::UNKNOWN,
              "Unknown errors happen when HscanGetKeysValsInBucket");
    }
    kvs_reply = hscan_reply->element[1];
    // fill Tensor keys and values
    for (size_t j = 0; j < kvs_reply->elements; ++j) {
      // read key
      redisReply const * temp_reply = kvs_reply->element[j];
      if (temp_reply->type == REDIS_REPLY_STRING) {
        ReplyMemcpyToKeyTensor<K>(pk_raw, temp_reply->str,temp_reply->len);
      }
      ++pk_raw;

      // read value
      ++j;
      temp_reply = kvs_reply->element[j];
      if (temp_reply->type == REDIS_REPLY_STRING) {
        ReplyMemcpyToValTensor<V>(pv_raw, temp_reply->str, runtime_value_dim_);
      }
      pv_raw += runtime_value_dim_;
      *dumped_counter += 1;
    }

    if (0 == dump_cursor_) {
      dump_index_ ++;
    }

    return TFRA_Status::OK();
  }

#if 0
  TFRA_Status LoadFromFileSystemImpl(FileSystem *fs,
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
      TF_RETURN_IF_ERROR(DoInsert(false, (K *)key_buffer.data(),
                                  (V *)value_buffer.data(), nkeys,
                                  runtime_value_dim_));
      key_file_offset += key_read_byte;
      remainder = key_file_size - key_file_offset;
    }

    LOG(INFO) << "Finish loading " << key_size << " keys and values from "
              << key_filepath << " and " << value_filepath << " in total.";

    return TFRA_Status::OK();
  }

  TFRA_Status LoadFromFileSystem(const string &dirpath,
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
        TF_RETURN_IF_ERROR(LoadFromFileSystemImpl(fs, fp, buffer_size));
      }
    } else {
      return LoadFromFileSystemImpl(fs, filepath, buffer_size);
    }
    return TFRA_Status::OK();
  }
#endif

  TFRA_DataType key_dtype() const override { return TFRA_DataTypeToEnum<K>::v(); }

  TFRA_DataType value_dtype() const override { return TFRA_DataTypeToEnum<V>::v(); }

  std::vector<int> key_shape() const final { return key_shape_; }

  std::vector<int> value_shape() const override { return value_shape_; }

  int64_t MemoryUsed() const override {
    // LOG(INFO) << __FILE__ << ":" << __func__ << ":" << __LINE__ << std::endl;

    int64_t ret = 0;
    ret = (int64_t)(size() * (sizeof(K) + sizeof(V)));
    return sizeof(RedisLookupTable) + ret;
  }
};

#define REGISTER_REDIS_KERNEL(key_dtype, value_dtype)                                             \
    REGISTER_LOOKUP_TABLE("STAND_REDIS", "CPU", key_dtype, value_dtype,                           \
      RedisLookupTable<key_dtype, value_dtype>)

REGISTER_REDIS_KERNEL(int32, double);
REGISTER_REDIS_KERNEL(int32, float);
REGISTER_REDIS_KERNEL(int32, int32);
REGISTER_REDIS_KERNEL(int64_t, double);
REGISTER_REDIS_KERNEL(int64_t, float);
REGISTER_REDIS_KERNEL(int64_t, int32);
REGISTER_REDIS_KERNEL(int64_t, int64_t);
// REGISTER_REDIS_KERNEL(int64_t, std::string_view);
REGISTER_REDIS_KERNEL(int64_t, int8);
// REGISTER_REDIS_KERNEL(std::string_view, bool);
// REGISTER_REDIS_KERNEL(std::string_view, double);
// REGISTER_REDIS_KERNEL(std::string_view, float);
// REGISTER_REDIS_KERNEL(std::string_view, int32);
// REGISTER_REDIS_KERNEL(std::string_view, int64_t);
// REGISTER_REDIS_KERNEL(std::string_view, int8);


// REGISTER_REDIS_KERNEL(int64_t, tstring);
REGISTER_REDIS_KERNEL(int64_t, Eigen::half);
// REGISTER_REDIS_KERNEL(tstring, bool);
// REGISTER_REDIS_KERNEL(tstring, double);
// REGISTER_REDIS_KERNEL(tstring, float);
// REGISTER_REDIS_KERNEL(tstring, int32);
// REGISTER_REDIS_KERNEL(tstring, int64_t);
// REGISTER_REDIS_KERNEL(tstring, int8);
// REGISTER_REDIS_KERNEL(tstring, Eigen::half);

#undef REGISTER_REDIS_KERNEL

}  // namespace redis_table
}  // namespace recommenders_addons
}  // namespace tensorflow
