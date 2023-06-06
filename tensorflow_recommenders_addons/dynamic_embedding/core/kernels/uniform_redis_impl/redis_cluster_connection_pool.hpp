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

#include <inttypes.h>
#include <sw/redis++/connection.h>
#include <sw/redis++/connection_pool.h>
#include <sw/redis++/redis++.h>
#include <sw/redis++/sentinel.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <algorithm>

#include "redis_connection_util.hpp"
#include "thread_pool.h"

using sw::redis::ConnectionOptions;
using sw::redis::ConnectionPoolOptions;
using sw::redis::Redis;
using sw::redis::RedisCluster;
using sw::redis::Role;

namespace tensorflow {
namespace recommenders_addons {
namespace redis_connection {
template <typename RedisInstance, typename K, typename V>
class RedisWrapper<RedisInstance, K, V,
                   typename std::enable_if<
                       std::is_same<RedisInstance, RedisCluster>::value>::type>
    : public RedisBaseWrapper<K, V> {
 private:
  ConnectionOptions conn_opts_;
  ConnectionPoolOptions pool_opts_;
  ThreadPool *network_worker_pool_ptr_;
  std::exception_ptr error_ptr_;

 public:
  // for the hungry singleton mode
  std::shared_ptr<RedisInstance> redis_conn_read_ = nullptr;
  std::shared_ptr<RedisInstance> redis_conn_write_ = nullptr;

 public:
  RedisWrapper(RedisInstance &&) = delete;
  RedisWrapper(const RedisInstance &) = delete;
  RedisWrapper &operator=(const RedisInstance &) = delete;

  ~RedisWrapper() {
    if (network_worker_pool_ptr_) {
      delete network_worker_pool_ptr_;
    }
    if (redis_conn_read_ == nullptr && redis_conn_write_ == nullptr) {
      return;
    }
    redis_conn_read_.reset();
    redis_conn_write_.reset();
    // LOG(INFO) << "RedisCluster connection pool destructor called successfully.";
  }

 private:
  RedisWrapper()  // In singleton mode, classes should not be initialized
                  // through constructor
  {
    network_worker_pool_ptr_ = new ThreadPool(hardware_concurrency_);
    // LOG(INFO)
    //     << "RedisCluster connection pool constructor called successfully.";
  }

 public:
  std::shared_ptr<RedisInstance> StartConn(size_t ip_port_count, Role role) {
    conn_opts_.host = this->redis_connection_params_.redis_host_ip[ip_port_count];
    conn_opts_.port =
        this->redis_connection_params_.redis_host_port[ip_port_count];

    this->SetPublicConnParams(conn_opts_, pool_opts_,
                              this->redis_connection_params_);

    try {
      auto redis_client = std::make_shared<RedisInstance>(
          RedisInstance(conn_opts_, pool_opts_, role));
      redis_client->set("key test for connecting", "val test for connecting",
                        std::chrono::milliseconds(1));
      if (this->RedisClusterEnabled(redis_client) == false) {
        // LOG(ERROR)
        //     << "Now is cluster mode but try to connect Redis single node. "
        //        "Please check redis_connection_mode in config file.";
        throw std::invalid_argument(
            "Can not connect to single node when in cluster mode, "
            "redis_connection_mode should be 1 when connect to single node.");
      }
      return redis_client;
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler--error: " << err.what();
      return nullptr;
    } catch (...) {
      // LOG(ERROR) << "RedisHandler--other crash";
      return nullptr;
    }
    return nullptr;
  }

  TFRA_Status Conn() override {
    assert(this->redis_connection_params_.redis_host_ip.size() ==
           this->redis_connection_params_.redis_host_port.size());
    auto role_read = Role::MASTER;
    if (this->redis_connection_params_.redis_read_access_slave) {
      role_read = Role::SLAVE;
    }
    if (this->isRedisConnect == false) {
      for (size_t i = 0; i < this->redis_connection_params_.redis_host_ip.size();
           ++i) {
        for (short j = 0; j < 10; j++) {
          if (redis_conn_read_ == nullptr) {
            redis_conn_read_ = StartConn(i, role_read);
          }
          if (redis_conn_write_ == nullptr) {
            redis_conn_write_ = StartConn(i, Role::MASTER);
          }
          if (redis_conn_read_ != nullptr && redis_conn_write_ != nullptr) {
            this->isRedisConnect = true;
            return TFRA_Status::OK();
          }
        }
        // LOG(WARNING) << "Can not access the host "
        //              << this->redis_connection_params_.redis_host_ip[i]
        //              << ". Delete it from the host list.";
        this->redis_connection_params_.redis_host_ip.erase(
            this->redis_connection_params_.redis_host_ip.begin() + i);
        this->redis_connection_params_.redis_host_port.erase(
            this->redis_connection_params_.redis_host_port.begin() + i);
      }
      if (this->isRedisConnect == false) {
        // LOG(ERROR) << "Can not connect to the Redis Cluster servers.";
        if (redis_conn_read_ == nullptr && redis_conn_write_ != nullptr) {
          return TFRA_Status(StatusCode::UNAVAILABLE,
                        "Can not access Redis Slave servers, Exit without any Redis connection.");
        }
        return TFRA_Status(StatusCode::UNAVAILABLE, "Exit without any Redis connection.");
      }
    }
    return TFRA_Status::OK();
  }

  static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> get_instance() {
    std::shared_ptr<RedisWrapper<RedisInstance, K, V>> instance_ptr(
        new RedisWrapper<RedisInstance, K, V>());
    return instance_ptr;
  }

 private:
  template <typename Cmd>
  std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> PipeExecRead(
      Cmd cmd, const unsigned &size_check,
      const std::unique_ptr<BucketContext> &bucket_context) {
    if (bucket_context->ptrs->size() >= size_check) {
      ::sw::redis::StringView hkey((*bucket_context->ptrs)[1],
                                   (*bucket_context->sizes)[1]);
      try {
        return redis_conn_read_->command(cmd, hkey, bucket_context->ptrs.get(),
                                        bucket_context->sizes.get());
      } catch (const std::exception &err) {
        // LOG(ERROR) << "RedisHandler error in PipeExecRead for slices "
        //            << hkey.data() << " -- " << err.what();
      }
    } else {
      return nullptr;
    }
    return nullptr;
  }

  template <typename Cmd>
  std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> PipeExecWrite(
      Cmd cmd, const unsigned &size_check,
      const std::unique_ptr<BucketContext> &bucket_context) {
    if (bucket_context->ptrs->size() >= size_check) {
      ::sw::redis::StringView hkey((*bucket_context->ptrs)[1],
                                   (*bucket_context->sizes)[1]);
      try {
        return redis_conn_write_->command(cmd, hkey, bucket_context->ptrs.get(),
                                         bucket_context->sizes.get());
      } catch (const std::exception &err) {
        // LOG(ERROR) << "RedisHandler error in PipeExecWrite for slices "
        //            << hkey.data() << " -- " << err.what();
      }
    } else {
      return nullptr;
    }
    return nullptr;
  }

 public:
  std::vector<std::string> GetKeyBucketsAndOptimizerParamsWithName(
      const std::string &keys_prefix_name,
      const bool only_get_buckets) override {
    std::vector<std::string> keys_prefix_name_slices_in_redis;
    std::string redis_command;
    // get cluster info
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey) {
      connection.send("CLUSTER SLOTS");
    };
    ::sw::redis::StringView _hkey("0");
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn_read_->command(cmd, _hkey);
    } catch (const std::exception &err) {
      // LOG(ERROR)
      //     << "RedisHandler error in "
      //        "GetKeyBucketsAndOptimizerParamsWithName(CLUSTER SLOTS) --  "
      //     << err.what();
      return keys_prefix_name_slices_in_redis;
    }

    std::vector<std::pair<std::string, long long>> ip_port_set;
    size_t servers_num = reply->elements;
    ip_port_set.reserve(servers_num);
    for (size_t i = 0; i < servers_num; ++i) {
      ip_port_set.emplace_back(std::make_pair<std::string, long long>(
          std::string(reply->element[i]->element[2]->element[0]->str,
                      reply->element[i]->element[2]->element[0]->len),
          std::move(reply->element[i]->element[2]->element[1]->integer)));
    }
    std::sort(ip_port_set.begin(), ip_port_set.end());
    ip_port_set.erase(std::unique(ip_port_set.begin(), ip_port_set.end()),
                      ip_port_set.end());

    std::unique_ptr<Redis> redis_client;
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply_server;
    ConnectionOptions connection_options;
    long long cursor = 0;
    const redisReply *set_reply;
    keys_prefix_name_slices_in_redis.reserve(
        this->redis_connection_params_.storage_slice);
    for (size_t i = 0; i < ip_port_set.size(); ++i) {
      connection_options.host = ip_port_set[i].first;  // Required.
      connection_options.port =
          ip_port_set[i].second;  // Optional. The default port is 6379.
      connection_options.user = this->redis_connection_params_.redis_user;
      connection_options.password =
          this->redis_connection_params_
              .redis_password;  // Optional. No redis_password by default.
      connection_options.db =
          this->redis_connection_params_
              .redis_db;  // Optional. Use the 0th database by default.
      redis_client.reset(new Redis(connection_options));
      auto cmd_per_server = [](::sw::redis::Connection &connection,
                               const char *str) { connection.send(str); };
      reply_server.reset();
      cursor = 0;
      while (true) {
        if (only_get_buckets) {
          redis_command = "SCAN " + std::to_string(cursor) + " MATCH " +
                          keys_prefix_name + "{[0123456789]*}";
        } else {
          redis_command = "SCAN " + std::to_string(cursor) + " MATCH " +
                          keys_prefix_name + "*{[0123456789]*}";
        }
        try {
          reply_server =
              redis_client->command(cmd_per_server, redis_command.data());
        } catch (const std::exception &err) {
          // LOG(ERROR)
          //     << "RedisHandler error "
          //        "GetKeyBucketsAndOptimizerParamsWithName(SCAN) in for IP "
          //     << ip_port_set[i].first << " --  " << err.what();
        }
        if (reply_server->element[0]->type == REDIS_REPLY_STRING) {
          // #define REDIS_REPLY_STRING 1
          cursor = std::atoll(reply_server->element[0]->str);
        }
        if (reply_server->element[1]->type == REDIS_REPLY_ARRAY) {
          set_reply = reply_server->element[1];
          for (size_t i = 0; i < set_reply->elements; ++i) {
            keys_prefix_name_slices_in_redis.emplace_back(std::string(
                set_reply->element[i]->str, set_reply->element[i]->len));
          }
        }
        if (cursor == 0) {
          break;
        }
      }
    }
    return keys_prefix_name_slices_in_redis;
  }

  /*
    If the number of slices in the Redis service is the same as the number set
    by the user, then 1 is returned. If the former is smaller, return 2.
    If there is no corresponding table in the Redis service, 0 is returned.
    Other exceptions return -1.
  */
  int CheckSlicesNum(const std::string &keys_prefix_name) override {
    std::vector<std::string> keys_prefix_name_slices_in_redis;
    try {
      keys_prefix_name_slices_in_redis = std::move(
          GetKeyBucketsAndOptimizerParamsWithName(keys_prefix_name, true));
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler error in CheckSlicesNum for SCAN "
      //            << keys_prefix_name << " -- " << err.what();
      return -1;
    }

    if (keys_prefix_name_slices_in_redis.size() == 0) {
      // LOG(INFO) << "There is not a corresponding table " << keys_prefix_name
      //           << " existing in Redis cluster servers";
      return 0;
    } else if (keys_prefix_name_slices_in_redis.size() ==
               this->redis_connection_params_.storage_slice) {
      // LOG(INFO) << "There is already a corresponding table " << keys_prefix_name
      //           << " existing in Redis cluster servers";
      return 1;
    } else if (keys_prefix_name_slices_in_redis.size() <=
               this->redis_connection_params_.storage_slice) {
      // LOG(WARNING) << "storage_slice in this->redis_connection_params_ which is "
      //              << this->redis_connection_params_.storage_slice
      //              << " is bigger than the slices number of this "
      //              << keys_prefix_name
      //              << " in the Redis Cluster servers which is "
      //              << keys_prefix_name_slices_in_redis.size();
      return 2;
    } else {
      // LOG(ERROR) << "storage_slice in this->redis_connection_params_ which is "
      //            << this->redis_connection_params_.storage_slice
      //            << " did not equal to the slices number of this "
      //            << keys_prefix_name
      //            << " in the Redis Cluster servers which is "
      //            << keys_prefix_name_slices_in_redis.size();
      return -1;
    }
    return -1;
  }

  std::vector<std::pair<unsigned, unsigned>> ClusterNodesSlots(
      bool full_slots) override {
    std::vector<std::pair<unsigned, unsigned>> cluster_slots;
    cluster_slots.reserve(this->redis_connection_params_.storage_slice);
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey) {
      connection.send("CLUSTER NODES");
    };
    ::sw::redis::StringView _hkey("0");
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn_read_->command(cmd, _hkey);
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler error in ClusterNodesSlots --  "
      //            << err.what();
      return cluster_slots;
    }
    if (reply->type == REDIS_REPLY_STRING) {
      std::vector<std::vector<::sw::redis::StringView>> csv_table;
      std::vector<::sw::redis::StringView> csv_table_row;
      csv_table.reserve(this->redis_connection_params_.storage_slice * 2);
      csv_table_row.reserve(10);
      const char *str_ptr = reply->str;
      const char *const str_ptr_begin = reply->str;
      for (size_t i = 0, col = 0; i < reply->len; ++i) {
        if (*(str_ptr_begin + i) == ' ') {
          csv_table_row.emplace_back(::sw::redis::StringView(str_ptr, i - col));
          col = i + 1;
          str_ptr = str_ptr_begin + i + 1;
        } else if (*(str_ptr_begin + i) == '\n') {
          csv_table_row.emplace_back(::sw::redis::StringView(str_ptr, i - col));
          csv_table.push_back(csv_table_row);
          csv_table_row.clear();
          col = i + 1;
          str_ptr = str_ptr_begin + i + 1;
        }
      }
      std::string tmp_slot_num;
      tmp_slot_num.reserve(5);
      unsigned tmp_first_slot = 0, tmp_second_slot = 0;
      for (auto row : csv_table) {
        if (strncmp(row.at(2).data(), "master", 6) == 0 ||
            strncmp(row.at(2).data(), "myself,master", 13) == 0) {
          if (full_slots) {
            for (size_t i = 8; i < row.size(); ++i) {
              for (const char *num = row.at(i).data();
                   num != num + row.at(i).size(); ++num) {
                if (*num != '-') {
                  tmp_slot_num.push_back(*num);
                } else {
                  tmp_first_slot = std::stoul(tmp_slot_num);
                  tmp_slot_num.clear();
                }
              }
              tmp_second_slot = std::stoul(tmp_slot_num);
              cluster_slots.push_back(
                  std::make_pair(tmp_first_slot, tmp_second_slot));
            }
          } else {
            const char *const num_end = row.at(8).data() + row.at(8).size();
            for (const char *num = row.at(8).data(); num != num_end; ++num) {
              if (*num != '-') {
                tmp_slot_num.push_back(*num);
              } else {
                tmp_first_slot = std::stoul(tmp_slot_num);
                tmp_slot_num.clear();
              }
            }
            tmp_second_slot = std::stoul(tmp_slot_num);
            tmp_slot_num.clear();
            cluster_slots.push_back(
                std::make_pair(tmp_first_slot, tmp_second_slot));
          }
        }
      }
    }
    std::sort(cluster_slots.begin(), cluster_slots.end());
    cluster_slots.erase(std::unique(cluster_slots.begin(), cluster_slots.end()),
                        cluster_slots.end());
    return cluster_slots;
  }

  size_t TableSizeInBucket(
      const std::string &keys_prefix_name_slice) override {
    const std::string command_string = "HLEN " + keys_prefix_name_slice;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn_read_->command(cmd, keys_prefix_name_slice,
                                       command_string.data());
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler error in TableSizeInBucket for slices "
      //            << keys_prefix_name_slice << " -- " << err.what();
      throw(err);
    }
    size_t size = 0;
    if (reply->type == REDIS_REPLY_INTEGER)  // #define REDIS_REPLY_STRING 1
    {
      size += reply->integer;  // decimal
    }
    return size;
  }

  TFRA_Status RemoveHkeysInBuckets(
      const std::string &keys_prefix_name_slice) override {
    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    std::string redis_command("DEL ");
    std::string command_string = redis_command + keys_prefix_name_slice;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    try {
      /*reply=*/redis_conn_write_->command(cmd, keys_prefix_name_slice,
                                          command_string.data());
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler error in RemoveHkeysInBuckets for slices "
      //            << keys_prefix_name_slice << " -- " << err.what();
      return TFRA_Status(StatusCode::UNKNOWN, err.what());
    }

    return TFRA_Status::OK();
  }

  std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>
  HscanGetKeysValsInBucket(const std::string &keys_prefix_name_slice,
                           long long *cursor, const long long count) override {
    std::string command_string = "HSCAN " + keys_prefix_name_slice + ' ' +
                                 std::to_string(*cursor) + " COUNT " +
                                 std::to_string(count);
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn_read_->command(cmd, keys_prefix_name_slice,
                                       command_string.data());
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler error in HscanGetKeysValsInBucket for slices "
      //            << keys_prefix_name_slice << " -- " << err.what();
    }
    if (reply->element[0]->type == REDIS_REPLY_STRING) {
      // #define REDIS_REPLY_STRING 1
      *cursor = std::atoll(reply->element[0]->str);
    } else {
      return nullptr;
    }
    return reply;
  }

  std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> MgetInBucket(
      const K *keys, const int64_t begin, const int64_t max_i,
      const std::string &keys_prefix_name_slice) override {
    std::unique_ptr<BucketContext> bucket_context_temp(new BucketContext());
    const static char *redis_command = "HMGET";
    const static std::size_t redis_command_byte = 5;
    const int &&total = max_i - begin;
    const int &&argc = total + 2;
    bucket_context_temp->HandleClear();
    bucket_context_temp->HandleReserve(argc);

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    bucket_context_temp->HandlePushBack(redis_command, redis_command_byte);
    bucket_context_temp->HandlePushBack(keys_prefix_name_slice.data(),
                                        keys_prefix_name_slice.size());

    for (; pk_raw != pk_raw_end; ++pk_raw) {
      // Direct access to Tensor data in TensorFlow
      bucket_context_temp->HandlePushBack(KContentPointer<K>(pk_raw),
                                          KTypeSize<K>(pk_raw));
    }

    auto cmd = [](::sw::redis::Connection &connection,
                  const ::sw::redis::StringView hkey,
                  const std::vector<const char *> *ptrs_i,
                  const std::vector<std::size_t> *sizes_i) {
      assert(strcmp(ptrs_i->front(), "HMGET") == 0);
      assert(sizes_i->front() == 5);
      assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);

      connection.send(static_cast<int>(ptrs_i->size()),
                      const_cast<const char **>(ptrs_i->data()),
                      sizes_i->data());
    };

    try {
      return PipeExecRead(cmd, 3U, bucket_context_temp);
    } catch (const std::exception &err) {
      return nullptr;
    }
  }

  TFRA_Status SetExpireBuckets(
      const std::string &keys_prefix_name) override {
    if (this->redis_connection_params_.expire_model_tag_in_seconds >= 0) {
      // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
      const std::string expire_command("EXPIRE ");
      std::string redis_command;
      auto cmd = [](::sw::redis::Connection &connection,
                    ::sw::redis::StringView hkey,
                    const char *str) { connection.send(str); };
      auto &&bucket_names =
          GetKeyBucketsAndOptimizerParamsWithName(keys_prefix_name, false);
      for (auto bucket_name : bucket_names) {
        redis_command.clear();
        redis_command =
            expire_command + bucket_name + ' ' +
            std::to_string(
                this->redis_connection_params_.expire_model_tag_in_seconds);
        try {
          /*reply=*/redis_conn_write_->command(cmd, bucket_name,
                                              redis_command.data());
        } catch (const std::exception &err) {
          // LOG(ERROR) << "RedisHandler error in SetExpireBuckets for "
          //            << bucket_name << " -- " << err.what();
          // return errors::Unknown(err.what());
          return TFRA_Status(StatusCode::UNKNOWN, err.what());
        }
      }
    }
    return TFRA_Status::OK();
  }

  TFRA_Status SetPersistBuckets(const std::string &keys_prefix_name) override {
    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    const std::string expire_command("PERSIST ");
    std::string redis_command;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    auto &&bucket_names =
        GetKeyBucketsAndOptimizerParamsWithName(keys_prefix_name, false);
    for (auto bucket_name : bucket_names) {
      redis_command.clear();
      redis_command = expire_command + bucket_name;
      try {
        /*reply=*/redis_conn_write_->command(cmd, bucket_name,
                                            redis_command.data());
      } catch (const std::exception &err) {
        // LOG(ERROR) << "RedisHandler error in SetPersistBuckets for "
        //            << bucket_name << " -- " << err.what();
        // return errors::Unknown(err.what());
        return TFRA_Status(StatusCode::UNKNOWN, err.what());
      }
    }

    return TFRA_Status::OK();
  }

  /*
  fds are the return of POSIX open file function declared in <fcntl.h>
  */
  TFRA_Status DumpToDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &wrs, const std::vector<int> &fds) override {
    if (fds.size() == 0) {
      return TFRA_Status::OK();
    }

    std::string redis_command;
    aiocb *wr;
    int ret;  // int fd;

    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;

    size_t buf_len;
    volatile void *tem_aio_buf;
    for (unsigned i = 0; i < this->redis_connection_params_.storage_slice; ++i) {
      redis_command = "DUMP " + keys_prefix_name_slices[i];
      reply.reset();
      try {
        reply = redis_conn_read_->command(cmd, keys_prefix_name_slices[i],
                                         redis_command.data());
      } catch (const std::exception &err) {
        // LOG(ERROR) << "RedisHandler error in DumpToDisk for slices "
        //            << keys_prefix_name_slices[i] << " -- " << err.what();
        // return errors::Unknown(err.what());
        return TFRA_Status(StatusCode::UNKNOWN, err.what());
      }

      wr = &wrs[i];
      if (wr->aio_nbytes > 0) {
        for (size_t i = 3; i > 0; --i) {
          while (aio_error(wr) == EINPROGRESS) {
            ;
          }
          if ((ret = aio_return(wr)) > 0) {
            break;
          } else {
            // LOG(WARNING) << "File handle " << wr->aio_fildes
            //              << " did not finish writing last round. "
            //              << "Try to write " << i << " more times";
            ret = aio_write(wr);
            if (ret < 0) perror("aio_write");
          }
        }
      }
      if (reply->type == REDIS_REPLY_STRING)  // #define REDIS_REPLY_STRING 1
      {
        buf_len = reply->len;
        tem_aio_buf = wr->aio_buf;
        wr->aio_buf = realloc((void *)tem_aio_buf,
                              buf_len);  // Be careful! The memory requested
                                         // here should be freed somewhere!
        memcpy((void *)(wr->aio_buf), reply->str, buf_len);
        wr->aio_nbytes = buf_len;
        wr->aio_fildes = fds[i];
        wr->aio_offset = 0;
        ret = aio_write(wr);
        if (ret < 0) perror("aio_write");
      } else {
        // LOG(ERROR) << "HKEY " << keys_prefix_name_slices[i]
        //            << " does not exist in the Redis server. ";
      }
    }

    return TFRA_Status::OK();
  }

  TFRA_Status RestoreFromDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &rds, const std::vector<int> &fds,
      const std::vector<unsigned long> &buf_sizes) override {
    if (fds.size() == 0) {
      TFRA_Status::OK();
    }

    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    const unsigned &&storage_slice = fds.size();
    aiocb *rd;
    int ret;  // int fd;

    auto cmd = [](::sw::redis::Connection &connection,
                  const ::sw::redis::StringView hkey,
                  const std::vector<const char *> &ptrs_i,
                  const std::vector<std::size_t> &sizes_i) {
      assert(strcmp(ptrs_i.front(), "RESTORE") == 0);
      assert(sizes_i.front() == 7);
      connection.send(static_cast<int>(ptrs_i.size()),
                      const_cast<const char **>(ptrs_i.data()), sizes_i.data());
    };

    size_t buf_len;
    volatile void *tem_aio_buf;

    std::vector<std::vector<const char *>> ptrs_i_i(storage_slice);
    std::vector<std::vector<std::size_t>> sizes_i_i(storage_slice);

    const static char *redis_command = "RESTORE";
    const static std::size_t &&redis_command_byte = 7;
    const static char *redis_command_param = "0";
    const static std::size_t &&redis_command_byte_param = 1;
    const static char *replace_command = "REPLACE";
    const static std::size_t &&replace_command_byte = 7;

    for (size_t i = 0; i < storage_slice; ++i) {
      rd = &rds[i];

      buf_len = buf_sizes[i];

      tem_aio_buf = rd->aio_buf;
      rd->aio_buf = realloc((void *)tem_aio_buf,
                            buf_len);  // Be careful! The memory requested here
                                       // should be freed somewhere!
      rd->aio_nbytes = buf_len;
      rd->aio_fildes = fds[i];
      rd->aio_offset = 0;
      ret = aio_read(rd);
      if (ret < 0) perror("aio_read");

      ptrs_i_i[i].reserve(5);
      ptrs_i_i[i].clear();
      ptrs_i_i[i].emplace_back(redis_command);
      ptrs_i_i[i].emplace_back(keys_prefix_name_slices[i].data());
      ptrs_i_i[i].emplace_back(redis_command_param);
      ptrs_i_i[i].emplace_back((const char *)rd->aio_buf);
      ptrs_i_i[i].emplace_back(replace_command);

      sizes_i_i[i].reserve(5);
      sizes_i_i[i].clear();
      sizes_i_i[i].emplace_back(redis_command_byte);
      sizes_i_i[i].emplace_back(keys_prefix_name_slices[i].size());
      sizes_i_i[i].emplace_back(redis_command_byte_param);
      sizes_i_i[i].emplace_back(rd->aio_nbytes);
      sizes_i_i[i].emplace_back(replace_command_byte);
    }

    int count_down = static_cast<int>(storage_slice);
    int reread_countdown[storage_slice];
    for (unsigned i = 0; i < storage_slice; ++i) {
      reread_countdown[i] = 4;
    }
    bool no_errors = true;

    while (count_down > 0) {
      for (size_t i = 0; i < storage_slice; ++i) {
        rd = &rds[i];

        if (reread_countdown[i] > 1) {
          if (rd->aio_nbytes > 0) {
            if (aio_error(rd) != EINPROGRESS) {
              if ((ret = aio_return(rd)) > 0) {
                try {
                  /*reply = */ redis_conn_write_->command(
                      cmd, keys_prefix_name_slices[i], ptrs_i_i[i],
                      sizes_i_i[i]);
                } catch (const std::exception &err) {
                  // LOG(ERROR)
                  //     << "RedisHandler error in RestoreFromDisk for slices "
                  //     << keys_prefix_name_slices[i] << " -- " << err.what();
                  if (rd->aio_buf) {
                    free((void *)rd->aio_buf);
                    rd->aio_buf = nullptr;
                    rd->aio_nbytes = 0;
                  }
                  // return errors::Unknown(err.what());
                  return TFRA_Status(StatusCode::UNKNOWN, err.what());
                }
                if (rd->aio_buf) {
                  free((void *)rd->aio_buf);
                  rd->aio_buf = nullptr;
                  rd->aio_nbytes = 0;
                }
                --count_down;
              } else {
                // LOG(WARNING) << "File handle " << rd->aio_fildes
                //              << " did not finish reading last round. "
                //              << "Try to read " << reread_countdown[i] - 1
                //              << " more times";
                ret = aio_read(rd);
                if (ret < 0) perror("aio_read");
                --reread_countdown[i];
              }
            }
          } else {
            // LOG(WARNING) << "File handle " << rd->aio_fildes << " for slice "
            //              << keys_prefix_name_slices[i]
            //              << " has nbytes 0. Ignore.";
            reread_countdown[i] = 0;
            --count_down;
            if (rd->aio_buf) {
              free((void *)rd->aio_buf);
              rd->aio_buf = nullptr;
              rd->aio_nbytes = 0;
            }
          }
        } else if (reread_countdown[i] == 1) {
          // LOG(ERROR) << "File handle " << rd->aio_fildes << " for slice "
          //            << keys_prefix_name_slices[i]
          //            << " has some troubles! Given up.";
          --reread_countdown[i];
          --count_down;
          if (rd->aio_buf) {
            free((void *)rd->aio_buf);
            rd->aio_buf = nullptr;
            rd->aio_nbytes = 0;
          }
          no_errors = false;
        }
      }
    }

    return no_errors
               ? TFRA_Status::OK()
               : TFRA_Status(StatusCode::UNKNOWN, "Unknown errors happen in file handles.");
  }

  void DoDuplicateInRedis(const std::string &keys_prefix_name_slice_old,
                          const std::string &keys_prefix_name_slice_new) {
    const std::string redis_dump_command = "DUMP " + keys_prefix_name_slice_old;

    auto cmd_dump = [](::sw::redis::Connection &connection,
                       ::sw::redis::StringView hkey,
                       const char *str) { connection.send(str); };

    auto cmd_restore = [](::sw::redis::Connection &connection,
                          const ::sw::redis::StringView hkey,
                          const std::vector<const char *> &ptrs_i,
                          const std::vector<std::size_t> &sizes_i) {
      assert(strcmp(ptrs_i.front(), "RESTORE") == 0);
      assert(sizes_i.front() == 7);
      connection.send(static_cast<int>(ptrs_i.size()),
                      const_cast<const char **>(ptrs_i.data()), sizes_i.data());
    };

    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    std::vector<const char *> ptrs_0;
    std::vector<std::size_t> sizes_0;
    ptrs_0.reserve(5);
    sizes_0.reserve(5);

    const static char *redis_restore_command = "RESTORE";
    const static std::size_t &&redis_restore_command_byte = 7;
    const static char *redis_restore_command_param = "0";
    const static std::size_t &&redis_restore_command_byte_param = 1;

    // LOG(INFO) << "Now try to duplicate the KV pair from "
    //           << keys_prefix_name_slice_old << " to "
    //           << keys_prefix_name_slice_new;

    try {
      reply = redis_conn_read_->command(cmd_dump, keys_prefix_name_slice_old,
                                       redis_dump_command.data());
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler error in dump_to_reply of DoDuplicateInRedis "
      //               "for slices "
      //            << keys_prefix_name_slice_old << " -- " << err.what();
      error_ptr_ = std::current_exception();
      throw(err);
    }

    if (reply->type == REDIS_REPLY_STRING)  // #define REDIS_REPLY_STRING 1
    {
      ptrs_0.emplace_back(redis_restore_command);
      ptrs_0.emplace_back(keys_prefix_name_slice_new.data());
      ptrs_0.emplace_back(redis_restore_command_param);
      ptrs_0.emplace_back(reply->str);
      sizes_0.emplace_back(redis_restore_command_byte);
      sizes_0.emplace_back(keys_prefix_name_slice_new.size());
      sizes_0.emplace_back(redis_restore_command_byte_param);
      sizes_0.emplace_back(reply->len);
    } else {
      // LOG(ERROR) << "HKEY " << keys_prefix_name_slice_new
      //            << " does not exist in the Redis server. ";
    }
    try {
      /*reply = */ redis_conn_write_->command(
          cmd_restore, keys_prefix_name_slice_new, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      // LOG(ERROR) << "RedisHandler error in restore_from_reply of "
      //               "DoDuplicateInRedis for slices "
      //            << keys_prefix_name_slice_new << " -- " << err.what();
      error_ptr_ = std::current_exception();
      throw(err);
    }
  }

  TFRA_Status DuplicateInRedis(
      const std::vector<std::string> &keys_prefix_name_slices_old,
      const std::vector<std::string> &keys_prefix_name_slices_new) override {
    try {
      for (unsigned i = 0; i < this->redis_connection_params_.storage_slice;
           ++i) {
        network_worker_pool_ptr_->enqueue([this, &keys_prefix_name_slices_old,
                                      &keys_prefix_name_slices_new, i] {
          DoDuplicateInRedis(keys_prefix_name_slices_old[i],
                             keys_prefix_name_slices_new[i]);
        });
      }
      if (error_ptr_) {
        std::rethrow_exception(error_ptr_);
      }
    } catch (const std::exception &err) {
      error_ptr_ = nullptr;
      return TFRA_Status(StatusCode::UNKNOWN, err.what());
    }
    return TFRA_Status::OK();
  }

 public:
  /*
  The structure of ptrs and sizes which for storing Redis command char
sequence pointer and size of parameters. For example: vector<ThreadContext>
(for multi-threads, index is thread id, also vector<vector<vector<const char
*>>>)

std::vector<ThreadContext> (better to be reserved before enter MXXX_COMMAND)
-------------upper var is outside of the MXXX_COMMAND function---------------
      |
      | Thread0 has its own ThreadContext
      |
every bucket has its own BucketContext for sending data---for locating reply-
    |                                                      |
    | std::vector<BucketContext>                           | std::vector
    |                                                          <unsigned>
    |
    |
--char* point to the data and size_t indicates the length of data------------
  |                    |
  | std::vector        | std::vector
  |  <const char*>     |  <std::size_t>
  |                    |
(Real Redis command sequence because m-cmd can only be used in same hash tag)

  PS: vector bucket_locs is only allocated in Redis Cluster mode!
  */
  std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> MgetCommand(
      const K *keys, ThreadContext *thread_context, const int64_t begin,
      const int64_t max_i,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total + 2;

    const static char *redis_command = "HMGET";
    const static std::size_t &&redis_command_byte = 5;

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    const unsigned &storage_slice = this->redis_connection_params_.storage_slice;
    const unsigned &&vector_len =
        (static_cast<int64_t>(reinterpret_cast<int>(argc)) /
         this->redis_connection_params_.storage_slice) +
        2;

    thread_context->HandleReserve(storage_slice, vector_len, total);

    for (unsigned i = 0; i < storage_slice; ++i) {
      thread_context->HandlePushBack(i, redis_command, redis_command_byte);
      thread_context->HandlePushBack(i, keys_prefix_name_slices[i].data(),
                                     keys_prefix_name_slices[i].size());
    }

    unsigned *pbucket_loc = thread_context->bucket_locs->data();
    unsigned key_bucket_locs = 0;
    for (; pk_raw != pk_raw_end; ++pk_raw) {
      key_bucket_locs =
          KBucketNum<K>(this->K_bucket_num_handle_, pk_raw, storage_slice);
      // The bucket to which the key belongs is recorded to facilitate future
      // memory writes that do not recompute the redis hash
      *pbucket_loc = key_bucket_locs;
      ++pbucket_loc;

      // Direct access to Tensor data in TensorFlow
      thread_context->HandlePushBack(
          key_bucket_locs, KContentPointer<K>(pk_raw), KTypeSize<K>(pk_raw));
    }

    auto cmd = [](::sw::redis::Connection &connection,
                  const ::sw::redis::StringView hkey,
                  const std::vector<const char *> *ptrs_i,
                  const std::vector<std::size_t> *sizes_i) {
      assert(strcmp(ptrs_i->front(), "HMGET") == 0);
      assert(sizes_i->front() == 5);
      assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);

      connection.send(static_cast<int>(ptrs_i->size()),
                      const_cast<const char **>(ptrs_i->data()),
                      sizes_i->data());
    };

    std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> replies(
        storage_slice);
    std::vector<
        std::future<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>>
        results;
    for (unsigned i = 0; i < storage_slice; ++i) {
      replies[i] = std::move(nullptr);
    }
    try {
      for (unsigned i = 0; i < storage_slice; ++i) {
        results.emplace_back(
            network_worker_pool_ptr_->enqueue([this, &cmd, thread_context, i] {
              return PipeExecRead(cmd, 3U, thread_context->buckets[i]);
            }));
      }
      for (unsigned i = 0; i < storage_slice; ++i) {
        replies[i] = std::move(results[i].get());
      }
    } catch (const std::exception &err) {
      return replies;
    }
    return replies;
  }

  inline void CopyDefaultToTensor(const bool is_full_default, const V *pv_raw,
                                  const V *dft_raw,
                                  const V *const dft_raw_begin,
                                  const int64_t Velems_per_dim0) {
    if (is_full_default) {
      DefaultMemcpyToTensor<V>(
          pv_raw, dft_raw,
          Velems_per_dim0);  // Direct access to Tensor data in TensorFlow
    } else {
      DefaultMemcpyToTensor<V>(
          pv_raw, dft_raw_begin,
          Velems_per_dim0);  // Direct access to Tensor data in TensorFlow
    }
  }

  TFRA_Status MgetToTensor(V *values, const V *default_value, const bool is_full_default,
            ThreadContext *thread_context,
            std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> &reply,
            const int64_t begin, const int64_t max_i,const int64_t Velems_per_dim0) override {
    const V *pv_raw = values + begin * Velems_per_dim0;
    const V *dft_raw = default_value + begin * Velems_per_dim0;
    const V *const dft_raw_begin = default_value;

    const std::vector<unsigned> *bucket_locs =
        thread_context->bucket_locs.get();
    const unsigned &storage_slice = this->redis_connection_params_.storage_slice;
    unsigned buckets_iters_nums[storage_slice];
    unsigned bucket_loc;
    memset(buckets_iters_nums, 0U, sizeof(buckets_iters_nums));
    redisReply *temp_reply;
    bool print_once[storage_slice];
    memset(print_once, false, sizeof(print_once));
    for (auto i = 0; i < (max_i - begin);
         ++i, pv_raw += Velems_per_dim0, dft_raw += Velems_per_dim0) {
      bucket_loc = (*bucket_locs)[i];
      if (reply[bucket_loc] != nullptr) {
        if (reply[bucket_loc]->type == REDIS_REPLY_ARRAY) {
          temp_reply =
              reply[bucket_loc]->element[buckets_iters_nums[bucket_loc]];
          ++(buckets_iters_nums[bucket_loc]);
          if (temp_reply->type ==
              REDIS_REPLY_STRING)  // #define REDIS_REPLY_STRING 1
          {
            ReplyMemcpyToValTensor<V>(
                pv_raw, temp_reply->str,
                Velems_per_dim0);  // Direct access to Tensor data in TensorFlow
          } else {
            CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                                Velems_per_dim0);
          }
        }
      } else {
        if (!print_once[bucket_loc]) {
          // LOG(WARNING) << "Redis reply in bucket_loc " << bucket_loc
          //              << " from MgetCommend has some problems with error "
          //              << ", using default values to repalce.";
          print_once[bucket_loc] = true;
        }
        ++(buckets_iters_nums[bucket_loc]);
        CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                            Velems_per_dim0);
      }
    }

    return TFRA_Status::OK();
  }

  TFRA_Status MgetToTensorWithExist(
      V *values, const V *default_value, bool *exists,
      const bool is_full_default, ThreadContext *thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0) override {
    const V *pv_raw = values + begin * Velems_per_dim0;
    const V *dft_raw = default_value + begin * Velems_per_dim0;
    const V *const dft_raw_begin = default_value;

    const std::vector<unsigned> *bucket_locs =
        thread_context->bucket_locs.get();
    const unsigned &storage_slice = this->redis_connection_params_.storage_slice;
    unsigned buckets_iters_nums[storage_slice];
    unsigned bucket_loc;
    memset(buckets_iters_nums, 0U, sizeof(buckets_iters_nums));
    redisReply *temp_reply;
    bool print_once[storage_slice];
    memset(print_once, false, sizeof(print_once));
    for (int64_t i = 0, j = begin; i < (max_i - begin);
         ++i, ++j, pv_raw += Velems_per_dim0, dft_raw += Velems_per_dim0) {
      bucket_loc = (*bucket_locs)[i];
      if (reply[bucket_loc] != nullptr) {
        if (reply[bucket_loc]->type == REDIS_REPLY_ARRAY) {
          temp_reply =
              reply[bucket_loc]->element[buckets_iters_nums[bucket_loc]];
          ++(buckets_iters_nums[bucket_loc]);
          if (temp_reply->type ==
              REDIS_REPLY_STRING)  // #define REDIS_REPLY_STRING 1
          {
            ReplyMemcpyToValTensor<V>(
                pv_raw, temp_reply->str,
                Velems_per_dim0);  // Direct access to Tensor data in TensorFlow
            exists[j] = true;
          } else {
            CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                                Velems_per_dim0);
            exists[j] = false;
          }
        }
      } else {
        if (!print_once[bucket_loc]) {
          // LOG(WARNING) << "Redis reply in bucket_loc " << bucket_loc
          //              << " from MgetCommend has some problems with error "
          //              << ", using default values to repalce.";
          print_once[bucket_loc] = true;
        }
        ++(buckets_iters_nums[bucket_loc]);
        CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                            Velems_per_dim0);
        exists[j] = false;
      }
    }

    return TFRA_Status::OK();
  }

  TFRA_Status MsetCommand(
      const K *keys, const V *values, ThreadContext *thread_context,
      const int64_t begin, const int64_t max_i, const int64_t Velems_per_dim0,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total * 2 + 2;

    const static char *redis_command = "HMSET";
    const static std::size_t &&redis_command_byte = 5;

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    const std::size_t &&V_byte_size = Velems_per_dim0 * sizeof(V);

    const V *pv_raw = values + begin * Velems_per_dim0;

    const unsigned &storage_slice = this->redis_connection_params_.storage_slice;
    const unsigned &&vector_len =
        (static_cast<int64_t>(reinterpret_cast<int>(argc)) /
         this->redis_connection_params_.storage_slice) +
        2;

    thread_context->HandleReserve(storage_slice, vector_len, total);

    for (unsigned i = 0; i < storage_slice; ++i) {
      thread_context->HandlePushBack(i, redis_command, redis_command_byte);
      thread_context->HandlePushBack(i, keys_prefix_name_slices[i].data(),
                                     keys_prefix_name_slices[i].size());
    }

    VContentAndTypeSizeResult VCATS_temp;
    // std::vector<char> for storage all string in one KV pair
    std::vector<std::vector<char>> buff_temp(total);
    unsigned key_bucket_locs = 0;
    for (int i = 0; pk_raw != pk_raw_end;
         ++i, ++pk_raw, pv_raw += Velems_per_dim0) {
      VCATS_temp = VContentAndTypeSize<V>(VCATS_temp, Velems_per_dim0,
                                          V_byte_size, pv_raw, buff_temp[i]);
      key_bucket_locs =
          KBucketNum<K>(this->K_bucket_num_handle_, pk_raw,
                        storage_slice);  // TODO: change it to AVX512

      // Direct access to Tensor data in TensorFlow
      thread_context->HandlePushBack(
          key_bucket_locs, KContentPointer<K>(pk_raw), KTypeSize<K>(pk_raw));
      thread_context->HandlePushBack(
          key_bucket_locs, VCATS_temp.VContentPointer, VCATS_temp.VTypeSize);
    }

    auto cmd = [](::sw::redis::Connection &connection,
                  const ::sw::redis::StringView &hkey,
                  const std::vector<const char *> *ptrs_i,
                  const std::vector<std::size_t> *sizes_i) {
      assert(strcmp(ptrs_i->front(), "HMSET") == 0);
      assert(sizes_i->front() == 5);
      assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);

      connection.send(static_cast<int>(ptrs_i->size()),
                      const_cast<const char **>(ptrs_i->data()),
                      sizes_i->data());
    };

    std::vector<
        std::future<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>>
        results;
    try {
      for (unsigned i = 0; i < storage_slice; ++i) {
        results.emplace_back(
            network_worker_pool_ptr_->enqueue([this, &cmd, thread_context, i] {
              return PipeExecWrite(cmd, 4U, thread_context->buckets[i]);
            }));
      }
      for (auto &&result : results) {
        result.wait();
      }
      if (error_ptr_) {
        std::rethrow_exception(error_ptr_);
      }
    } catch (const std::exception &err) {
      error_ptr_ = nullptr;
      return TFRA_Status(StatusCode::UNKNOWN, err.what());
      // return errors::Unknown(err.what());
    }

    return TFRA_Status::OK();
  }

  TFRA_Status MaccumCommand(
      const K *keys, const V *values_or_delta, const bool *exists,
      ThreadContext *thread_context, const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0, std::string &values_dtype_str,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total * 2 + 4;

    const static char *redis_command = "HMACCUM";
    const static std::size_t &&redis_command_byte = 7;
    size_t dtype_str_size = values_dtype_str.size();

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    const std::size_t &&V_byte_size = Velems_per_dim0 * sizeof(V);

    const V *pv_raw = values_or_delta + begin * Velems_per_dim0;

    const unsigned &storage_slice = this->redis_connection_params_.storage_slice;
    const unsigned &&vector_len =
        (static_cast<int64_t>(reinterpret_cast<int>(argc)) /
         this->redis_connection_params_.storage_slice) +
        4;

    thread_context->HandleReserve(storage_slice, vector_len, total);

    for (unsigned i = 0; i < storage_slice; ++i) {
      thread_context->HandlePushBack(i, redis_command, redis_command_byte);
      thread_context->HandlePushBack(i, keys_prefix_name_slices[i].data(),
                                     keys_prefix_name_slices[i].size());
      thread_context->HandlePushBack(i, values_dtype_str.c_str(),
                                     dtype_str_size);
    }

    VContentAndTypeSizeResult VCATS_temp;
    // std::vector<char> for storage all string in one KV pair
    std::vector<std::vector<char>> buff_temp(total);
    unsigned key_bucket_locs = 0;
    for (int i = 0; pk_raw != pk_raw_end;
         ++i, ++pk_raw, pv_raw += Velems_per_dim0) {
      VCATS_temp = VContentAndTypeSize<V>(VCATS_temp, Velems_per_dim0,
                                          V_byte_size, pv_raw, buff_temp[i]);
      key_bucket_locs =
          KBucketNum<K>(this->K_bucket_num_handle_, pk_raw,
                        storage_slice);  // TODO: change it to AVX512

      // Direct access to Tensor data in TensorFlow
      thread_context->HandlePushBack(
          key_bucket_locs, KContentPointer<K>(pk_raw), KTypeSize<K>(pk_raw));
      thread_context->HandlePushBack(
          key_bucket_locs, VCATS_temp.VContentPointer, VCATS_temp.VTypeSize);
    }

    const bool *pe_raw = exists + begin;
    for (unsigned i = 0; i < storage_slice; ++i) {
      thread_context->HandlePushBack(i, KContentPointer<bool>(pe_raw),
                                     total * KTypeSize<bool>(pe_raw));
    }

    auto cmd = [](::sw::redis::Connection &connection,
                  const ::sw::redis::StringView &hkey,
                  const std::vector<const char *> *ptrs_i,
                  const std::vector<std::size_t> *sizes_i) {
      assert(strcmp(ptrs_i->front(), "HMACCUM") == 0);
      assert(sizes_i->front() == redis_command_byte);
      assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);

      connection.send(static_cast<int>(ptrs_i->size()),
                      const_cast<const char **>(ptrs_i->data()),
                      sizes_i->data());
    };

    std::vector<
        std::future<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>>
        results;
    try {
      for (unsigned i = 0; i < storage_slice; ++i) {
        results.emplace_back(
            network_worker_pool_ptr_->enqueue([this, &cmd, thread_context, i] {
              return PipeExecWrite(cmd, 6U, thread_context->buckets[i]);
            }));
      }
      for (auto &&result : results) {
        result.wait();
      }
      if (error_ptr_) {
        std::rethrow_exception(error_ptr_);
      }
    } catch (const std::exception &err) {
      error_ptr_ = nullptr;
      // return errors::Unknown(err.what());
      return TFRA_Status(StatusCode::UNKNOWN, err.what());
    }

    return TFRA_Status::OK();
  }

  TFRA_Status DelCommand(
      const K *keys, ThreadContext *thread_context, const int64_t begin,
      const int64_t max_i,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total + 2;

    const static char *redis_command = "HDEL";
    const static std::size_t &&redis_command_byte = 4;

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    const unsigned &storage_slice = this->redis_connection_params_.storage_slice;
    const unsigned &&vector_len =
        (static_cast<int64_t>(reinterpret_cast<int>(argc)) /
         this->redis_connection_params_.storage_slice) +
        2;

    thread_context->HandleReserve(storage_slice, vector_len, total);

    for (unsigned i = 0; i < storage_slice; ++i) {
      thread_context->HandlePushBack(i, redis_command, redis_command_byte);
      thread_context->HandlePushBack(i, keys_prefix_name_slices[i].data(),
                                     keys_prefix_name_slices[i].size());
    }

    unsigned *pbucket_loc = thread_context->bucket_locs->data();
    unsigned key_bucket_locs = 0;
    for (; pk_raw != pk_raw_end; ++pk_raw) {
      key_bucket_locs =
          KBucketNum<K>(this->K_bucket_num_handle_, pk_raw, storage_slice);
      // The bucket to which the key belongs is recorded to facilitate future
      // memory writes that do not recompute the redis hash
      *pbucket_loc = key_bucket_locs;
      ++pbucket_loc;

      // Direct access to Tensor data in TensorFlow
      thread_context->HandlePushBack(
          key_bucket_locs, KContentPointer<K>(pk_raw), KTypeSize<K>(pk_raw));
    }

    auto cmd = [](::sw::redis::Connection &connection,
                  const ::sw::redis::StringView hkey,
                  const std::vector<const char *> *ptrs_i,
                  const std::vector<std::size_t> *sizes_i) {
      assert(strcmp(ptrs_i->front(), "HDEL") == 0);
      assert(sizes_i->front() == 4);
      assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);

      connection.send(static_cast<int>(ptrs_i->size()),
                      const_cast<const char **>(ptrs_i->data()),
                      sizes_i->data());
    };

    std::vector<
        std::future<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>>
        results;
    try {
      for (unsigned i = 0; i < storage_slice; ++i) {
        results.emplace_back(
            network_worker_pool_ptr_->enqueue([this, &cmd, thread_context, i] {
              return PipeExecWrite(cmd, 3U, thread_context->buckets[i]);
            }));
      }
      for (auto &&result : results) {
        result.wait();
      }
      if (error_ptr_) {
        std::rethrow_exception(error_ptr_);
      }
    } catch (const std::exception &err) {
      error_ptr_ = nullptr;
      return TFRA_Status(StatusCode::UNKNOWN, err.what());
      // return errors::Unknown(err.what());
    }

    return TFRA_Status::OK();
  }
};  // class RedisWrapper
}  // namespace redis_connection
}  // namespace recommenders_addons
}  // namespace tensorflow
