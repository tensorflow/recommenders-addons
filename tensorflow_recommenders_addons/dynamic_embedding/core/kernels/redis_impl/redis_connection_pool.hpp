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

#include "redis_connection_util.hpp"

using sw::redis::ConnectionOptions;
using sw::redis::ConnectionPoolOptions;
using sw::redis::Redis;
using sw::redis::Role;
using sw::redis::Sentinel;
using sw::redis::SentinelOptions;

namespace tensorflow {
namespace recommenders_addons {
namespace redis_connection {
template <typename RedisInstance, typename K, typename V>
class RedisWrapper<
    RedisInstance, K, V,
    typename std::enable_if<std::is_same<RedisInstance, Redis>::value>::type>
    : public RedisBaseWrapper<K, V> {
 private:
  SentinelOptions sentinel_opts;
  bool using_sentinel = true;
  ConnectionOptions conn_opts;
  ConnectionPoolOptions pool_opts;
  ThreadPool *network_worker_pool;
  std::exception_ptr error_ptr;

 public:
  std::shared_ptr<RedisInstance> redis_conn_read =
      nullptr;  // for the hungry singleton mode
  std::shared_ptr<RedisInstance> redis_conn_write =
      nullptr;  // for the hungry singleton mode

 public:
  RedisWrapper(RedisInstance &&) = delete;
  RedisWrapper(const RedisInstance &) = delete;
  RedisWrapper &operator=(const RedisInstance &) = delete;

  ~RedisWrapper() {
    if (redis_conn_read == nullptr && redis_conn_write == nullptr) {
      return;
    }
    redis_conn_read.reset();
    redis_conn_write.reset();
    LOG(INFO)
        << "RedisSentinel connection pool destructor called successfully.";
  }

 private:
  RedisWrapper()  // In singleton mode, classes should not be initialized
                  // through constructor
  {
    LOG(INFO)
        << "RedisSentinel connection pool constructor called successfully.";
  }

 public:
  void SetUsingSentinel(bool use_sentinel) {
    this->using_sentinel = use_sentinel;
  }

  std::shared_ptr<RedisInstance> StartConn(Role role) {
    assert(this->redis_connection_params.redis_host_ip.size() ==
           this->redis_connection_params.redis_host_port.size());

    this->SetPublicConnParams(conn_opts, pool_opts,
                              this->redis_connection_params);

    if (this->using_sentinel) {
      sentinel_opts.nodes.clear();
      for (size_t i = 0; i < this->redis_connection_params.redis_host_ip.size();
           ++i) {
        sentinel_opts.nodes.push_back(
            {this->redis_connection_params.redis_host_ip[i],
             this->redis_connection_params.redis_host_port[i]});
      }

      sentinel_opts.password =
          this->redis_connection_params.redis_sentinel_password;
      // Optional. Timeout before we successfully connect to Redis Sentinel.
      sentinel_opts.connect_timeout = std::chrono::milliseconds(
          this->redis_connection_params.redis_sentinel_connect_timeout);
      // Optional. Timeout before we successfully send request to or receive
      // response from Redis Sentinel.
      sentinel_opts.socket_timeout = std::chrono::milliseconds(
          this->redis_connection_params.redis_sentinel_socket_timeout);

      auto sentinel = std::make_shared<Sentinel>(sentinel_opts);

      try {
        auto redis_client = std::make_shared<RedisInstance>(RedisInstance(
            sentinel, this->redis_connection_params.redis_master_name, role,
            conn_opts, pool_opts));
        redis_client->ping();
        if (this->RedisClusterEnabled(redis_client) == true) {
          LOG(ERROR)
              << "Now is sentinel mode but try to connect Redis cluster nodes. "
                 "Please check redis_connection_mode in config file.";
          throw std::invalid_argument(
              "Can not connect to cluster nodes when in sentinel mode, "
              "redis_connection_mode should be 0 when connect to cluster "
              "nodes.");
        }
        return redis_client;
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler--error: " << err.what();
        LOG(INFO) << "Failed to connect to the Sentinel server. Try to connect "
                     "directly with the input IP address as if it were a Redis "
                     "server.";
        return start_conn_without_sentinel();
      } catch (...) {
        LOG(ERROR) << "RedisHandler--other crash";
        return nullptr;
      }
    } else {
      try {
        return start_conn_without_sentinel();
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler--error: " << err.what();
        return nullptr;
      }
    }
    return nullptr;
  }

  std::shared_ptr<RedisInstance> start_conn_without_sentinel() {
    // Redis connection options
    conn_opts.host = this->redis_connection_params.redis_host_ip[0];
    conn_opts.port = this->redis_connection_params.redis_host_port[0];

    this->SetPublicConnParams(conn_opts, pool_opts,
                              this->redis_connection_params);

    try {
      auto redis_client =
          std::make_shared<RedisInstance>(RedisInstance(conn_opts, pool_opts));
      redis_client->ping();
      if (this->RedisClusterEnabled(redis_client) == true) {
        LOG(ERROR)
            << "Now is single mode but try to connect Redis cluster nodes. "
               "Please check redis_connection_mode in config file.";
        throw std::invalid_argument(
            "Can not connect to cluster nodes when in single mode, "
            "redis_connection_mode should be 0 when connect to cluster nodes.");
      }
      return redis_client;
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler--error: " << err.what();
      return nullptr;
    } catch (...) {
      LOG(ERROR) << "RedisHandler--other crash";
      return nullptr;
    }
    return nullptr;
  }

  virtual Status Conn() override {
    auto role_read = Role::MASTER;
    if (this->redis_connection_params.redis_read_access_slave) {
      role_read = Role::SLAVE;
    }
    if (this->isRedisConnect == false) {
      for (short i = 0; i < 10; i++) {
        if (redis_conn_read == nullptr) {
          redis_conn_read = StartConn(role_read);
        }
        if (redis_conn_write == nullptr) {
          redis_conn_write = StartConn(Role::MASTER);
        }
        if (redis_conn_read != nullptr && redis_conn_write != nullptr) {
          this->isRedisConnect = true;
          return TFOkStatus;
        }
      }
      if (this->isRedisConnect == false) {
        LOG(ERROR) << "Can not connect to the Redis Master servers.";
        if (redis_conn_read == nullptr && redis_conn_write != nullptr) {
          return Status(error::UNAVAILABLE,
                        "Can not access Redis Slave service, Exit without any "
                        "Redis connection.");
        }
        return Status(error::UNAVAILABLE, "Exit without any Redis connection.");
      }
    }
    return TFOkStatus;
  }

  static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> get_instance(
      bool use_sentinel = true) {
    std::shared_ptr<RedisWrapper<RedisInstance, K, V>> instance_ptr(
        new RedisWrapper<RedisInstance, K, V>());
    instance_ptr->SetUsingSentinel(use_sentinel);
    return instance_ptr;
  }

 public:
  virtual std::vector<std::string> GetKeyBucketsAndOptimizerParamsWithName(
      const std::string &keys_prefix_name,
      const bool only_get_buckets) override {
    std::vector<std::string> keys_prefix_name_slices_in_redis;
    std::string redis_command;
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply_server;
    long long cursor = 0;
    const redisReply *set_reply;
    keys_prefix_name_slices_in_redis.reserve(
        this->redis_connection_params.storage_slice);
    while (true) {
      if (only_get_buckets) {
        redis_command = "SCAN " + std::to_string(cursor) + " MATCH " +
                        keys_prefix_name + "{[0123456789]*}";
      } else {
        redis_command = "SCAN " + std::to_string(cursor) + " MATCH " +
                        keys_prefix_name + "*{[0123456789]*}";
      }
      try {
        reply_server = redis_conn_read->command(cmd, redis_command.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in "
                      "GetKeyBucketsAndOptimizerParamsWithName for SCAN "
                   << keys_prefix_name << " -- " << err.what();
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
    return keys_prefix_name_slices_in_redis;
  }

  /*
    If the number of slices in the Redis service is the same as the number set
    by the user, then 1 is returned. If the former is smaller, return 2.
    If there is no corresponding table in the Redis service, 0 is returned.
    Other exceptions return -1.
  */
  virtual int CheckSlicesNum(const std::string &keys_prefix_name) override {
    std::vector<std::string> keys_prefix_name_slices_in_redis;
    try {
      keys_prefix_name_slices_in_redis = std::move(
          GetKeyBucketsAndOptimizerParamsWithName(keys_prefix_name, true));
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in CheckSlicesNum for SCAN "
                 << keys_prefix_name << " -- " << err.what();
      return -1;
    }

    if (keys_prefix_name_slices_in_redis.size() == 0) {
      LOG(INFO) << "There is not a corresponding table " << keys_prefix_name
                << " existing in Redis server";
      return 0;
    } else if (keys_prefix_name_slices_in_redis.size() ==
               this->redis_connection_params.storage_slice) {
      LOG(INFO) << "There is already a corresponding table " << keys_prefix_name
                << " existing in Redis server";
      return 1;
    } else if (keys_prefix_name_slices_in_redis.size() <=
               this->redis_connection_params.storage_slice) {
      LOG(WARNING) << "storage_slice in this->redis_connection_params which is "
                   << this->redis_connection_params.storage_slice
                   << " is bigger than the slices number of this "
                   << keys_prefix_name
                   << " in the Redis Cluster servers which is "
                   << keys_prefix_name_slices_in_redis.size();
      return 2;
    } else {
      LOG(WARNING) << "storage_slice in this->redis_connection_params which is "
                   << this->redis_connection_params.storage_slice
                   << " did not equal to the slices number of this "
                   << keys_prefix_name
                   << " in the Redis Single servers which is "
                   << keys_prefix_name_slices_in_redis.size();
      return -1;
    }
    return -1;
  }

  virtual std::vector<std::pair<unsigned, unsigned>> ClusterNodesSlots(
      bool full_slots) override {
    return std::vector<std::pair<unsigned, unsigned>>();
  }

  virtual size_t TableSizeInBucket(
      const std::string &keys_prefix_name_slice) override {
    const std::string redis_command = "HLEN " + keys_prefix_name_slice;
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn_read->command(cmd, redis_command.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in TableSizeInBucket for HLEN "
                 << keys_prefix_name_slice << " -- " << err.what();
      throw(err);
    }
    size_t size = 0;
    if (reply->type == REDIS_REPLY_INTEGER)  // #define REDIS_REPLY_STRING 1
    {
      size = reply->integer;  // decimal
    }
    return size;
  }

  virtual Status RemoveHkeysInBuckets(
      const std::string &keys_prefix_name_slice) override {
    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    const std::string redis_command = "DEL " + keys_prefix_name_slice;
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    try {
      /*reply=*/redis_conn_write->command(cmd, redis_command.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in RemoveHkeysInBuckets for "
                 << keys_prefix_name_slice << " -- " << err.what();
      return errors::Unknown(err.what());
    }
    return TFOkStatus;
  }

  virtual std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>
  HscanGetKeysValsInBucket(const std::string &keys_prefix_name_slice,
                           long long *cursor, const long long count) override {
    std::string command_string = "HSCAN " + keys_prefix_name_slice + ' ' +
                                 std::to_string(*cursor) + " COUNT " +
                                 std::to_string(count);
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn_read->command(cmd, command_string.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in HscanGetKeysValsInBucket for slices "
                 << keys_prefix_name_slice << " -- " << err.what();
    }
    if (!reply.get()) {
      return nullptr;
    }
    if (reply->element[0]->type == REDIS_REPLY_STRING) {
      // #define REDIS_REPLY_STRING 1
      *cursor = std::atoll(reply->element[0]->str);
    } else {
      return nullptr;
    }
    return reply;
  }

  virtual std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> MgetInBucket(
      const K *keys, const int64_t begin, const int64_t max_i,
      const std::string &keys_prefix_name_slice) override {
    std::unique_ptr<BucketContext> bucket_context_temp(new BucketContext());
    const static char *redis_command = "HMGET";
    const static std::size_t redis_command_byte = 5;
    const int &&argc = (max_i - begin) + 2;
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

    assert(bucket_context_temp->ptrs->front() == redis_command);
    assert(bucket_context_temp->sizes->front() == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> *ptrs_0,
                  const std::vector<std::size_t> *sizes_0) {
      connection.send(argc, const_cast<const char **>(ptrs_0->data()),
                      sizes_0->data());
    };

    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      return redis_conn_read->command(cmd, argc,
                                      bucket_context_temp->ptrs.get(),
                                      bucket_context_temp->sizes.get());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in MgetInBucket for HMGET "
                 << keys_prefix_name_slice << " -- " << err.what();
    }
    return nullptr;
  }

  virtual Status SetExpireBuckets(
      const std::string &keys_prefix_name) override {
    if (this->redis_connection_params.expire_model_tag_in_seconds >= 0) {
      // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
      const std::string expire_command("EXPIRE ");
      std::string redis_command;
      auto cmd = [](::sw::redis::Connection &connection, const char *str) {
        connection.send(str);
      };
      auto &&bucket_names =
          GetKeyBucketsAndOptimizerParamsWithName(keys_prefix_name, false);
      for (auto bucket_name : bucket_names) {
        redis_command.clear();
        redis_command =
            expire_command + bucket_name + ' ' +
            std::to_string(
                this->redis_connection_params.expire_model_tag_in_seconds);
        try {
          /*reply=*/redis_conn_write->command(cmd, redis_command.data());
        } catch (const std::exception &err) {
          LOG(ERROR) << "RedisHandler error in SetExpireBuckets for "
                     << bucket_name << " -- " << err.what();
          return errors::Unknown(err.what());
        }
      }
    }
    return TFOkStatus;
  }

  virtual Status SetPersistBuckets(
      const std::string &keys_prefix_name) override {
    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    const std::string expire_command("PERSIST ");
    std::string redis_command;
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    auto &&bucket_names =
        GetKeyBucketsAndOptimizerParamsWithName(keys_prefix_name, false);
    for (auto bucket_name : bucket_names) {
      redis_command.clear();
      redis_command = expire_command + bucket_name;
      try {
        /*reply=*/redis_conn_write->command(cmd, redis_command.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in SetPersistBuckets for "
                   << bucket_name << " -- " << err.what();
        return errors::Unknown(err.what());
      }
    }

    return TFOkStatus;
  }

  /*
  fds are the return of POSIX open file function declared in <fcntl.h>
  */
  virtual Status DumpToDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &wrs, const std::vector<int> &fds) override {
    if (fds.size() == 0) {
      return TFOkStatus;
      ;
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
    for (unsigned i = 0; i < this->redis_connection_params.storage_slice; ++i) {
      redis_command = "DUMP " + keys_prefix_name_slices[i];
      reply.reset();
      try {
        reply = redis_conn_read->command(cmd, keys_prefix_name_slices[i],
                                         redis_command.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in DumpToDisk for slices "
                   << keys_prefix_name_slices[i] << " -- " << err.what();
        return errors::Unknown(err.what());
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
            LOG(WARNING) << "File handle " << wr->aio_fildes
                         << " did not finish writing last round. "
                         << "Try to write " << i << " more times";
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
        LOG(ERROR) << "HKEY " << keys_prefix_name_slices[i]
                   << " does not exist in the Redis server. ";
      }
    }

    return TFOkStatus;
  }

  virtual Status RestoreFromDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &rds, const std::vector<int> &fds,
      const std::vector<unsigned long> &buf_sizes) override {
    if (fds.size() == 0) {
      return TFOkStatus;
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
                  /*reply = */ redis_conn_write->command(
                      cmd, keys_prefix_name_slices[i], ptrs_i_i[i],
                      sizes_i_i[i]);
                } catch (const std::exception &err) {
                  LOG(ERROR)
                      << "RedisHandler error in RestoreFromDisk for slices "
                      << keys_prefix_name_slices[i] << " -- " << err.what();
                  if (rd->aio_buf) {
                    free((void *)rd->aio_buf);
                    rd->aio_buf = nullptr;
                    rd->aio_nbytes = 0;
                  }
                  return errors::Unknown(err.what());
                }
                if (rd->aio_buf) {
                  free((void *)rd->aio_buf);
                  rd->aio_buf = nullptr;
                  rd->aio_nbytes = 0;
                }
                --count_down;
              } else {
                LOG(WARNING) << "File handle " << rd->aio_fildes
                             << " did not finish reading last round. "
                             << "Try to read " << reread_countdown[i] - 1
                             << " more times";
                ret = aio_read(rd);
                if (ret < 0) perror("aio_read");
                --reread_countdown[i];
              }
            }
          } else {
            LOG(WARNING) << "File handle " << rd->aio_fildes << " for slice "
                         << keys_prefix_name_slices[i]
                         << " has nbytes 0. Ignore.";
            reread_countdown[i] = 0;
            --count_down;
            if (rd->aio_buf) {
              free((void *)rd->aio_buf);
              rd->aio_buf = nullptr;
              rd->aio_nbytes = 0;
            }
          }
        } else if (reread_countdown[i] == 1) {
          LOG(ERROR) << "File handle " << rd->aio_fildes << " for slice "
                     << keys_prefix_name_slices[i]
                     << " has some troubles! Given up.";
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
               ? TFOkStatus
               : errors::Unknown("Unknown errors happen in file handles.");
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

    LOG(INFO) << "Now try to duplicate the KV pair from "
              << keys_prefix_name_slice_old << " to "
              << keys_prefix_name_slice_new;

    try {
      reply = redis_conn_read->command(cmd_dump, keys_prefix_name_slice_old,
                                       redis_dump_command.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in dump_to_reply of DoDuplicateInRedis "
                    "for slices "
                 << keys_prefix_name_slice_old << " -- " << err.what();
      error_ptr = std::current_exception();
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
      LOG(ERROR) << "HKEY " << keys_prefix_name_slice_new
                 << " does not exist in the Redis server. ";
    }
    try {
      /*reply = */ redis_conn_write->command(
          cmd_restore, keys_prefix_name_slice_new, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in restore_from_reply of "
                    "DoDuplicateInRedis for slices "
                 << keys_prefix_name_slice_new << " -- " << err.what();
      error_ptr = std::current_exception();
      throw(err);
    }
  }

  virtual Status DuplicateInRedis(
      const std::vector<std::string> &keys_prefix_name_slices_old,
      const std::vector<std::string> &keys_prefix_name_slices_new) override {
    try {
      for (unsigned i = 0; i < this->redis_connection_params.storage_slice;
           ++i) {
        network_worker_pool->enqueue([this, &keys_prefix_name_slices_old,
                                      &keys_prefix_name_slices_new, i] {
          DoDuplicateInRedis(keys_prefix_name_slices_old[i],
                             keys_prefix_name_slices_new[i]);
        });
      }
      if (error_ptr) {
        std::rethrow_exception(error_ptr);
      }
    } catch (const std::exception &err) {
      error_ptr = nullptr;
      return errors::Unknown(err.what());
    }
    return TFOkStatus;
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
  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  MgetCommand(
      const K *keys, ThreadContext *thread_context, const int64_t begin,
      const int64_t max_i,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int argc = (max_i - begin) + 2;

    const static char *redis_command = "HMGET";
    const static std::size_t redis_command_byte = 5;

    thread_context->HandleReserve(1U, argc, 0);

    std::vector<const char *> *ptrs_0 = thread_context->buckets[0]->ptrs.get();
    std::vector<std::size_t> *sizes_0 = thread_context->buckets[0]->sizes.get();

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    auto ptrs_iter = ptrs_0->begin();
    *ptrs_iter = redis_command;
    ++ptrs_iter;
    *ptrs_iter = keys_prefix_name_slices[0].data();
    ++ptrs_iter;

    auto sizes_iter = sizes_0->begin();
    *sizes_iter = redis_command_byte;
    ++sizes_iter;
    *sizes_iter = keys_prefix_name_slices[0].size();
    ++sizes_iter;

    for (; pk_raw != pk_raw_end; ++pk_raw) {
      *ptrs_iter = KContentPointer<K>(
          pk_raw);  // Direct access to Tensor data in TensorFlow
      ++ptrs_iter;
      *sizes_iter = KTypeSize<K>(pk_raw);  // key data char size
      ++sizes_iter;
    }

    assert(ptrs_0->front() == redis_command);
    assert(sizes_0->front() == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> *ptrs_0,
                  const std::vector<std::size_t> *sizes_0) {
      connection.send(argc, const_cast<const char **>(ptrs_0->data()),
                      sizes_0->data());
    };

    std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> reply;
    try {
      reply.push_back(redis_conn_read->command(cmd, argc, ptrs_0, sizes_0));
    } catch (const std::exception &err) {
      reply.push_back(nullptr);
      LOG(ERROR) << "RedisHandler error in MGET_COMMAND for HMGET "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }
    return reply;
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

  virtual Status MgetToTensor(
      V *values, const V *default_value, const bool is_full_default,
      ThreadContext *thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0) override {
    const V *pv_raw = values + begin * Velems_per_dim0;

    const V *dft_raw = default_value + begin * Velems_per_dim0;
    const V *const dft_raw_begin = default_value;

    redisReply *temp_reply;
    bool print_once = false;
    for (auto i = 0; i < max_i - begin;
         ++i, pv_raw += Velems_per_dim0, dft_raw += Velems_per_dim0) {
      if (reply[0] != nullptr) {
        if (reply[0]->type == REDIS_REPLY_ARRAY) {
          temp_reply = reply[0]->element[i];
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
        if (!print_once) {
          LOG(WARNING)
              << "Redis reply from MgetCommend has some problems with error "
              << ", using default values to repalce.";
          print_once = true;
        }
        CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                            Velems_per_dim0);
      }
    }

    return TFOkStatus;
  }

  virtual Status MgetToTensorWithExist(
      V *values, const V *default_value, bool *exists,
      const bool is_full_default, ThreadContext *thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0) override {
    const V *pv_raw = values + begin * Velems_per_dim0;

    const V *dft_raw = default_value + begin * Velems_per_dim0;
    const V *const dft_raw_begin = default_value;

    redisReply *temp_reply;
    bool print_once = false;
    for (int64_t i = 0, j = begin; i < max_i - begin;
         ++i, ++j, pv_raw += Velems_per_dim0, dft_raw += Velems_per_dim0) {
      if (reply[0] != nullptr) {
        if (reply[0]->type == REDIS_REPLY_ARRAY) {
          temp_reply = reply[0]->element[i];
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
        if (!print_once) {
          LOG(WARNING)
              << "Redis reply from MgetCommend has some problems with error "
              << ", using default values to repalce.";
          print_once = true;
        }
        CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                            Velems_per_dim0);
        exists[j] = false;
      }
    }

    return TFOkStatus;
  }

  virtual Status MsetCommand(
      const K *keys, const V *values, ThreadContext *thread_context,
      const int64_t begin, const int64_t max_i, const int64_t Velems_per_dim0,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total * 2 + 2;

    const static char *redis_command = "HMSET";
    const static std::size_t redis_command_byte = 5;

    thread_context->HandleReserve(1U, argc, 0);

    std::vector<const char *> *ptrs_0 = thread_context->buckets[0]->ptrs.get();
    std::vector<std::size_t> *sizes_0 = thread_context->buckets[0]->sizes.get();

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    const std::size_t &&V_byte_size = Velems_per_dim0 * sizeof(V);

    const V *pv_raw = values + begin * Velems_per_dim0;

    auto ptrs_iter = ptrs_0->begin();
    *ptrs_iter = redis_command;
    ++ptrs_iter;
    *ptrs_iter = keys_prefix_name_slices[0].data();
    ++ptrs_iter;

    auto sizes_iter = sizes_0->begin();
    *sizes_iter = redis_command_byte;
    ++sizes_iter;
    *sizes_iter = keys_prefix_name_slices[0].size();
    ++sizes_iter;

    VContentAndTypeSizeResult VCATS_temp;
    // std::vector<char> for storage all string in one KV pair
    std::vector<std::vector<char>> buff_temp(total);

    for (int i = 0; pk_raw != pk_raw_end;
         ++i, ++pk_raw, pv_raw += Velems_per_dim0) {
      VCATS_temp = VContentAndTypeSize<V>(VCATS_temp, Velems_per_dim0,
                                          V_byte_size, pv_raw, buff_temp[i]);

      *ptrs_iter = KContentPointer<K>(
          pk_raw);  // Direct access to Tensor data in TensorFlow
      *(++ptrs_iter) = VCATS_temp.VContentPointer;
      ++ptrs_iter;

      *sizes_iter = KTypeSize<K>(pk_raw);  // key data char size
      *(++sizes_iter) = VCATS_temp.VTypeSize;
      ++sizes_iter;
    }

    assert(ptrs_0->front() == redis_command);
    assert(sizes_0->front() == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> *ptrs_0,
                  const std::vector<std::size_t> *sizes_0) {
      connection.send(argc, const_cast<const char **>(ptrs_0->data()),
                      sizes_0->data());
    };

    try {
      redis_conn_write->command(cmd, argc, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in MSET_COMMAND for HMSET "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
      return errors::Unknown(err.what());
    }

    return TFOkStatus;
  }

  virtual Status MaccumCommand(
      const K *keys, const V *values_or_delta, const bool *exists,
      ThreadContext *thread_context, const int64_t begin, const int64_t max_i,
      const int64_t Velems_per_dim0, std::string &values_dtype_str,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total * 2 + 4;

    const static char *redis_command = "HMACCUM";
    const static std::size_t redis_command_byte = 7;

    thread_context->HandleReserve(1U, argc, 0);

    std::vector<const char *> *ptrs_0 = thread_context->buckets[0]->ptrs.get();
    std::vector<std::size_t> *sizes_0 = thread_context->buckets[0]->sizes.get();

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    const std::size_t &&V_byte_size = Velems_per_dim0 * sizeof(V);

    const V *pv_raw = values_or_delta + begin * Velems_per_dim0;

    auto ptrs_iter = ptrs_0->begin();
    *ptrs_iter = redis_command;
    ++ptrs_iter;
    *ptrs_iter = keys_prefix_name_slices[0].data();
    ++ptrs_iter;
    *ptrs_iter = values_dtype_str.c_str();
    ++ptrs_iter;

    auto sizes_iter = sizes_0->begin();
    *sizes_iter = redis_command_byte;
    ++sizes_iter;
    *sizes_iter = keys_prefix_name_slices[0].size();
    ++sizes_iter;
    *sizes_iter = values_dtype_str.size();
    ++sizes_iter;

    VContentAndTypeSizeResult VCATS_temp;
    // std::vector<char> for storage all string in one KV pair
    std::vector<std::vector<char>> buff_temp(total);

    for (int i = 0; pk_raw != pk_raw_end;
         ++i, ++pk_raw, pv_raw += Velems_per_dim0) {
      VCATS_temp = VContentAndTypeSize<V>(VCATS_temp, Velems_per_dim0,
                                          V_byte_size, pv_raw, buff_temp[i]);

      *ptrs_iter = KContentPointer<K>(
          pk_raw);  // Direct access to Tensor data in TensorFlow
      *(++ptrs_iter) = VCATS_temp.VContentPointer;
      ++ptrs_iter;

      *sizes_iter = KTypeSize<K>(pk_raw);  // key data char size
      *(++sizes_iter) = VCATS_temp.VTypeSize;
      ++sizes_iter;
    }

    const bool *pe_raw = exists + begin;
    *ptrs_iter = KContentPointer<bool>(pe_raw);
    *sizes_iter = total * KTypeSize<bool>(pe_raw);

    assert(ptrs_0->front() == redis_command);
    assert(sizes_0->front() == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> *ptrs_0,
                  const std::vector<std::size_t> *sizes_0) {
      connection.send(argc, const_cast<const char **>(ptrs_0->data()),
                      sizes_0->data());
    };

    try {
      redis_conn_write->command(cmd, argc, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in MACCUM_COMMAND for HMACCUM "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
      return errors::Unknown(err.what());
    }

    return TFOkStatus;
  }

  virtual Status DelCommand(
      const K *keys, ThreadContext *thread_context, const int64_t begin,
      const int64_t max_i,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int argc = (max_i - begin) + 2;

    const static char *redis_command = "HDEL";
    const static std::size_t redis_command_byte = 4;

    thread_context->HandleReserve(1U, argc, 0);

    std::vector<const char *> *ptrs_0 = thread_context->buckets[0]->ptrs.get();
    std::vector<std::size_t> *sizes_0 = thread_context->buckets[0]->sizes.get();

    const K *const pk_raw_end = keys + max_i;
    const K *pk_raw = keys + begin;

    auto ptrs_iter = ptrs_0->begin();
    *ptrs_iter = redis_command;
    ++ptrs_iter;
    *ptrs_iter = keys_prefix_name_slices[0].data();
    ++ptrs_iter;

    auto sizes_iter = sizes_0->begin();
    *sizes_iter = redis_command_byte;
    ++sizes_iter;
    *sizes_iter = keys_prefix_name_slices[0].size();
    ++sizes_iter;

    for (; pk_raw != pk_raw_end; ++pk_raw) {
      *ptrs_iter = KContentPointer<K>(
          pk_raw);  // Direct access to Tensor data in TensorFlow
      ++ptrs_iter;
      *sizes_iter = KTypeSize<K>(pk_raw);  // key data char size
      ++sizes_iter;
    }

    assert(ptrs_0->front() == redis_command);
    assert(sizes_0->front() == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> *ptrs_0,
                  const std::vector<std::size_t> *sizes_0) {
      connection.send(argc, const_cast<const char **>(ptrs_0->data()),
                      sizes_0->data());
    };

    try {
      /*auto reply=*/redis_conn_write->command(cmd, argc, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in DEL_COMMAND for HDEL "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
      return errors::Unknown(err.what());
    }

    return TFOkStatus;
  }
};  // namespace redis_connection

}  // namespace redis_connection
}  // namespace recommenders_addons
}  // namespace tensorflow
