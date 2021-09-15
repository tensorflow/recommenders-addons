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
#include "thread_pool.h"

using sw::redis::ConnectionOptions;
using sw::redis::ConnectionPoolOptions;
using sw::redis::Redis;
using sw::redis::RedisCluster;

namespace tensorflow {
namespace recommenders_addons {
namespace redis_connection {
template <typename RedisInstance, typename K, typename V>
class RedisWrapper<RedisInstance, K, V,
                   typename std::enable_if<
                       std::is_same<RedisInstance, RedisCluster>::value>::type>
    : public RedisVirtualWrapper {
 private:
  ConnectionOptions conn_opts;
  ConnectionPoolOptions pool_opts;
  ThreadPool *network_worker_pool;

 public:
  std::shared_ptr<RedisInstance> redis_conn;  // for the hungry singleton mode

 public:
  RedisWrapper(RedisInstance &&) = delete;
  RedisWrapper(const RedisInstance &) = delete;
  RedisWrapper &operator=(const RedisInstance &) = delete;

  ~RedisWrapper() {
    if (redis_conn == nullptr) {
      return;
    }
    redis_conn.reset();
    LOG(INFO) << "RedisCluster connection pool destructor called successfully.";
  }

 private:
  RedisWrapper()  // In singleton mode, classes should not be initialized
                  // through constructor
  {
    network_worker_pool = new ThreadPool(hardware_concurrency_);
    LOG(INFO)
        << "RedisCluster connection pool constructor called successfully.";
  }

 public:
  std::shared_ptr<RedisInstance> StartConn(size_t ip_port_count) {
    conn_opts.host = redis_connection_params.redis_host_ip[ip_port_count];
    conn_opts.port = redis_connection_params.redis_host_port[ip_port_count];
    // Redis connection options
    conn_opts.user = redis_connection_params.redis_user;
    conn_opts.password =
        redis_connection_params
            .redis_password;  // Optional. No redis_password by default.
    conn_opts.db = redis_connection_params.redis_db;
    conn_opts.keep_alive = redis_connection_params.redis_connect_keep_alive;
    conn_opts.connect_timeout = std::chrono::milliseconds(
        redis_connection_params.redis_connect_timeout);
    conn_opts.socket_timeout =
        std::chrono::milliseconds(redis_connection_params.redis_socket_timeout);
    // Redis connection pool options
    pool_opts.size = redis_connection_params.redis_conn_pool_size;
    pool_opts.wait_timeout =
        std::chrono::milliseconds(redis_connection_params.redis_wait_timeout);
    pool_opts.connection_lifetime =
        std::chrono::minutes(redis_connection_params.redis_connection_lifetime);

    try {
      static auto redis_client =
          std::make_shared<RedisInstance>(RedisInstance(conn_opts, pool_opts));
      redis_client->set("key test for connecting", "val test for connecting",
                        std::chrono::milliseconds(1));
      auto info_cluster = redis_client->command("info", "cluster");
      auto tmp_char = strtok(info_cluster->str, "\n");
      tmp_char = strtok(NULL, "\n");
      tmp_char = strtok(tmp_char, ":");
      auto cluster_bool = strtok(NULL, ":");
      if (strcmp(cluster_bool, "1\r") != 0) {
        LOG(ERROR)
            << "Now is cluster mode but try to connect Redis single node. "
               "Please check redis_connection_mode in config file.";
        throw std::invalid_argument(
            "Can not connect to single node when in cluster mode, "
            "redis_connection_mode should be 1 when connect to single node.");
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

  virtual void Conn() override {
    assert(redis_connection_params.redis_host_ip.size() ==
           redis_connection_params.redis_host_port.size());
    if (isRedisConnect == false) {
      for (size_t i = 0; i < redis_connection_params.redis_host_ip.size();
           ++i) {
        for (short j = 0; j < 10; j++) {
          redis_conn = StartConn(i);
          if (redis_conn) {
            isRedisConnect = true;
            return;
          }
        }
        LOG(WARNING) << "Can not access the host "
                     << redis_connection_params.redis_host_ip[i]
                     << ". Delete it from the host list.";
        redis_connection_params.redis_host_ip.erase(
            redis_connection_params.redis_host_ip.begin() + i);
        redis_connection_params.redis_host_port.erase(
            redis_connection_params.redis_host_port.begin() + i);
      }
      if (isRedisConnect == false) {
        LOG(ERROR) << "Can not connect to the Redis Cluster servers.";
        throw(std::runtime_error("Exit without any Redis connection."));
      }
    }
  }

  static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> get_instance() {
    /* for the Meyer's Singleton mode.
      When make Class constructor private, it will be not allow using
      make_shared to init the Class. It' not safe, but having no choice. */
    static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> instance_ptr(
        new RedisWrapper<RedisInstance, K, V>());
    return instance_ptr;
  }

 private:
  template <typename Cmd>
  std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> PipeExec(
      Cmd cmd, const unsigned &size_check,
      const std::unique_ptr<BucketContext> &bucket_context) {
    if (bucket_context->ptrs->size() >= size_check) {
      ::sw::redis::StringView hkey((*bucket_context->ptrs)[1],
                                   (*bucket_context->sizes)[1]);
      try {
        return redis_conn->command(cmd, hkey, bucket_context->ptrs.get(),
                                   bucket_context->sizes.get());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in PipeExec for slices "
                   << hkey.data() << " -- " << err.what();
      }
    } else {
      return nullptr;
    }
    return nullptr;
  }

 public:
  virtual std::vector<std::string> GetKeyBucketsAndOptimizerParamsWithName(
      const std::string &keys_prefix_name,
      const bool only_get_buckets) override {
    std::vector<std::string> keys_prefix_name_slices_in_redis;
    std::string redis_command;
    if (only_get_buckets) {
      redis_command = "KEYS " + keys_prefix_name + "{[0123456789]*}";
    } else {
      redis_command = "KEYS " + keys_prefix_name + "*{[0123456789]*}";
    }
    // get cluster info
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey) {
      connection.send("CLUSTER SLOTS");
    };
    ::sw::redis::StringView _hkey("0");
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn->command(cmd, _hkey);
    } catch (const std::exception &err) {
      LOG(ERROR)
          << "RedisHandler error in "
             "GetKeyBucketsAndOptimizerParamsWithName(CLUSTER SLOTS) --  "
          << err.what();
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
    keys_prefix_name_slices_in_redis.reserve(
        redis_connection_params.storage_slice);
    for (size_t i = 0; i < ip_port_set.size(); ++i) {
      connection_options.host = ip_port_set[i].first;  // Required.
      connection_options.port =
          ip_port_set[i].second;  // Optional. The default port is 6379.
      connection_options.user = redis_connection_params.redis_user;
      connection_options.password =
          redis_connection_params
              .redis_password;  // Optional. No redis_password by default.
      connection_options.db =
          redis_connection_params
              .redis_db;  // Optional. Use the 0th database by default.
      redis_client.reset(new Redis(connection_options));
      auto cmd_per_server = [](::sw::redis::Connection &connection,
                               const char *str) { connection.send(str); };
      reply_server.reset();
      try {
        reply_server =
            redis_client->command(cmd_per_server, redis_command.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error "
                      "GetKeyBucketsAndOptimizerParamsWithName(KEYS) in for IP "
                   << ip_port_set[i].first << " --  " << err.what();
      }
      for (size_t i = 0; i < reply_server->elements; ++i) {
        keys_prefix_name_slices_in_redis.emplace_back(std::string(
            reply_server->element[i]->str, reply_server->element[i]->len));
      }
    }
    return keys_prefix_name_slices_in_redis;
  }

  /*
    If the number of slices in the Redis service is the same as the number set
    by the user, then 1 is returned. If there is no corresponding table in the
    Redis service, 0 is returned. Other exceptions return -1.
  */
  virtual int CheckSlicesNum(const std::string &keys_prefix_name) override {
    std::vector<std::string> keys_prefix_name_slices_in_redis;
    try {
      keys_prefix_name_slices_in_redis = std::move(
          GetKeyBucketsAndOptimizerParamsWithName(keys_prefix_name, true));
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in CheckSlicesNum for KEYS "
                 << keys_prefix_name << " -- " << err.what();
      return -1;
    }
    if (keys_prefix_name_slices_in_redis.size() ==
        redis_connection_params.storage_slice) {
      LOG(INFO) << "There is already a corresponding table " << keys_prefix_name
                << " existing in Redis cluster servers";
      return 1;
    } else if (keys_prefix_name_slices_in_redis.size() == 0) {
      LOG(INFO) << "There is not a corresponding table " << keys_prefix_name
                << " existing in Redis cluster servers";
      return 0;
    } else {
      LOG(ERROR) << "storage_slice in redis_connection_params which is "
                 << redis_connection_params.storage_slice
                 << " did not equal to the slices number of this "
                 << keys_prefix_name
                 << " in the Redis Cluster servers which is "
                 << keys_prefix_name_slices_in_redis.size();
      return -1;
    }
    return -1;
  }

  virtual size_t TableSizeInBuckets(
      const std::vector<std::string> &keys_prefix_name_slices) override {
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    std::string redis_command("HLEN ");
    std::string command_string;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    size_t size = 0;
    for (unsigned i = 0; i < redis_connection_params.storage_slice; ++i) {
      command_string.clear();
      command_string =
          command_string + redis_command + keys_prefix_name_slices[i];
      reply.reset();
      try {
        reply = redis_conn->command(cmd, keys_prefix_name_slices[i],
                                    command_string.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in TableSizeInBuckets for slices "
                   << keys_prefix_name_slices[i] << " -- " << err.what();
      }
      if (reply->type == REDIS_REPLY_INTEGER)  // #define REDIS_REPLY_STRING 1
      {
        size += reply->integer;  // decimal
      }
    }

    return size;
  }

  virtual void RemoveHkeysInBuckets(
      const std::string &keys_prefix_name_slice) override {
    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    std::string redis_command("DEL ");
    std::string command_string = redis_command + keys_prefix_name_slice;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    try {
      /*reply=*/redis_conn->command(cmd, keys_prefix_name_slice,
                                    command_string.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in RemoveHkeysInBuckets for slices "
                 << keys_prefix_name_slice << " -- " << err.what();
    }
  }

  virtual std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>
  GetKeysInBucket(const std::string &keys_prefix_name_slice) override {
    std::string redis_command = "HKEYS ";
    std::string command_string;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };

    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    command_string = command_string + redis_command + keys_prefix_name_slice;
    try {
      reply = redis_conn->command(cmd, keys_prefix_name_slice,
                                  command_string.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in GetKeysInBucket for slices "
                 << keys_prefix_name_slice << " -- " << err.what();
    }

    return reply;
  }

  virtual std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> MgetInBucket(
      const Tensor &keys, const int64 begin, const int64 max_i,
      const std::string &keys_prefix_name_slice) override {
    std::unique_ptr<BucketContext> bucket_context_temp(new BucketContext());
    const static char *redis_command = "HMGET";
    const static std::size_t redis_command_byte = 5;
    const int &&total = max_i - begin;
    const int &&argc = total + 2;
    bucket_context_temp->HandleClear();
    bucket_context_temp->HandleReserve(argc);

    const K *const pk_raw_end =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + max_i;
    const K *pk_raw =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + begin;

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

    return PipeExec(cmd, 3U, bucket_context_temp);
  }

  virtual void SetExpireBuckets(const std::string &keys_prefix_name) override {
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
          std::to_string(redis_connection_params.expire_model_tag_in_seconds);
      try {
        /*reply=*/redis_conn->command(cmd, bucket_name, redis_command.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in SetExpireBuckets for "
                   << bucket_name << " -- " << err.what();
      }
    }
  }

  virtual void SetPersistBuckets(const std::string &keys_prefix_name) override {
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
        /*reply=*/redis_conn->command(cmd, bucket_name, redis_command.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in SetPersistBuckets for "
                   << bucket_name << " -- " << err.what();
      }
    }
  }

  /*
  fds are the return of POSIX open file function declared in <fcntl.h>
  */
  virtual void DumpToDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &wrs, const std::vector<int> &fds) override {
    if (fds.size() == 0) {
      return;
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
    for (unsigned i = 0; i < redis_connection_params.storage_slice; ++i) {
      redis_command = "DUMP " + keys_prefix_name_slices[i];
      reply.reset();
      try {
        reply = redis_conn->command(cmd, keys_prefix_name_slices[i],
                                    redis_command.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in DumpToDisk for slices "
                   << keys_prefix_name_slices[i] << " -- " << err.what();
      }

      wr = &wrs[i];
      if (wr->aio_nbytes > 0) {
        for (size_t i = 3; i > 0; --i) {
          while (aio_error(wr) == EINPROGRESS)
            ;
          if ((ret = aio_return(wr)) > 0) {
            // LOG(INFO) << "File handle " << wr->aio_fildes
            //           << " finished writing last round.";
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
  }

  virtual void RestoreFromDisk(
      const std::vector<std::string> &keys_prefix_name_slices,
      std::vector<aiocb> &rds, const std::vector<int> &fds,
      const std::vector<unsigned long> &buf_sizes) override {
    if (fds.size() == 0) {
      return;
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

    size_t count_down = storage_slice;
    std::vector<size_t> reread_countdown(storage_slice);
    std::string empty_str;

    while (count_down != 0) {
      for (size_t i = 0; i < storage_slice; ++i) {
        rd = &rds[i];

        if (rd->aio_nbytes > 0) {
          if (aio_error(rd) != EINPROGRESS) {
            if ((ret = aio_return(rd)) > 0) {
              // LOG(INFO) << "File handle " << rd->aio_fildes
              //           << " finished reading last round.";
              try {
                /*reply = */ redis_conn->command(
                    cmd, keys_prefix_name_slices[i], ptrs_i_i[i], sizes_i_i[i]);
              } catch (const std::exception &err) {
                LOG(ERROR)
                    << "RedisHandler error in RestoreFromDisk for slices "
                    << keys_prefix_name_slices[i] << " -- " << err.what();
              }
              free((void *)rd->aio_buf);
              rd->aio_buf = nullptr;
              rd->aio_nbytes = 0;
              --count_down;
            } else {
              LOG(WARNING) << "File handle " << rd->aio_fildes
                           << " did not finish reading last round. "
                           << "Try to read " << reread_countdown[i]
                           << " more times";
              if (reread_countdown[i] > 0) {
                ret = aio_read(rd);
                if (ret < 0) perror("aio_read");
                --reread_countdown[i];
              }
            }
          }
        }
      }
    }
  }

  void DoDuplicateInRedis(const std::string &keys_prefix_name_slice_old,
                          const std::string &keys_prefix_name_slice_new) {
    std::string redis_dump_command = "DUMP " + keys_prefix_name_slice_old;

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
      reply = redis_conn->command(cmd_dump, keys_prefix_name_slice_old,
                                  redis_dump_command.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in dump_to_reply of DoDuplicateInRedis "
                    "for slices "
                 << keys_prefix_name_slice_old << " -- " << err.what();
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
      /*reply = */ redis_conn->command(cmd_restore, keys_prefix_name_slice_new,
                                       ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in restore_from_reply of "
                    "DoDuplicateInRedis for slices "
                 << keys_prefix_name_slice_new << " -- " << err.what();
    }
  }

  virtual void DuplicateInRedis(
      const std::vector<std::string> &keys_prefix_name_slices_old,
      const std::vector<std::string> &keys_prefix_name_slices_new) override {
    for (unsigned i = 0; i < redis_connection_params.storage_slice; ++i) {
      network_worker_pool->enqueue([this, &keys_prefix_name_slices_old,
                                    &keys_prefix_name_slices_new, i] {
        DoDuplicateInRedis(keys_prefix_name_slices_old[i],
                           keys_prefix_name_slices_new[i]);
      });
    }
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
      const Tensor &keys, ThreadContext *thread_context, const int64 begin,
      const int64 max_i,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total + 2;

    const static char *redis_command = "HMGET";
    const static std::size_t &&redis_command_byte = 5;

    const K *const pk_raw_end =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + max_i;
    const K *pk_raw =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + begin;

    const unsigned &storage_slice = redis_connection_params.storage_slice;
    const unsigned &&vector_len =
        (static_cast<int64>(reinterpret_cast<int>(argc)) >>
         redis_connection_params.storage_slice_log2) +
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
      key_bucket_locs = KBucketNum<K>(pk_raw, storage_slice);
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
      results.emplace_back(
          network_worker_pool->enqueue([this, &cmd, &thread_context, i] {
            return PipeExec(cmd, 3U, thread_context->buckets[i]);
          }));
    }
    for (unsigned i = 0; i < storage_slice; ++i) {
      replies[i] = std::move(results[i].get());
    }

    return replies;
  }

  inline void CopyDefaultToTensor(const bool is_full_default, const V *pv_raw,
                                  const V *dft_raw,
                                  const V *const dft_raw_begin,
                                  const int64 Velems_per_dim0) {
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

  virtual void MgetToTensor(
      Tensor *values, const Tensor &default_value, const bool is_full_default,
      ThreadContext *thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const int64 begin, const int64 max_i,
      const int64 Velems_per_dim0) override {
    const V *pv_raw =
        reinterpret_cast<const V *>(values->tensor_data().data()) +
        begin * Velems_per_dim0;
    const V *dft_raw =
        reinterpret_cast<const V *>(default_value.tensor_data().data()) +
        begin * Velems_per_dim0;
    const V *const dft_raw_begin =
        reinterpret_cast<const V *>(default_value.tensor_data().data());

    const std::vector<unsigned> *bucket_locs =
        thread_context->bucket_locs.get();
    const unsigned &storage_slice = redis_connection_params.storage_slice;
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
          LOG(WARNING) << "Redis reply in bucket_loc " << bucket_loc
                       << " from MgetCommend has some problems with error "
                       << ", using default values to repalce.";
          print_once[bucket_loc] = true;
        }
        ++(buckets_iters_nums[bucket_loc]);
        CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                            Velems_per_dim0);
      }
    }
  }

  virtual void MsetCommand(
      const Tensor &keys, const Tensor &values, ThreadContext *thread_context,
      const int64 begin, const int64 max_i, const int64 Velems_per_dim0,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total * 2 + 2;

    const static char *redis_command = "HMSET";
    const static std::size_t &&redis_command_byte = 5;

    const K *const pk_raw_end =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + max_i;
    const K *pk_raw =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + begin;

    const std::size_t &&V_byte_size = Velems_per_dim0 * sizeof(V);

    const V *pv_raw = reinterpret_cast<const V *>(values.tensor_data().data()) +
                      begin * Velems_per_dim0;

    const unsigned &storage_slice = redis_connection_params.storage_slice;
    const unsigned &&vector_len =
        (static_cast<int64>(reinterpret_cast<int>(argc)) >>
         redis_connection_params.storage_slice_log2) +
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
          KBucketNum<K>(pk_raw, storage_slice);  // TODO: change it to AVX512

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
    for (unsigned i = 0; i < storage_slice; ++i) {
      results.emplace_back(
          network_worker_pool->enqueue([this, &cmd, &thread_context, i] {
            return PipeExec(cmd, 4U, thread_context->buckets[i]);
          }));
    }
    for (auto &&result : results) {
      result.wait();
    }
  }

  virtual void DelCommand(
      const Tensor &keys, ThreadContext *thread_context, const int64 begin,
      const int64 max_i,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total + 2;

    const static char *redis_command = "HDEL";
    const static std::size_t &&redis_command_byte = 4;

    const K *const pk_raw_end =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + max_i;
    const K *pk_raw =
        reinterpret_cast<const K *>(keys.tensor_data().data()) + begin;

    const unsigned &storage_slice = redis_connection_params.storage_slice;
    const unsigned &&vector_len =
        (static_cast<int64>(reinterpret_cast<int>(argc)) >>
         redis_connection_params.storage_slice_log2) +
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
      key_bucket_locs = KBucketNum<K>(pk_raw, storage_slice);
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
    for (unsigned i = 0; i < storage_slice; ++i) {
      results.emplace_back(
          network_worker_pool->enqueue([this, &cmd, &thread_context, i] {
            return PipeExec(cmd, 3U, thread_context->buckets[i]);
          }));
    }
    for (auto &&result : results) {
      result.wait();
    }
  }
};  // namespace redis_connection
}  // namespace redis_connection
}  // namespace recommenders_addons
}  // namespace tensorflow
