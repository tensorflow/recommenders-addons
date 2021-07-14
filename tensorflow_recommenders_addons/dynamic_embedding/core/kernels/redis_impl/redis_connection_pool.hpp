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

#include <inttypes.h>
#include <nmmintrin.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include "tensorflow/core/framework/tensor.h"
#include <sw/redis++/connection.h>
#include <sw/redis++/connection_pool.h>
#include <sw/redis++/redis++.h>
#include <sw/redis++/sentinel.h>

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
    : public RedisVirtualWrapper {
private:
  SentinelOptions sentinel_opts;
  ConnectionOptions conn_opts;
  ConnectionPoolOptions pool_opts;

public:
  std::shared_ptr<RedisInstance> redis_conn; // for the hungry singleton mode

public:
  RedisWrapper(RedisInstance &&) = delete;
  RedisWrapper(const RedisInstance &) = delete;
  RedisWrapper &operator=(const RedisInstance &) = delete;

  ~RedisWrapper() {
    if (redis_conn == nullptr) {
      return;
    }
    redis_conn.reset();
    LOG(INFO) << "RedisSentinel connection pool destructor called!";
  }

private:
  RedisWrapper() // In singleton mode, classes should not be initialized
                 // through constructor
  {
    LOG(INFO) << "RedisSentinel connection pool constructor called!";
  }

public:
  std::shared_ptr<RedisInstance> StartConn() {
    sentinel_opts.nodes = {{redis_connection_params.redis_host_ip,
                            redis_connection_params.redis_host_port}};
    // Optional. Timeout before we successfully connect to Redis Sentinel.
    sentinel_opts.connect_timeout = std::chrono::milliseconds(
        redis_connection_params.redis_sentinel_connect_timeout);
    // Optional. Timeout before we successfully send request to or receive
    // response from Redis Sentinel.
    sentinel_opts.socket_timeout = std::chrono::milliseconds(
        redis_connection_params.sentinel_socket_timeout);

    // Redis connection options
    conn_opts.password =
        redis_connection_params
            .redis_password; // Optional. No redis_password by default.
    conn_opts.db = redis_connection_params.redis_db;
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

    auto sentinel = std::make_shared<Sentinel>(sentinel_opts);

    try {
      static auto redis_client = std::make_shared<RedisInstance>(
          RedisInstance(sentinel, redis_connection_params.redis_master_name,
                        Role::MASTER, conn_opts, pool_opts));
      redis_client->ping();
      return redis_client;
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler--other " << err.what();
      LOG(INFO)
          << "Failed to connect to the Sentinel server. Try to connect "
             "directly with the input IP address as if it were a Redis server.";
      return start_conn_without_sentinel();
    } catch (...) {
      LOG(ERROR) << "RedisHandler--other crash";
      return nullptr;
    }
    return nullptr;
  }

  std::shared_ptr<RedisInstance> start_conn_without_sentinel() {
    // Redis connection options
    conn_opts.host = redis_connection_params.redis_host_ip;
    conn_opts.port = redis_connection_params.redis_host_port;
    conn_opts.password =
        redis_connection_params
            .redis_password; // Optional. No redis_password by default.
    conn_opts.db = redis_connection_params.redis_db;
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
      redis_client->ping();
      return redis_client;
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler--other " << err.what();
      return nullptr;
    } catch (...) {
      LOG(ERROR) << "RedisHandler--other crash";
      return nullptr;
    }
    return nullptr;
  }

  virtual void Conn() override {
    if (isRedisConnect == false) {
      for (short i = 0; i < 10; i++) {
        redis_conn = StartConn();
        if (redis_conn) {
          isRedisConnect = true;
          break;
        }
      }
      if (isRedisConnect == false) {
        LOG(ERROR) << "Can not connect to the Redis Master servers.";
        throw(std::runtime_error("Exit without any Redis connection."));
      }
    }
  }

  static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> get_instance() {
    /* for the Meyer's Singleton mode.
      When make Class constructor private, it will be not allow using
      make_shared to init the Class. It' not safe, but having no choice. */
    // static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> instance_ptr =
    //     std::make_shared<RedisWrapper<RedisInstance, K, V>>();
    static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> instance_ptr(
        new RedisWrapper<RedisInstance, K, V>());
    return instance_ptr;
  }

public:
  /*
    If the number of slices in the Redis service is the same as the number set
    by the user, then 1 is returned. If there is no corresponding table in the
    Redis service, 0 is returned. Other exceptions return -1.
  */
  virtual int CheckSlicesNum(const std::string &keys_prefix_name) override {
    std::string redis_command = "KEYS " + keys_prefix_name + "[0123456789]*";
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn->command(cmd, redis_command.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in check_slices_num for KEYS "
                 << keys_prefix_name << " -- " << err.what();
      return -1;
    }
    if (reply->elements == redis_connection_params.storage_slice) {
      return 1;
    } else if (reply->elements == 0) {
      LOG(INFO) << "There is not a corresponding table " << keys_prefix_name
                << " existing in Redis server";
      return 0;
    } else {
      LOG(ERROR) << "storage_slice in redis_connection_params which is "
                 << redis_connection_params.storage_slice
                 << " did not equal to the slices number of this "
                 << keys_prefix_name
                 << " in the Redis Cluster servers which is "
                 << reply->elements;
      return -1;
    }
    return -1;
  }

  virtual size_t TableSizeInSlots(
      const std::vector<std::string> &keys_prefix_name_slices) override {
    std::string redis_command = "HLEN " + keys_prefix_name_slices[0];
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn->command(cmd, redis_command.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in table_size_in_slots for HLEN "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }
    size_t size = 0;
    if (reply->type == REDIS_REPLY_INTEGER) // #define REDIS_REPLY_STRING 1
    {
      size = reply->integer; // decimal
    }
    return size;
  }

  virtual void RemoveHkeysInSlots(
      const std::vector<std::string> &keys_prefix_name_slices) override {
    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    std::string redis_command = "DEL " + keys_prefix_name_slices[0];
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    /*reply=*/redis_conn->command(cmd, redis_command.data());
  }

  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  GetKeysInHkeys(
      const std::vector<std::string> &keys_prefix_name_slices) override {
    std::string redis_command = "HKEYS " + keys_prefix_name_slices[0];
    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };

    std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> reply;
    reply.reserve(1);
    try {
      reply.push_back(redis_conn->command(cmd, redis_command.data()));
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in get_keys_in_hkeys for HKEYS "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }

    return reply;
  }

  /*
  fds are the return of POSIX open file function declared in <fcntl.h>
  */
  virtual void
  DumpToDisk(const std::vector<std::string> &keys_prefix_name_slices,
             std::vector<aiocb> &wrs, const std::vector<int> &fds) override {
    if (fds.size() == 0) {
      return;
    }

    std::string redis_command = "DUMP " + keys_prefix_name_slices[0];
    aiocb *wr = &wrs.front();
    int ret; // int fd;

    auto cmd = [](::sw::redis::Connection &connection, const char *str) {
      connection.send(str);
    };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn->command(cmd, redis_command.data());
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in dump_to_disk for DUMP "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }

    size_t buf_len;
    volatile void *tem_aio_buf;
    // std::string file_name =
    // redis_connection_params.model_lib_abs_dir+keys_prefix_name_slices[0]+".rdb";
    // fd = open(file_name,O_WRONLY | O_APPEND);
    // if(fd < 0) perror(file_name);
    if (wr->aio_nbytes > 0) {
      for (size_t i = 3; i > 0; --i) {
        while (aio_error(wr) == EINPROGRESS)
          ;
        if ((ret = aio_return(wr)) > 0) {
          LOG(INFO) << "File handle " << wr->aio_fildes
                    << " finished writing last round.";
          break;
        } else {
          LOG(ERROR) << "File handle " << wr->aio_fildes
                     << " did not finish writing last round. "
                     << "Try to write " << i << " more times";
          ret = aio_write(wr);
          if (ret < 0)
            perror("aio_write");
        }
      }
    }
    if (reply->type == REDIS_REPLY_STRING) // #define REDIS_REPLY_STRING 1
    {
      buf_len = reply->len;
      tem_aio_buf = wr->aio_buf;
      wr->aio_buf = realloc((void *)tem_aio_buf,
                            buf_len); // Be careful! The memory requested here
                                      // should be freed somewhere!
      memcpy((void *)(wr->aio_buf), reply->str, buf_len);
      wr->aio_nbytes = buf_len;
      wr->aio_fildes = fds[0];
      wr->aio_offset = 0;
      ret = aio_write(wr);
      if (ret < 0)
        perror("aio_write");
    } else {
      LOG(ERROR) << "HKEY " << keys_prefix_name_slices[0]
                 << " does not exist in the Redis server. ";
    }
  }

  virtual void
  RestoreFromDisk(const std::vector<std::string> &keys_prefix_name_slices,
                  std::vector<aiocb> &rds, const std::vector<int> &fds,
                  const std::vector<unsigned long> &buf_sizes) override {
    if (fds.size() == 0) {
      return;
    }

    aiocb *rd = &rds.front();
    int ret;

    auto cmd = [](::sw::redis::Connection &connection,
                  const std::vector<const char *> &ptrs_0,
                  const std::vector<std::size_t> &sizes_0) {
      assert(strcmp(ptrs_0[0], "RESTORE") == 0);
      assert(sizes_0[0] == 7);
      connection.send(static_cast<int>(ptrs_0.size()),
                      const_cast<const char **>(ptrs_0.data()), sizes_0.data());
    };

    size_t buf_len;
    volatile void *tem_aio_buf;

    std::vector<const char *> ptrs_0;
    std::vector<std::size_t> sizes_0;
    ptrs_0.reserve(4);
    sizes_0.reserve(4);

    const static char *redis_command = "RESTORE";
    const static std::size_t &&redis_command_byte = 7;
    const static char *redis_command_param = "0";
    const static std::size_t &&redis_command_byte_param = 1;

    buf_len = buf_sizes[0];

    tem_aio_buf = rd->aio_buf;
    rd->aio_buf = realloc((void *)tem_aio_buf,
                          buf_len); // Be careful! The memory requested here
                                    // should be freed somewhere!
    rd->aio_nbytes = buf_len;
    rd->aio_fildes = fds[0];
    rd->aio_offset = 0;
    ret = aio_read(rd);
    if (ret < 0)
      perror("aio_read");

    ptrs_0.clear();
    ptrs_0.push_back(redis_command);
    ptrs_0.push_back(keys_prefix_name_slices[0].data());
    ptrs_0.push_back(redis_command_param);
    ptrs_0.push_back((const char *)rd->aio_buf);

    sizes_0.clear();
    sizes_0.push_back(redis_command_byte);
    sizes_0.push_back(keys_prefix_name_slices[0].size());
    sizes_0.push_back(redis_command_byte_param);
    sizes_0.push_back(rd->aio_nbytes);

    if (rd->aio_nbytes > 0) {
      for (size_t i = 3; i > 0; --i) {
        while (aio_error(rd) == EINPROGRESS)
          ;
        if ((ret = aio_return(rd)) > 0) {
          LOG(INFO) << "File handle " << rd->aio_fildes
                    << " finished reading last round.";
          break;
        } else {
          LOG(ERROR) << "File handle " << rd->aio_fildes
                     << " did not finish reading last round. "
                     << "Try to read " << i << " more times";
          ret = aio_read(rd);
          if (ret < 0)
            perror("aio_read");
        }
      }
    }

    try {
      /*std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply = */
      redis_conn->command(cmd, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in restore_from_disk for RESTORE "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }
  }

public:
  /*
  The structure of ptrs and sizes which for storing Redis command char
sequence pointer and size of parameters. For example: vector<ThreadContext>
(for multi-threads, index is thread id, also vector<vector<vector<const char
*>>>)
                                  |
                                /   \
                               /     \    upper var is outside of the
MXXX_COMMAND function /       \
-----------------------------------------------------  (std::vector<unsigned>
slot_locs for this thread) /         \  under var is inside of the
MXXX_COMMAND function
                            /           \
            vector<SlotContext>    vector<vector<const char *>>   (Different
thread, map for storing different hash tag in Redis. Reserve(storage_slice) )
                  /   \
                 /     \    better to be reserved before entering this
function /       \ -----------------------------------------------------
(std::vector<unsigned> slot_locs for this thread) /         \  be reserved in
this function
              /           \
vector<const char *>     vector<const char *>          ............ (Real
Redis command sequence because m-cmd can only be used in same hash tag)

  PS: vector slot_locs is only allocated in Redis Cluster mode!
  */
  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  MgetCommand(
      const ::tensorflow::Tensor &keys, ThreadContext &thread_context,
      const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int argc = (max_i - begin) + 2;

    const static char *redis_command = "HMGET";
    const static std::size_t redis_command_byte = 5;

    thread_context.HandleReserve(1U, argc, 0);

    std::vector<const char *> &ptrs_0 = thread_context.slots[0].ptrs;
    std::vector<std::size_t> &sizes_0 = thread_context.slots[0].sizes;

    // const ::tensorflow::int64 dim0_size = kkeys.NumElements();
    // const ::tensorflow::int64 elems_per_dim0 = keys.NumElements() /
    // dim0_size; const std::size_t key_byte_size = elems_per_dim0 *
    // sizeof(K);
    const K *const pk_raw_end =
        reinterpret_cast<K *>(keys.data()) + max_i;
    // const char *const pk_end = reinterpret_cast<const char *>(pk_raw_end);
    const K *pk_raw = reinterpret_cast<K *>(keys.data()) + begin;
    // const char *pk = reinterpret_cast<const char *>(pk_raw);

    const char **ptrs_iter = &ptrs_0[0];
    *ptrs_iter = redis_command;
    ++ptrs_iter;
    *ptrs_iter = keys_prefix_name_slices[0].data();
    ++ptrs_iter;

    std::size_t *sizes_iter = &sizes_0[0];
    *sizes_iter = redis_command_byte;
    ++sizes_iter;
    *sizes_iter = keys_prefix_name_slices[0].size();
    ++sizes_iter;

    for (; pk_raw != pk_raw_end; ++pk_raw) {
      *ptrs_iter = KContentPointer<K>(
          pk_raw); // Direct access to ::tensorflow::Tensor data in TensorFlow
      ++ptrs_iter;
      *sizes_iter = KTypeSize<K>(pk_raw); // key data char size
      ++sizes_iter;
    }

    assert(ptrs_0[0] == redis_command);
    assert(sizes_0[0] == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> &ptrs_0,
                  const std::vector<std::size_t> &sizes_0) {
      // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
      connection.send(argc, const_cast<const char **>(ptrs_0.data()),
                      sizes_0.data());
    };

    std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> reply;
    reply.reserve(1);
    try {
      reply.push_back(redis_conn->command(cmd, argc, ptrs_0, sizes_0));
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in MGET_COMMAND for HMGET "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }
    return reply;
  }

  virtual void MgetToTensor(
      ::tensorflow::Tensor *values, const ::tensorflow::Tensor &default_value,
      const bool &is_full_default, ThreadContext &thread_context,
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
          &reply,
      const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
      const ::tensorflow::int64 &Velems_per_dim0) override {
    // const ::tensorflow::int64 &&Velems_per_dim0 =
    //     values->NumElements() / values->dim_size(0);
    V *pv_raw = reinterpret_cast<V *>(values->data()) + begin * Velems_per_dim0;
    // char *pv = reinterpret_cast<const char *>(pv_raw);
    const V *dft_raw = reinterpret_cast<const V *>(default_value.data()) +
                       begin * Velems_per_dim0;
    const V *const dft_raw_begin =
        reinterpret_cast<const V *>(default_value.data());

    redisReply *temp_reply;
    for (auto i = begin; i < max_i;
         ++i, pv_raw += Velems_per_dim0, dft_raw += Velems_per_dim0) {
      temp_reply = reply[0]->element[i];
      if (temp_reply->type ==
          REDIS_REPLY_STRING) // #define REDIS_REPLY_STRING 1
      {
        ReplyMemcpyToValTensor<V>(
            pv_raw, temp_reply->str,
            Velems_per_dim0); // Direct access to Tensor data in TensorFlow
      } else {
        if (is_full_default) {
          DefaultMemcpyToTensor<V>(
              pv_raw, dft_raw,
              Velems_per_dim0); // Direct access to Tensor data in TensorFlow
        } else {
          DefaultMemcpyToTensor<V>(
              pv_raw, dft_raw_begin,
              Velems_per_dim0); // Direct access to Tensor data in TensorFlow
        }
      }
    }
  }

  virtual void MsetCommand(
      const ::tensorflow::Tensor &keys, const ::tensorflow::Tensor &values,
      ThreadContext &thread_context, const ::tensorflow::int64 &begin,
      const ::tensorflow::int64 &max_i,
      const ::tensorflow::int64 &Velems_per_dim0,
      const std::vector<std::string> &keys_prefix_name_slices) override {
    const int &&total = max_i - begin;
    const int &&argc = total * 2 + 2;

    const static char *redis_command = "HMSET";
    const static std::size_t redis_command_byte = 5;

    thread_context.HandleReserve(1U, argc, 0);

    std::vector<const char *> &ptrs_0 = thread_context.slots[0].ptrs;
    std::vector<std::size_t> &sizes_0 = thread_context.slots[0].sizes;

    const K *const pk_raw_end = reinterpret_cast<K *>(keys.data()) + max_i;
    // const char *const pk_end = reinterpret_cast<const char *>(pk_raw_end);
    const K *pk_raw = reinterpret_cast<K *>(keys.data()) + begin;
    // const char *pk = reinterpret_cast<const char *>(pk_raw);

    // const ::tensorflow::int64 &&Vdim0_size = values.dim_size(0);
    // const ::tensorflow::int64 &&Velems_per_dim0 =
    //     values.NumElements() / Vdim0_size;
    const std::size_t &&V_byte_size = Velems_per_dim0 * sizeof(V);
    // const V *const pv_raw_end = reinterpret_cast<V*>(values.data()) +
    // (max_i-begin)*Velems_per_dim0; const char *const pv_end =
    // reinterpret_cast<const char *>(pv_raw_end);
    const V *pv_raw =
        reinterpret_cast<V *>(values.data()) + begin * Velems_per_dim0;
    // const char *pv = reinterpret_cast<const char *>(pv_raw);

    const char **ptrs_iter = &ptrs_0[0];
    *ptrs_iter = redis_command;
    ++ptrs_iter;
    *ptrs_iter = keys_prefix_name_slices[0].data();
    ++ptrs_iter;

    std::size_t *sizes_iter = &sizes_0[0];
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
          pk_raw); // Direct access to ::tensorflow::Tensor data in TensorFlow
      *(++ptrs_iter) = VCATS_temp.VContentPointer;
      ++ptrs_iter;

      *sizes_iter = KTypeSize<K>(pk_raw); // key data char size
      *(++sizes_iter) = VCATS_temp.VTypeSize;
      ++sizes_iter;
    }

    assert(ptrs_0[0] == redis_command);
    assert(sizes_0[0] == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> &ptrs_0,
                  const std::vector<std::size_t> &sizes_0) {
      // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
      connection.send(argc, const_cast<const char **>(ptrs_0.data()),
                      sizes_0.data());
    };

    try {
      redis_conn->command(cmd, argc, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in MSET_COMMAND for HMSET "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }
  }

  virtual void
  DelCommand(const ::tensorflow::Tensor &keys, ThreadContext &thread_context,
             const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
             const std::vector<std::string> &keys_prefix_name_slices) override {
    const int argc = (max_i - begin) + 2;

    const static char *redis_command = "HDEL";
    const static std::size_t redis_command_byte = 4;

    thread_context.HandleReserve(1U, argc, 0);

    std::vector<const char *> &ptrs_0 = thread_context.slots[0].ptrs;
    std::vector<std::size_t> &sizes_0 = thread_context.slots[0].sizes;

    const K *const pk_raw_end =
        reinterpret_cast<K *>(keys.data()) + max_i;
    const K *pk_raw = reinterpret_cast<K *>(keys.data()) + begin;

    const char **ptrs_iter = &ptrs_0[0];
    *ptrs_iter = redis_command;
    ++ptrs_iter;
    *ptrs_iter = keys_prefix_name_slices[0].data();
    ++ptrs_iter;

    std::size_t *sizes_iter = &sizes_0[0];
    *sizes_iter = redis_command_byte;
    ++sizes_iter;
    *sizes_iter = keys_prefix_name_slices[0].size();
    ++sizes_iter;

    for (; pk_raw != pk_raw_end; ++pk_raw) {
      *ptrs_iter = KContentPointer<K>(
          pk_raw); // Direct access to ::tensorflow::Tensor data in TensorFlow
      ++ptrs_iter;
      *sizes_iter = KTypeSize<K>(pk_raw); // key data char size
      ++sizes_iter;
    }

    assert(ptrs_0[0] == redis_command);
    assert(sizes_0[0] == redis_command_byte);

    auto cmd = [](::sw::redis::Connection &connection, const int argc,
                  const std::vector<const char *> &ptrs_0,
                  const std::vector<std::size_t> &sizes_0) {
      // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
      connection.send(argc, const_cast<const char **>(ptrs_0.data()),
                      sizes_0.data());
    };

    try {
      /*auto reply=*/redis_conn->command(cmd, argc, ptrs_0, sizes_0);
    } catch (const std::exception &err) {
      LOG(ERROR) << "RedisHandler error in DEL_COMMAND for HDEL "
                 << keys_prefix_name_slices[0] << " -- " << err.what();
    }
  }
};

} // namespace redis_connection
} // namespace recommenders_addons
} // namespace tensorflow
