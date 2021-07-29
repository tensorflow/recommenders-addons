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
    LOG(INFO) << "RedisCluster connection pool destructor called!";
  }

 private:
  RedisWrapper()  // In singleton mode, classes should not be initialized
                  // through constructor
  {
    LOG(INFO) << "RedisCluster connection pool constructor called!";
  }

 public:
  std::shared_ptr<RedisInstance> StartConn(size_t ip_port_count) {
    conn_opts.host = redis_connection_params.redis_host_ip[ip_port_count];
    conn_opts.port = redis_connection_params.redis_host_port[ip_port_count];
    // Redis connection options
    conn_opts.password =
        redis_connection_params
            .redis_password;  // Optional. No redis_password by default.
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
      redis_client->set("key test for connecting", "val test for connecting",
                        std::chrono::milliseconds(1));
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
  std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> PipeExec(
      Cmd cmd, const unsigned &storage_slice, const unsigned &size_check,
      const ThreadContext *thread_context) {
    std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> replies;
    for (unsigned i = 0; i < storage_slice; ++i) {
      if (thread_context->slots[i]->ptrs->size() >= size_check) {
        ::sw::redis::StringView hkey((*thread_context->slots[i]->ptrs)[1],
                                     (*thread_context->slots[i]->sizes)[1]);
        try {
          replies.push_back(
              redis_conn->command(cmd, hkey, thread_context->slots[i]->ptrs,
                                  thread_context->slots[i]->sizes));
        } catch (const std::exception &err) {
          LOG(ERROR) << "RedisHandler error in pipe_exec for slices "
                     << hkey.data() << " -- " << err.what();
        }
      } else {
        replies.push_back(nullptr);
      }
    }
    return replies;
  }

 public:
  /*
    If the number of slices in the Redis service is the same as the number set
    by the user, then 1 is returned. If there is no corresponding table in the
    Redis service, 0 is returned. Other exceptions return -1.
  */
  virtual int CheckSlicesNum(const std::string &keys_prefix_name) override {
    std::string redis_command = "KEYS " + keys_prefix_name + "[0123456789]*";
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    try {
      reply = redis_conn->command(cmd, keys_prefix_name, redis_command.data());
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
        LOG(ERROR) << "RedisHandler error in table_size_in_slots for slices "
                   << keys_prefix_name_slices[i] << " -- " << err.what();
      }
      if (reply->type == REDIS_REPLY_INTEGER)  // #define REDIS_REPLY_STRING 1
      {
        size += reply->integer;  // decimal
      }
    }

    return size;
  }

  virtual void RemoveHkeysInSlots(
      const std::vector<std::string> &keys_prefix_name_slices) override {
    // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
    std::string redis_command("DEL ");
    std::string command_string;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };
    for (unsigned i = 0; i < redis_connection_params.storage_slice; ++i) {
      command_string.clear();
      command_string =
          command_string + redis_command + keys_prefix_name_slices[i];
      try {
        /*reply=*/redis_conn->command(cmd, keys_prefix_name_slices[i],
                                      command_string.data());
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in remove_hkeys_in_slots for slices "
                   << keys_prefix_name_slices[i] << " -- " << err.what();
      }
    }
  }

  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  GetKeysInHkeys(
      const std::vector<std::string> &keys_prefix_name_slices) override {
    std::string redis_command = "HKEYS ";
    std::string command_string;
    auto cmd = [](::sw::redis::Connection &connection,
                  ::sw::redis::StringView hkey,
                  const char *str) { connection.send(str); };

    auto &storage_slice = redis_connection_params.storage_slice;

    std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> replies;
    for (unsigned i = 0; i < storage_slice; ++i) {
      command_string.clear();
      command_string =
          command_string + redis_command + keys_prefix_name_slices[i];
      try {
        replies.push_back(redis_conn->command(cmd, keys_prefix_name_slices[i],
                                              command_string.data()));
      } catch (const std::exception &err) {
        LOG(ERROR) << "RedisHandler error in get_keys_in_hkeys for slices "
                   << keys_prefix_name_slices[i] << " -- " << err.what();
      }
    }

    return replies;
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
        LOG(ERROR) << "RedisHandler error in dump_to_disk for slices "
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
                    << "RedisHandler error in restore_from_disk for slices "
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
every slot has its own SlotContext for sending data---for locating the reply-
    |                                               |
    | std::vector<SlotContext>                      | std::vector<unsigned>
    |
    |
--char* point to the data and size_t indicates the length of data------------
  |                    |
  | std::vector        | std::vector
  |  <const char*>     |  <std::size_t>
  |                    |
(Real Redis command sequence because m-cmd can only be used in same hash tag)

  PS: vector slot_locs is only allocated in Redis Cluster mode!
  */
  virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>>
  MgetCommand(
      const Tensor &keys, ThreadContext *thread_context, const int64 &begin,
      const int64 &max_i,
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

    unsigned *pslot_loc = thread_context->slot_locs->data();
    unsigned key_slot_locs = 0;
    for (; pk_raw != pk_raw_end; ++pk_raw) {
      key_slot_locs = KSlotNum<K>(pk_raw, storage_slice);
      // The slot to which the key belongs is recorded to facilitate future
      // memory writes that do not recompute the hash
      *pslot_loc = key_slot_locs;
      ++pslot_loc;

      // Direct access to Tensor data in TensorFlow
      thread_context->HandlePushBack(key_slot_locs, KContentPointer<K>(pk_raw),
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

    return PipeExec(cmd, storage_slice, 3U, thread_context);
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

    const std::vector<unsigned> *slot_locs = thread_context->slot_locs;
    const unsigned &storage_slice = redis_connection_params.storage_slice;
    unsigned slots_iters_nums[storage_slice];
    unsigned slot_loc;
    memset(slots_iters_nums, 0U, sizeof(slots_iters_nums));
    redisReply *temp_reply;
    for (auto i = 0; i < (max_i - begin);
         ++i, pv_raw += Velems_per_dim0, dft_raw += Velems_per_dim0) {
      slot_loc = (*slot_locs)[i];
      if (reply[slot_loc]->type == REDIS_REPLY_ARRAY) {
        temp_reply = reply[slot_loc]->element[slots_iters_nums[slot_loc]];
        ++(slots_iters_nums[slot_loc]);
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
      } else {
        CopyDefaultToTensor(is_full_default, pv_raw, dft_raw, dft_raw_begin,
                            Velems_per_dim0);
      }
    }
  }

  virtual void MsetCommand(
      const Tensor &keys, const Tensor &values, ThreadContext *thread_context,
      const int64 &begin, const int64 &max_i, const int64 &Velems_per_dim0,
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
    unsigned key_slot_locs = 0;
    for (int i = 0; pk_raw != pk_raw_end;
         ++i, ++pk_raw, pv_raw += Velems_per_dim0) {
      VCATS_temp = VContentAndTypeSize<V>(VCATS_temp, Velems_per_dim0,
                                          V_byte_size, pv_raw, buff_temp[i]);
      key_slot_locs =
          KSlotNum<K>(pk_raw, storage_slice);  // TODO: change it to AVX512

      // Direct access to Tensor data in TensorFlow
      thread_context->HandlePushBack(key_slot_locs, KContentPointer<K>(pk_raw),
                                     KTypeSize<K>(pk_raw));
      thread_context->HandlePushBack(key_slot_locs, VCATS_temp.VContentPointer,
                                     VCATS_temp.VTypeSize);
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

    PipeExec(cmd, storage_slice, 4U, thread_context);
  }

  virtual void DelCommand(
      const Tensor &keys, ThreadContext *thread_context, const int64 &begin,
      const int64 &max_i,
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

    unsigned *pslot_loc = thread_context->slot_locs->data();
    unsigned key_slot_locs = 0;
    for (; pk_raw != pk_raw_end; ++pk_raw) {
      key_slot_locs = KSlotNum<K>(pk_raw, storage_slice);
      // The slot to which the key belongs is recorded to facilitate future
      // memory writes that do not recompute the hash
      *pslot_loc = key_slot_locs;
      ++pslot_loc;

      // Direct access to Tensor data in TensorFlow
      thread_context->HandlePushBack(key_slot_locs, KContentPointer<K>(pk_raw),
                                     KTypeSize<K>(pk_raw));
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

    PipeExec(cmd, storage_slice, 3U, thread_context);
  }
};
}  // namespace redis_connection
}  // namespace recommenders_addons
}  // namespace tensorflow
