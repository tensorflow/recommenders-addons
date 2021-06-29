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

#include <unistd.h>
#include <inttypes.h>
#include <chrono>
#include <iostream>

#include <openssl/md5.h>

#include "tensorflow/core/framework/tensor.h"

#include <sw/redis++/redis++.h>
#include <sw/redis++/sentinel.h>
#include <sw/redis++/connection.h>
#include <sw/redis++/connection_pool.h>

#include "redis_connection_util.hpp"

namespace sw::redis
{
  namespace redis_connection
  {
    template <typename RedisInstance, typename K, typename V>
    class RedisWrapper<RedisInstance, K, V, typename std::enable_if<std::is_same<RedisInstance, RedisCluster>::value>::type> : public RedisVirtualWrapper
    {
    private:
      ConnectionOptions conn_opts;
      ConnectionPoolOptions pool_opts;

    public:
      std::shared_ptr<RedisInstance> redis_conn; //for the hungry singleton mode

    public:
      RedisWrapper(RedisInstance &&) = delete;
      RedisWrapper(const RedisInstance &) = delete;
      RedisWrapper &operator=(const RedisInstance &) = delete;

      RedisWrapper() // In singleton mode, classes should not be initialized through constructor
      {
        std::cout << "RedisCluster connection pool constructor called!" << std::endl;
      }

      ~RedisWrapper()
      {
        if (redis_conn == nullptr)
        {
          return;
        }
        redis_conn.reset();
        std::cout << "RedisCluster connection pool destructor called!" << std::endl;
      }

    public:
      std::shared_ptr<RedisInstance> start_conn()
      {
        conn_opts.host = redis_connection_params.host_ip;
        conn_opts.port = redis_connection_params.host_port;
        // Redis connection options
        conn_opts.password = redis_connection_params.password; // Optional. No password by default.
        conn_opts.db = redis_connection_params.db;
        conn_opts.connect_timeout = std::chrono::milliseconds(redis_connection_params.connect_timeout);
        conn_opts.socket_timeout = std::chrono::milliseconds(redis_connection_params.socket_timeout);
        // Redis connection pool options
        pool_opts.size = redis_connection_params.pool_size;
        pool_opts.wait_timeout = std::chrono::milliseconds(redis_connection_params.wait_timeout);
        pool_opts.connection_lifetime = std::chrono::minutes(redis_connection_params.connection_lifetime);

        try
        {
          static auto redis_client = std::make_shared<RedisInstance>(RedisInstance(conn_opts, pool_opts));
          redis_client->set("key test for connecting", "val test for connecting", std::chrono::milliseconds(1));
          return redis_client;
        }
        catch (const std::exception &err)
        {
          std::cerr << "RedisHandler--other " << err.what() << std::endl;
          return nullptr;
        }
        catch (...)
        {
          std::cerr << "RedisHandler--other crash" << std::endl;
          return nullptr;
        }
        return nullptr;
      }

      virtual void conn() override
      {
        if (isRedisConnect == false)
        {
          for (short i = 0; i < 10; i++)
          {
            redis_conn = start_conn();
            if (redis_conn)
            {
              isRedisConnect = true;
              break;
            }
          }
          if (isRedisConnect == false)
          {
            std::cerr << "Can not connect to the Redis Cluster servers." << std::endl;
            throw(std::runtime_error("Exit without any Redis connection."));
          }
        }
      }

      static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> get_instance()
      {
        // for the Meyer's Singleton mode
        static std::shared_ptr<RedisWrapper<RedisInstance, K, V>> instance_ptr = std::make_shared<RedisWrapper<RedisInstance, K, V>>();
        return instance_ptr;
      }

    private:
      template <typename Cmd>
      std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> pipe_exec(
          Cmd cmd, const unsigned &storage_slice, const unsigned &size_check, const ThreadContext &thread_context)
      {
        std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> replies;
        replies.reserve(storage_slice);
        for (unsigned i = 0; i < storage_slice; ++i)
        {
          if (thread_context.slots[i].ptrs.size() >= size_check)
          {
            ::sw::redis::StringView hkey(thread_context.slots[i].ptrs[1], thread_context.slots[i].sizes[1]);
            replies.push_back(redis_conn->command(cmd, hkey, thread_context.slots[i].ptrs, thread_context.slots[i].sizes));
          }
          else
          {
            replies.push_back(nullptr);
          }
        }
        return replies;
      }

    public:
      virtual bool check_slices_num(const std::string &keys_prefix_name) override
      {
        std::string redis_command = "keys *" + keys_prefix_name + "*";

        // get cluster info
        auto cmd = [](::sw::redis::Connection &connection, ::sw::redis::StringView hkey)
        { connection.send("CLUSTER SLOTS"); };
        ::sw::redis::StringView _hkey("0");
        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply = redis_conn->command(cmd, _hkey);

        std::vector<std::string> IP_set;
        size_t servers_num = reply->elements;
        IP_set.reserve(servers_num);
        for (size_t i = 0; i < servers_num; ++i)
        {
          IP_set.emplace_back(std::string(reply->element[i]->element[2]->element[0]->str, reply->element[i]->element[2]->element[0]->len));
        }

        std::unique_ptr<Redis> redis_client;
        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply_server;
        ConnectionOptions connection_options;
        size_t slices_in_redis = 0;
        for (size_t i = 0; i < IP_set.size(); ++i)
        {
          connection_options.host = IP_set[i];                            // Required.
          connection_options.port = redis_connection_params.host_port;    // Optional. The default port is 6379.
          connection_options.password = redis_connection_params.password; // Optional. No password by default.
          connection_options.db = redis_connection_params.db;             // Optional. Use the 0th database by default.
          redis_client.reset(new Redis(connection_options));
          auto cmd_per_server = [](::sw::redis::Connection &connection, const char *str)
          { connection.send(str); };
          reply_server.reset();
          reply_server = redis_client->command(cmd_per_server, redis_command.data());
          slices_in_redis += reply_server->elements;
        }

        if (slices_in_redis != redis_connection_params.storage_slice)
        {
          std::cerr << "storage_slice in redis_connection_params did not equal to the slices number of this keys_prefix_name in the Redis server" << std::endl;
          return false;
        }
        else
        {
          return true;
        }
        return false;
      }

      virtual size_t table_size_in_slots(const std::vector<std::string> &keys_prefix_name_slices) override
      {
        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
        std::string redis_command("HLEN ");
        std::string command_string;
        auto cmd = [](::sw::redis::Connection &connection, ::sw::redis::StringView hkey, const char *str)
        { connection.send(str); };
        size_t size = 0;
        for (unsigned i = 0; i < redis_connection_params.storage_slice; ++i)
        {
          command_string.clear();
          command_string = command_string + redis_command + keys_prefix_name_slices[i];
          reply.reset();
          reply = redis_conn->command(cmd, keys_prefix_name_slices[i], command_string.data());
          size += strtoumax(reply->element[0]->str, nullptr, 10); // decimal
        }

        return size;
      }

      virtual void remove_hkeys_in_slots(const std::vector<std::string> &keys_prefix_name_slices) override
      {
        // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
        std::string redis_command("DEL ");
        std::string command_string;
        auto cmd = [](::sw::redis::Connection &connection, ::sw::redis::StringView hkey, const char *str)
        { connection.send(str); };
        for (unsigned i = 0; i < redis_connection_params.storage_slice; ++i)
        {
          command_string.clear();
          command_string = command_string + redis_command + keys_prefix_name_slices[i];
          /*reply=*/redis_conn->command(cmd, keys_prefix_name_slices[i], command_string.data());
        }
      }

      /*
      fds are the return of POSIX open file function declared in <fcntl.h> 
      */
      virtual void dump_to_disk(const std::vector<std::string> &keys_prefix_name_slices, std::vector<aiocb> &wrs, const std::vector<int> &fds) override
      {
        std::string redis_command;
        aiocb *wr;
        int ret; // int fd;

        auto cmd = [](::sw::redis::Connection &connection, ::sw::redis::StringView hkey, const char *str)
        { connection.send(str); };
        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;

        size_t buf_len;
        for (unsigned i = 0; i < redis_connection_params.storage_slice; ++i)
        {
          redis_command = "DUMP " + keys_prefix_name_slices[i];
          reply.reset();
          reply = redis_conn->command(cmd, keys_prefix_name_slices[i], redis_command.data());

          // std::string file_name = redis_connection_params.model_lib_abs_dir+keys_prefix_name_slices[i]+".rdb";
          // fd = open(file_name,O_WRONLY | O_APPEND);
          // if(fd < 0) perror(file_name);
          wr = &wrs[i];
          if (wr->aio_nbytes > 0)
          {
            for (size_t i = 3; i > 0; --i)
            {
              while (aio_error(wr) == EINPROGRESS)
                ;
              if ((ret = aio_return(wr)) > 0)
              {
                std::cout << "File handle " << wr->aio_fildes << " finished writing last round." << std::endl;
                break;
              }
              else
              {
                std::cerr << "File handle " << wr->aio_fildes << " did not finish writing last round. "
                          << "Try to write " << i << " more times" << std::endl;
                ret = aio_write(wr);
                if (ret < 0)
                  perror("aio_write");
              }
            }
          }
          free((void *)(wr->aio_buf)); // Dangerous behavior! Note that when creating AIOCB objects, you need to set aio_buf to nullptr!
          wr->aio_buf = nullptr;
          bzero(wr, sizeof(*wr));
          buf_len = reply->element[0]->len;
          wr->aio_buf = malloc(buf_len);
          memcpy((void *)(wr->aio_buf), reply->element[0]->str, buf_len);
          wr->aio_nbytes = buf_len;
          wr->aio_fildes = fds[i];
          wr->aio_offset = 0;
          ret = aio_write(wr);
          if (ret < 0)
            perror("aio_write");
        }
      }

      virtual void restore_from_disk(const std::vector<std::string> &keys_prefix_name_slices, std::vector<aiocb> &rds,
                                     const std::vector<int> &fds, const std::vector<unsigned long> &buf_sizes) override
      {
        // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
        const unsigned &storage_slice = redis_connection_params.storage_slice;
        std::vector<std::string> redis_command(storage_slice);
        std::string tmp_redis_command;
        // "RESTORE "=8, '0'=1, reset for enough mem space, because keys_prefix_name_slices have different length.
        size_t command_capacity = keys_prefix_name_slices[0].size() + 19;
        aiocb *rd;
        int ret; // int fd;

        auto cmd = [](::sw::redis::Connection &connection, ::sw::redis::StringView hkey, const char *str)
        { connection.send(str); };

        size_t buf_len;
        size_t tmp_redis_command_size;

        for (size_t i = 0; i < storage_slice; ++i)
        {
          rd = &rds[i];
          bzero(rd, sizeof(*rd));

          buf_len = buf_sizes[i];

          tmp_redis_command = "RESTORE " + keys_prefix_name_slices[i] + " 0";
          tmp_redis_command_size = tmp_redis_command.size();
          redis_command[i].reserve(command_capacity + buf_len + 1);
          redis_command[i].replace(0, tmp_redis_command_size, tmp_redis_command);
          redis_command[i].replace(tmp_redis_command_size, command_capacity - tmp_redis_command_size, command_capacity - tmp_redis_command_size, ' ');

          rd->aio_buf = &redis_command[i][command_capacity];
          rd->aio_nbytes = buf_len;
          rd->aio_fildes = fds[i];
          rd->aio_offset = 0;
          ret = aio_read(rd);
          if (ret < 0)
            perror("aio_read");
        }

        size_t count_down = storage_slice;
        std::vector<size_t> reread_countdown(storage_slice);
        std::string empty_str;

        while (count_down != 0)
        {
          for (size_t i = 0; i < storage_slice; ++i)
          {
            rd = &rds[i];

            if (rd->aio_nbytes > 0)
            {
              if (aio_error(rd) != EINPROGRESS)
              {
                if ((ret = aio_return(rd)) > 0)
                {
                  std::cout << "File handle " << rd->aio_fildes << " finished reading last round." << std::endl;
                  /*reply = */ redis_conn->command(cmd, keys_prefix_name_slices[i], redis_command[i].data());
                  redis_command[i].swap(empty_str);
                  bzero(rd, sizeof(*rd));
                  --count_down;
                }
                else
                {
                  std::cerr << "File handle " << rd->aio_fildes << " did not finish reading last round. "
                            << "Try to rdite " << reread_countdown[i] << " more times" << std::endl;
                  if (reread_countdown[i] > 0)
                  {
                    ret = aio_read(rd);
                    if (ret < 0)
                      perror("aio_read");
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
      The structure of ptrs and sizes which for storing Redis command char sequence pointer and size of parameters. For example:
                            vector<ThreadContext>    (for multi-threads, index is thread id, also vector<vector<vector<const char *>>>)
                                      |
                                    /   \
                                   /     \    upper var is outside of the MXXX_COMMAND function
                                  /       \ -----------------------------------------------------  (std::vector<unsigned> slot_locs for this thread)
                                 /         \  under var is inside of the MXXX_COMMAND function
                                /           \
                vector<SlotContext>    vector<vector<const char *>>   (Different thread, map for storing different hash tag in Redis. Reserve(storage_slice) )
                      /   \
                     /     \    better to be reserved before entering this function
                    /       \ -----------------------------------------------------     (std::vector<unsigned> slot_locs for this thread)
                   /         \  be reserved in this function
                  /           \
    vector<const char *>     vector<const char *>          ............        (Real Redis command sequence because m-cmd can only be used in same hash tag)

      PS: vector slot_locs is only allocated in Redis Cluster mode!
      */
      virtual std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> MGET_COMMAND(
          const ::tensorflow::Tensor &keys, ThreadContext &thread_context,
          const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
          const std::vector<std::string> &keys_prefix_name_slices) override
      {
        const int &&total = max_i - begin;
        const int &&argc = total + 2;

        const static char *redis_command = "HMGET";
        const static std::size_t &&redis_command_byte = 5;

        // const ::tensorflow::int64 dim0_size = keys.dim_size(0);
        // const ::tensorflow::int64 elems_per_dim0 = keys.NumElements() / dim0_size;
        // const std::size_t key_byte_size = elems_per_dim0 * sizeof(K);
        const K *const pk_raw_end = reinterpret_cast<K *>(keys.data()) + (total);
        // const char *const pk_end = reinterpret_cast<const char *>(pk_raw_end);
        const K *pk_raw = reinterpret_cast<K *>(keys.data()) + begin;
        // const char *pk = reinterpret_cast<const char *>(pk_raw);

        const unsigned &storage_slice = redis_connection_params.storage_slice;
        const unsigned &&vector_len = (static_cast<::tensorflow::int64>(reinterpret_cast<int>(argc)) >> redis_connection_params.storage_slice_log2) + 2;

        thread_context.HandleReserve(storage_slice, vector_len, total);

        for (unsigned i = 0; i < storage_slice; ++i)
        {
          thread_context.HandlePushBack(i, redis_command, redis_command_byte);
          thread_context.HandlePushBack(i, keys_prefix_name_slices[i].data(), keys_prefix_name_slices[i].size());
        }

        unsigned *pslot_loc = thread_context.slot_locs.data();
        unsigned key_slot_locs = 0;
        for (; pk_raw != pk_raw_end; ++pk_raw)
        {
          key_slot_locs = KSlotNum<K>(pk_raw, storage_slice);
          // The slot to which the key belongs is recorded to facilitate future memory writes that do not recompute the hash
          *pslot_loc = key_slot_locs;
          ++pslot_loc;

          // Direct access to ::tensorflow::Tensor data in TensorFlow
          thread_context.HandlePushBack(key_slot_locs, KContentPointer<K>(pk_raw), KTypeSize<K>(pk_raw));
        }

        auto cmd = [](::sw::redis::Connection &connection, const ::sw::redis::StringView hkey,
                      const std::vector<const char *> &ptrs_i, const std::vector<std::size_t> &sizes_i)
        {
          assert(strcmp(ptrs_i[0], "HMGET") == 0);
          assert(sizes_i[0] == 5);
          assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);
          // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
          connection.send(static_cast<int>(ptrs_i.size()), const_cast<const char **>(ptrs_i.data()), sizes_i.data());
        };

        return pipe_exec(cmd, storage_slice, 3U, thread_context);
      }

      virtual void MGET_to_Tensor(
          ::tensorflow::Tensor *values, const ::tensorflow::Tensor &default_value, const bool &is_full_default,
          ThreadContext &thread_context, std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> &reply,
          const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i) override
      {
        const ::tensorflow::int64 Velems_per_dim0 = values->NumElements() / values->dim_size(0);
        V *pv_raw = reinterpret_cast<V *>(values->data()) + begin * Velems_per_dim0;
        // char *pv = reinterpret_cast<const char *>(pv_raw);
        const V *dft_raw = reinterpret_cast<const V *>(default_value.data()) + begin * Velems_per_dim0;
        const V *const dft_raw_begin = reinterpret_cast<const V *>(default_value.data());

        const std::vector<unsigned> &slot_locs = thread_context.slot_locs;
        const unsigned storage_slice = redis_connection_params.storage_slice;
        unsigned slots_iters_nums[storage_slice];
        unsigned slot_loc;
        memset(slots_iters_nums, 0, storage_slice);
        redisReply *temp_reply;
        for (auto i = 0; i < (max_i - begin); ++i, pv_raw += Velems_per_dim0, dft_raw += Velems_per_dim0)
        {
          slot_loc = slot_locs[i];
          temp_reply = reply[slot_loc]->element[slots_iters_nums[slot_loc]];
          ++(slots_iters_nums[slot_loc]);
          if (temp_reply->type == 1) // #define REDIS_REPLY_STRING 1
          {
            reply_memcpy_to_tensor(pv_raw, temp_reply->str, Velems_per_dim0); // Direct access to Tensor data in TensorFlow
          }
          else
          {
            if (is_full_default)
            {
              default_memcpy_to_tensor(pv_raw, dft_raw, Velems_per_dim0); // Direct access to Tensor data in TensorFlow
            }
            else
            {
              default_memcpy_to_tensor(pv_raw, dft_raw_begin, Velems_per_dim0); // Direct access to Tensor data in TensorFlow
            }
          }
        }
      }

      virtual void MSET_COMMAND(
          const ::tensorflow::Tensor &keys, const ::tensorflow::Tensor &values,
          ThreadContext &thread_context,
          const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
          const std::vector<std::string> &keys_prefix_name_slices) override
      {
        const int &&total = max_i - begin;
        const int &&argc = total * 2 + 2;

        const static char *redis_command = "HMSET";
        const static std::size_t &&redis_command_byte = 5;

        const K *const pk_raw_end = reinterpret_cast<K *>(keys.data()) + (total);
        // const char *const pk_end = reinterpret_cast<const char *>(pk_raw_end);
        const K *pk_raw = reinterpret_cast<K *>(keys.data()) + begin;
        // const char *pk = reinterpret_cast<const char *>(pk_raw);

        const ::tensorflow::int64 &&Vdim0_size = values.dim_size(0);
        const ::tensorflow::int64 &&Velems_per_dim0 = values.NumElements() / Vdim0_size;
        const std::size_t &&V_byte_size = values.NumElements() / Vdim0_size * sizeof(V);
        // const V *const pv_raw_end = reinterpret_cast<V*>(values.data()) + (total)*Velems_per_dim0;
        // const char *const pv_end = reinterpret_cast<const char *>(pv_raw_end);
        const V *pv_raw = reinterpret_cast<V *>(values.data()) + begin * Velems_per_dim0;
        // const char *pv = reinterpret_cast<const char *>(pv_raw);

        const unsigned &storage_slice = redis_connection_params.storage_slice;
        const unsigned &&vector_len = (static_cast<::tensorflow::int64>(reinterpret_cast<int>(argc)) >> redis_connection_params.storage_slice_log2) + 2;

        thread_context.HandleReserve(storage_slice, vector_len, total);

        for (unsigned i = 0; i < storage_slice; ++i)
        {
          thread_context.HandlePushBack(i, redis_command, redis_command_byte);
          thread_context.HandlePushBack(i, keys_prefix_name_slices[i].data(), keys_prefix_name_slices[i].size());
        }

        VContentAndTypeSizeResult VCATS_temp;
        std::vector<char> buff_temp;
        unsigned key_slot_locs = 0;
        for (; pk_raw != pk_raw_end; ++pk_raw, pv_raw += Velems_per_dim0)
        {
          VCATS_temp = VContentAndTypeSize<V>(VCATS_temp, Velems_per_dim0, V_byte_size, pv_raw, buff_temp);
          key_slot_locs = KSlotNum<K>(pk_raw, storage_slice); // TODO: change it to AVX512

          // Direct access to ::tensorflow::Tensor data in TensorFlow
          thread_context.HandlePushBack(key_slot_locs, KContentPointer<K>(pk_raw), KTypeSize<K>(pk_raw));
          thread_context.HandlePushBack(key_slot_locs, VCATS_temp.VContentPointer, VCATS_temp.VTypeSize);
        }

        auto cmd = [](::sw::redis::Connection &connection, const ::sw::redis::StringView &hkey,
                      const std::vector<const char *> &ptrs_i, const std::vector<std::size_t> &sizes_i)
        {
          assert(strcmp(ptrs_i[0], "HMSET") == 0);
          assert(sizes_i[0] == 5);
          assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);
          // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
          connection.send(static_cast<int>(ptrs_i.size()), const_cast<const char **>(ptrs_i.data()), sizes_i.data());
        };

        pipe_exec(cmd, storage_slice, 4U, thread_context);
      }

      virtual void DEL_COMMAND(
          const ::tensorflow::Tensor &keys, ThreadContext &thread_context,
          const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
          const std::vector<std::string> &keys_prefix_name_slices) override
      {
        const int &&total = max_i - begin;
        const int &&argc = total + 2;

        const static char *redis_command = "HDEL";
        const static std::size_t &&redis_command_byte = 4;

        const K *const pk_raw_end = reinterpret_cast<K *>(keys.data()) + (total);
        const K *pk_raw = reinterpret_cast<K *>(keys.data()) + begin;

        const unsigned &storage_slice = redis_connection_params.storage_slice;
        const unsigned &&vector_len = (static_cast<::tensorflow::int64>(reinterpret_cast<int>(argc)) >> redis_connection_params.storage_slice_log2) + 2;

        thread_context.HandleReserve(storage_slice, vector_len, total);

        for (unsigned i = 0; i < storage_slice; ++i)
        {
          thread_context.HandlePushBack(i, redis_command, redis_command_byte);
          thread_context.HandlePushBack(i, keys_prefix_name_slices[i].data(), keys_prefix_name_slices[i].size());
        }

        unsigned *pslot_loc = thread_context.slot_locs.data();
        unsigned key_slot_locs = 0;
        for (; pk_raw != pk_raw_end; ++pk_raw)
        {
          key_slot_locs = KSlotNum<K>(pk_raw, storage_slice);
          // The slot to which the key belongs is recorded to facilitate future memory writes that do not recompute the hash
          *pslot_loc = key_slot_locs;
          ++pslot_loc;

          // Direct access to ::tensorflow::Tensor data in TensorFlow
          thread_context.HandlePushBack(key_slot_locs, KContentPointer<K>(pk_raw), KTypeSize<K>(pk_raw));
        }

        auto cmd = [](::sw::redis::Connection &connection, const ::sw::redis::StringView hkey,
                      const std::vector<const char *> &ptrs_i, const std::vector<std::size_t> &sizes_i)
        {
          assert(strcmp(ptrs_i[0], "HDEL") == 0);
          assert(sizes_i[0] == 4);
          assert(std::string(hkey.data()).compare(ptrs_i[1]) == 0);
          // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
          connection.send(static_cast<int>(ptrs_i.size()), const_cast<const char **>(ptrs_i.data()), sizes_i.data());
        };

        pipe_exec(cmd, storage_slice, 3U, thread_context);
      }
    };
  } // namespace redis_lookup
} // namespace sw::redis
