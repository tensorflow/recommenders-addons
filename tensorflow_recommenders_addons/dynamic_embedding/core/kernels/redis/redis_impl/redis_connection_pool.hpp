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

#include <nmmintrin.h>

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
    class RedisWrapper<RedisInstance, K, V, typename std::enable_if<std::is_same<RedisInstance, Redis>::value>::type>:public RedisVirtualWrapper
    {
    private:
      SentinelOptions sentinel_opts;
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
        std::cout << "RedisSentinel connection pool constructor called!" << std::endl;
      }

      ~RedisWrapper()
      {
        if (redis_conn == nullptr)
        {
          return;
        }
        redis_conn.reset();
        std::cout << "RedisSentinel connection pool destructor called!" << std::endl;
      }

    public:
      std::shared_ptr<RedisInstance> start_conn()
      {
        sentinel_opts.nodes = {{redis_connection_params.host_ip, redis_connection_params.host_port}};
        // Optional. Timeout before we successfully connect to Redis Sentinel.
        sentinel_opts.connect_timeout = std::chrono::milliseconds(redis_connection_params.sentinel_connect_timeout);
        // Optional. Timeout before we successfully send request to or receive response from Redis Sentinel.
        sentinel_opts.socket_timeout = std::chrono::milliseconds(redis_connection_params.sentinel_socket_timeout);

        // Redis connection options
        conn_opts.password = redis_connection_params.password; // Optional. No password by default.
        conn_opts.db = redis_connection_params.db;
        conn_opts.connect_timeout = std::chrono::milliseconds(redis_connection_params.connect_timeout);
        conn_opts.socket_timeout = std::chrono::milliseconds(redis_connection_params.socket_timeout);
        // Redis connection pool options
        pool_opts.size = redis_connection_params.pool_size;
        pool_opts.wait_timeout = std::chrono::milliseconds(redis_connection_params.wait_timeout);
        pool_opts.connection_lifetime = std::chrono::minutes(redis_connection_params.connection_lifetime);

        auto sentinel = std::make_shared<Sentinel>(sentinel_opts);

        try
        {
          static auto redis_client = std::make_shared<RedisInstance>(
            RedisInstance(sentinel, redis_connection_params.master_name, Role::MASTER, conn_opts, pool_opts));
          redis_client->ping();
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
            std::cerr << "Can not connect to the Redis Master servers." << std::endl;
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
    
    public:

      virtual bool check_slices_num(const std::string &keys_prefix_name) override
      {
        std::string redis_command = "keys " + '*' + keys_prefix_name+ '*';
        auto cmd = [](::sw::redis::Connection &connection, char *str) {connection.send(str);};
        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply = redis_conn->command(cmd, redis_command.data());
        if (reply->elements != redis_connection_params.storage_slice)
        {
          std::cerr << "storage_slice in redis_connection_params did not equal to the slices number of this keys_prefix_name in the Redis server" <<std::endl;
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
        std::string redis_command = "hlen " + keys_prefix_name_slices[0];
        auto cmd = [](::sw::redis::Connection &connection, char *str) {connection.send(str);};
        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply = redis_conn->command(cmd, redis_command.data());
        size_t size = strtoumax(reply->element[0]->str, nullptr, 10); // decimal
        
        return size;
      }

      virtual void remove_hkeys_in_slots(const std::vector<std::string> &keys_prefix_name_slices) override
      {
        // std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply;
        std::string redis_command = "del " + keys_prefix_name_slices[0];
        auto cmd = [](::sw::redis::Connection &connection, char *str) {connection.send(str);};
        /*reply=*/redis_conn->command(cmd, redis_command.data());
      }
      
      /*
      fds are the return of POSIX open file function declared in <fcntl.h> 
      */
      virtual void dump_to_disk(const std::vector<std::string> &keys_prefix_name_slices, std::vector<aiocb> &wrs, const std::vector<int> &fds) override
      {
        std::string redis_command = "dump " + keys_prefix_name_slices[0];
        aiocb *wr = &wrs.front();
        int ret; // int fd;

        auto cmd = [](::sw::redis::Connection &connection, char *str) {connection.send(str);};
        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply = redis_conn->command(cmd, redis_command.data());

        size_t buf_len; 
        // std::string file_name = redis_connection_params.model_lib_abs_dir+keys_prefix_name_slices[0]+".rdb";
        // fd = open(file_name,O_WRONLY | O_APPEND);
        // if(fd < 0) perror(file_name);
        if (wr->aio_nbytes > 0)
        {
          for (size_t i = 3; i > 0; --i)
          {
            while (aio_error(wr) == EINPROGRESS);
            if ((ret = aio_return(wr)) > 0) 
            {
              std::cout << "File handle " << wr->aio_fildes << " finished writing last round." << std::endl;
              break;
            } 
            else 
            {
              std::cerr << "File handle " << wr->aio_fildes << " did not finish writing last round. " << \
                          "Try to write " << i << " more times" << std::endl;
              ret = aio_write(wr);
              if(ret < 0) perror("aio_write");
            }
          }
        }
        free(wr->aio_buf); // Dangerous behavior! Note that when creating AIOCB objects, you need to set aio_buf to nullptr!
        wr->aio_buf = nullptr;
        bzero(wr, sizeof(*wr));
        buf_len = reply->element[0]->len;
        wr->aio_buf = malloc(buf_len);
        memcpy(wr->aio_buf, reply->element[0]->str, buf_len);
        wr->aio_nbytes = buf_len;
        wr->aio_fildes = fds[0];
        wr->aio_offset = 0;
        ret = aio_write(wr);
        if(ret < 0) perror("aio_write");
      }

      virtual void restore_from_disk(const std::vector<std::string> &keys_prefix_name_slices, std::vector<aiocb> &rds, 
        const std::vector<int> &fds, const std::vector<unsigned long> &buf_sizes) override
      {
        std::string redis_command;
        std::string tmp_redis_command = "restore " + keys_prefix_name_slices[0] + " 0";
        size_t command_capacity = keys_prefix_name_slices[0].size() + 19; // "restore "=8, '0'=1, reset for enough mem space.
        aiocb *rd = &rds.front();
        int ret; // int fd;

        auto cmd = [](::sw::redis::Connection &connection, char *str) {connection.send(str);};

        size_t buf_len; 
        buf_len = buf_sizes[0];

        bzero(rd, sizeof(*rd));

        size_t &&tmp_redis_command_size = tmp_redis_command.size();
        redis_command.reserve(command_capacity + buf_len + 1); 
        redis_command.replace(0, tmp_redis_command_size, tmp_redis_command);
        redis_command.replace(tmp_redis_command_size, command_capacity-tmp_redis_command_size, command_capacity-tmp_redis_command_size, ' ');

        rd->aio_buf = &redis_command[command_capacity];
        rd->aio_nbytes = buf_len;
        rd->aio_fildes = fds[0];
        rd->aio_offset = 0;
        ret = aio_read(rd);
        if(ret < 0) perror("aio_read");

        if (rd->aio_nbytes > 0)
        {
          for (size_t i = 3; i > 0; --i)
          {
            while (aio_error(rd) == EINPROGRESS);
            if ((ret = aio_return(rd)) > 0) 
            {
              std::cout << "File handle " << rd->aio_fildes << " finished reading last round." << std::endl;
              break;
            } 
            else 
            {
              std::cerr << "File handle " << rd->aio_fildes << " did not finish reading last round. " << \
                          "Try to rdite " << i << " more times" << std::endl;
              ret = aio_read(rd);
              if(ret < 0) perror("aio_read");
            }
          }
        }

        std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter> reply = redis_conn->command(cmd, redis_command.data());
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
        const std::vector<std::string> &keys_prefix_name_slices
      ) override
      {
        const int argc = (max_i - begin) + 2; 

        const static char *redis_command = "hmget";
        const static std::size_t redis_command_byte = 5; 

        thread_context.HandleReserve(1, argc, 0);

        std::vector<const char *> &ptrs_0 = thread_context.slots[0].ptrs;
        std::vector<std::size_t> &sizes_0 = thread_context.slots[0].sizes;

        // const ::tensorflow::int64 dim0_size = keys.dim_size(0);
        // const ::tensorflow::int64 elems_per_dim0 = keys.NumElements() / dim0_size;
        // const std::size_t key_byte_size = elems_per_dim0 * sizeof(K);
        const K *const pk_raw_end = reinterpret_cast<K*>(keys.data()) + (max_i-begin);
        // const char *const pk_end = reinterpret_cast<const char *>(pk_raw_end);
        const K *pk_raw = reinterpret_cast<K*>(keys.data()) + begin;
        // const char *pk = reinterpret_cast<const char *>(pk_raw);

        const char **ptrs_iter=&ptrs_0[0];
        *ptrs_iter = redis_command;
        ++ptrs_iter;
        *ptrs_iter = keys_prefix_name_slices[0].data();
        ++ptrs_iter;

        std::size_t *sizes_iter=&sizes_0[0];
        *sizes_iter = redis_command_byte;
        ++sizes_iter;
        *sizes_iter = keys_prefix_name_slices[0].size();
        ++sizes_iter;

        for (; pk_raw != pk_raw_end; ++pk_raw )
        { 
          *ptrs_iter = KContentPointer<K>(pk_raw); // Direct access to ::tensorflow::Tensor data in TensorFlow
          ++ptrs_iter;
          *sizes_iter = KTypeSize<K>(pk_raw); // key data char size
          ++sizes_iter;
        }

        assert(ptrs_0[0]==redis_command);
        assert(sizes_0[0]==redis_command_byte);

        auto cmd = [](::sw::redis::Connection &connection, const int argc,
                      std::vector<const char *> &ptrs_0, std::vector<std::size_t> &sizes_0)
        {
          // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */ 
          connection.send(argc, &ptrs_0[0], &sizes_0[0]);
        };

        std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> reply;
        reply.reserve(1);
        reply.push_back(redis_conn->command(cmd, argc, ptrs_0, sizes_0));

        return reply;
      }

      virtual void MGET_to_Tensor(
        ::tensorflow::Tensor *values, const ::tensorflow::Tensor &default_value, const bool &is_full_default,
        ThreadContext &thread_context,std::vector<std::unique_ptr<redisReply, ::sw::redis::ReplyDeleter>> &reply,
        const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i
      )
      {
        const ::tensorflow::int64 &&Velems_per_dim0 = values->NumElements() / values->dim_size(0);
        V *pv_raw = reinterpret_cast<V*>(values->data()) + begin*Velems_per_dim0;
        // char *pv = reinterpret_cast<const char *>(pv_raw);
        const V *dft_raw = reinterpret_cast<const V*>(default_value.data()) + begin*Velems_per_dim0;
        const V *const dft_raw_begin = reinterpret_cast<const V*>(default_value.data());

        redisReply *temp_reply;
        for (auto i = begin; i < max_i; ++i, pv_raw+=Velems_per_dim0, dft_raw+=Velems_per_dim0)
        {
          temp_reply = reply[0]->element[i];
          if (temp_reply->type == 1) // #define REDIS_REPLY_STRING 1
          {
            reply_memcpy_to_tensor<V>(pv_raw, temp_reply->str, Velems_per_dim0); // Direct access to Tensor data in TensorFlow
          }
          else
          {
            if (is_full_default)
            {
              default_memcpy_to_tensor<V>(pv_raw, dft_raw, Velems_per_dim0); // Direct access to Tensor data in TensorFlow
            }
            else
            {
              default_memcpy_to_tensor<V>(pv_raw, dft_raw_begin, Velems_per_dim0); // Direct access to Tensor data in TensorFlow
            }
          }
        }
      }

      virtual void MSET_COMMAND(
        const ::tensorflow::Tensor &keys, const ::tensorflow::Tensor &values,
        ThreadContext &thread_context,
        const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
        const std::vector<std::string> &keys_prefix_name_slices
      ) override
      {
        const int &&argc = (max_i - begin)*2 + 2;

        const static char *redis_command = "hmset";
        const static std::size_t redis_command_byte = 5;

        thread_context.HandleReserve(1, argc, 0);

        std::vector<const char *> &ptrs_0 = thread_context.slots[0].ptrs;
        std::vector<std::size_t> &sizes_0 = thread_context.slots[0].sizes;

        const K *const pk_raw_end = reinterpret_cast<K*>(keys.data()) + (max_i-begin);
        // const char *const pk_end = reinterpret_cast<const char *>(pk_raw_end);
        const K *pk_raw = reinterpret_cast<K*>(keys.data()) + begin;
        // const char *pk = reinterpret_cast<const char *>(pk_raw);

        const ::tensorflow::int64 &&Vdim0_size = values.dim_size(0);
        const ::tensorflow::int64 &&Velems_per_dim0 = values.NumElements() / Vdim0_size;
        const std::size_t &&V_byte_size = values.NumElements() / Vdim0_size * sizeof(V);
        // const V *const pv_raw_end = reinterpret_cast<V*>(values.data()) + (max_i-begin)*Velems_per_dim0;
        // const char *const pv_end = reinterpret_cast<const char *>(pv_raw_end);
        const V *pv_raw = reinterpret_cast<V*>(values.data()) + begin*Velems_per_dim0;
        // const char *pv = reinterpret_cast<const char *>(pv_raw);

        const char **ptrs_iter=&ptrs_0[0];
        *ptrs_iter = redis_command;
        ++ptrs_iter;
        *ptrs_iter = keys_prefix_name_slices[0].data();
        ++ptrs_iter;

        std::size_t *sizes_iter=&sizes_0[0];
        *sizes_iter = redis_command_byte;
        ++sizes_iter;
        *sizes_iter = keys_prefix_name_slices[0].size();
        ++sizes_iter;

        VContentAndTypeSizeResult VCATS_temp;
        std::vector<char> buff_temp;
        
        for (; pk_raw != pk_raw_end; ++pk_raw,pv_raw+=Velems_per_dim0 )
        { 
          VCATS_temp = VContentAndTypeSize<V>(VCATS_temp, Velems_per_dim0, V_byte_size, pv_raw, buff_temp);

          *ptrs_iter = KContentPointer<K>(pk_raw); // Direct access to ::tensorflow::Tensor data in TensorFlow
          *(++ptrs_iter) = VCATS_temp.VContentPointer;
          ++ptrs_iter;

          *sizes_iter = KTypeSize<K>(pk_raw); // key data char size
          *(++sizes_iter) = VCATS_temp.VTypeSize;
          ++sizes_iter;      
        }

        assert(ptrs_0[0]==redis_command);
        assert(sizes_0[0]==redis_command_byte);

        auto cmd = [](::sw::redis::Connection &connection, const int argc,
                      std::vector<const char *> &ptrs_0, std::vector<std::size_t> &sizes_0)
        {   
          // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */
          connection.send(argc, &ptrs_0[0], &sizes_0[0]);
        };

        redis_conn->command(cmd, argc, ptrs_0, sizes_0);

      }

      virtual void DEL_COMMAND(
        const ::tensorflow::Tensor &keys, ThreadContext &thread_context,
        const ::tensorflow::int64 &begin, const ::tensorflow::int64 &max_i,
        const std::vector<std::string> &keys_prefix_name_slices
      ) override
      {
        const int argc = (max_i - begin) + 2; 

        const static char *redis_command = "hdel";
        const static std::size_t redis_command_byte = 4; 

        thread_context.HandleReserve(1, argc, 0);

        std::vector<const char *> &ptrs_0 = thread_context.slots[0].ptrs;
        std::vector<std::size_t> &sizes_0 = thread_context.slots[0].sizes;

        const K *const pk_raw_end = reinterpret_cast<K*>(keys.data()) + (max_i-begin);
        const K *pk_raw = reinterpret_cast<K*>(keys.data()) + begin;

        const char **ptrs_iter=&ptrs_0[0];
        *ptrs_iter = redis_command;
        ++ptrs_iter;
        *ptrs_iter = keys_prefix_name_slices[0].data();
        ++ptrs_iter;

        std::size_t *sizes_iter=&sizes_0[0];
        *sizes_iter = redis_command_byte;
        ++sizes_iter;
        *sizes_iter = keys_prefix_name_slices[0].size();
        ++sizes_iter;

        for (; pk_raw != pk_raw_end; ++pk_raw )
        { 
          *ptrs_iter = KContentPointer<K>(pk_raw); // Direct access to ::tensorflow::Tensor data in TensorFlow
          ++ptrs_iter;
          *sizes_iter = KTypeSize<K>(pk_raw); // key data char size
          ++sizes_iter;
        }

        assert(ptrs_0[0]==redis_command);
        assert(sizes_0[0]==redis_command_byte);

        auto cmd = [](::sw::redis::Connection &connection, const int argc,
                      std::vector<const char *> &ptrs_0, std::vector<std::size_t> &sizes_0)
        {
          // raise(SIGTRAP);  /* To continue from here in GDB: "signal 0". */ 
          connection.send(argc, &ptrs_0[0], &sizes_0[0]);
        };

        /*auto reply=*/redis_conn->command(cmd, argc, ptrs_0, sizes_0);
      }

    };

  } // namespace redis_lookup
} // namespace sw::redis
