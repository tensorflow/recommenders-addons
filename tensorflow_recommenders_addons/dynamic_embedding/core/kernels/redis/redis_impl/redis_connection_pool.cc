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

#include "redis_connection_pool.h"

namespace sw::redis
{
  namespace redis_connection
  {
    //for singleton mode
    template <typename RedisInstance>
    std::shared_ptr<RedisWrapper<RedisInstance>> RedisWrapper<RedisInstance>::instance_ptr(new RedisWrapper<RedisInstance>());

    template <typename RedisInstance>
    std::shared_ptr<RedisWrapper<RedisInstance>> RedisWrapper<RedisInstance>::get_instance(struct Redis_Connection_Params &conn_params_input)
    {
      return instance_ptr;
    }

    template <typename RedisInstance>
    RedisWrapper<RedisInstance>::RedisWrapper() // In singleton mode, classes should not be initialized through constructor
    {
      std::cout << "Redis connection pool constructor called!" << std::endl;
    }

    template <typename RedisInstance>
    RedisWrapper<RedisInstance>::~RedisWrapper()
    {
      if (instance_ptr == nullptr)
      {
        return;
      }
      instance_ptr.reset();
      std::cout << "Redis connection pool destructor called!" << std::endl;
    }

    template <typename RedisInstance>
    std::shared_ptr<RedisInstance> RedisWrapper<RedisInstance>::start_conn()
    {
      switch (redis_connection_params.connection_mode)
      {
      case ClusterMode:
      {
        conn_opts.host = redis_connection_params.host_name;
        conn_opts.port = redis_connection_params.host_port;
        break;
      }
      case SentinelMode:
      {
        sentinel_opts.nodes = {{redis_connection_params.host_name, redis_connection_params.host_port}};
        // Optional. Timeout before we successfully connect to Redis Sentinel.
        sentinel_opts.connect_timeout = std::chrono::milliseconds(redis_connection_params.sentinel_connect_timeout);
        // Optional. Timeout before we successfully send request to or receive response from Redis Sentinel.
        sentinel_opts.socket_timeout = std::chrono::milliseconds(redis_connection_params.sentinel_socket_timeout);
        break;
      }
      case StreamMode:
      {
        std::cout << StreamMode << std::endl;
        std::cerr << "Sorry! connection_mode=" << redis_connection_params.connection_mode << " The Stream connection mode is still being TODO." << std::endl;
        throw(redis_connection_params.connection_mode);
        break;
      }
      default:
      {
        std::cerr << "There are only three Redis connection modes, which Cluster=0/Sentinel=1/Stream=2." << std::endl;
        throw(redis_connection_params.connection_mode);
        break;
      }
      }
      // Redis connection options
      conn_opts.password = redis_connection_params.password; // Optional. No password by default.
      conn_opts.db = redis_connection_params.db;
      conn_opts.connect_timeout = std::chrono::milliseconds(redis_connection_params.connect_timeout);
      conn_opts.socket_timeout = std::chrono::milliseconds(redis_connection_params.socket_timeout);
      // Redis connection pool options
      pool_opts.size = redis_connection_params.size;
      pool_opts.wait_timeout = std::chrono::milliseconds(redis_connection_params.wait_timeout);
      pool_opts.connection_lifetime = std::chrono::minutes(redis_connection_params.connection_lifetime);

      switch (redis_connection_params.connection_mode)
      {
      case ClusterMode:
      {
        static std::shared_ptr<RedisInstance> redis_client = nullptr;
        try
        {
          redis_client = std::make_shared<RedisInstance>(RedisInstance(conn_opts, pool_opts));
        }
        catch (const ReplyError &err)
        {
          printf("RedisHandler-- ReplyError：%s \n", err.what());
          return nullptr;
        }
        catch (const TimeoutError &err)
        {
          printf("RedisHandler-- TimeoutError%s \n", err.what());
          return nullptr;
        }
        catch (const ClosedError &err)
        {
          printf("RedisHandler-- ClosedError%s \n", err.what());
          return nullptr;
        }
        catch (const IoError &err)
        {
          printf("RedisHandler-- IoError%s \n", err.what());
          return nullptr;
        }
        catch (const Error &err)
        {
          printf("RedisHandler-- other%s \n", err.what());
          return nullptr;
        }
        return redis_client;
      }
      case SentinelMode:
      {
        auto sentinel = std::make_shared<Sentinel>(sentinel_opts);
        static std::shared_ptr<RedisInstance> redis_client = nullptr;
        try
        {
          redis_client = std::make_shared<RedisInstance>(RedisInstance(sentinel, redis_connection_params.master_name, Role::MASTER, conn_opts, pool_opts));
        }
        catch (const ReplyError &err)
        {
          printf("RedisHandler-- ReplyError：%s \n", err.what());
          return nullptr;
        }
        catch (const TimeoutError &err)
        {
          printf("RedisHandler-- TimeoutError%s \n", err.what());
          return nullptr;
        }
        catch (const ClosedError &err)
        {
          printf("RedisHandler-- ClosedError%s \n", err.what());
          return nullptr;
        }
        catch (const IoError &err)
        {
          printf("RedisHandler-- IoError%s \n", err.what());
          return nullptr;
        }
        catch (const Error &err)
        {
          printf("RedisHandler-- other%s \n", err.what());
          return nullptr;
        }
        return redis_client;
      }
      case StreamMode:
      {
        std::cerr << "Sorry! connection_mode=" << redis_connection_params.connection_mode << " The Stream connection mode is still being TODO." << std::endl;
        throw(redis_connection_params.connection_mode);
        break;
      }
      default:
      {
        std::cerr << "There are only three Redis connection modes, which Cluster=0/Sentinel=1/Stream=2." << std::endl;
        throw(redis_connection_params.connection_mode);
        break;
      }
      }
    }

    template <typename RedisInstance>
    std::shared_ptr<RedisInstance> RedisWrapper<RedisInstance>::conn()
    {
      if (isRedisConnect == false)
      {
        switch (redis_connection_params.connection_mode)
        {
        case ClusterMode:
        {
          for (short i = 0; i < 10; i++)
          {
            auto redis_conn = RedisWrapper<RedisInstance>::start_conn();
            if (redis_conn)
            {
              return redis_conn;
            }
          }
          std::cerr << "Can not connect to the Redis servers." << std::endl;
          throw(std::runtime_error("Exit without any Redis connection."));
        }
        case SentinelMode:
        {
          for (short i = 0; i < 10; i++)
          {
            auto redis_conn = RedisWrapper<RedisInstance>::start_conn();
            if (redis_conn)
            {
              return redis_conn;
            }
          }
          std::cerr << "Can not connect to the Redis servers." << std::endl;
          throw(std::runtime_error("Exit without any Redis connection."));
        }
        case StreamMode:
        {
          std::cerr << "Sorry! connection_mode=" << redis_connection_params.connection_mode << " The Stream connection mode is still being TODO." << std::endl;
          throw(redis_connection_params.connection_mode);
          break;
        }
        default:
        {
          std::cerr << "There are only three Redis connection modes, which Cluster=0/Sentinel=1/Stream=2." << std::endl;
          throw(redis_connection_params.connection_mode);
          break;
        }
        }

        isRedisConnect = true;
      }
      std::cerr << "There is no Redis client connection for return" << std::endl;
      exit(1);
    }
  } // namespace redis_lookup
} // namespace sw::redis
