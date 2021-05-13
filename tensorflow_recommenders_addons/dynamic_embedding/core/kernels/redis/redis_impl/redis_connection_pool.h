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
#include <chrono>
#include <iostream>

#include <sw/redis++/redis++.h>
#include <sw/redis++/sentinel.h>
#include <sw/redis++/connection.h>
#include <sw/redis++/connection_pool.h>

namespace sw::redis
{
    namespace redis_lookup
    {
        enum Connection_Mode
        {
            ClusterMode = 0,
            SentinelMode = 1,
            StreamMode = 2
        };

        struct Redis_Connection_Params
        {
            int connection_mode = 1;
            std::string master_name = "master";
            // connection_options
            std::string host_name = "127.0.0.1";
            int host_port = 26379;
            std::string password = "";
            int db = 0;
            int connect_timeout = 100; // milliseconds
            int socket_timeout = 100;  // milliseconds
            // connection_pool_options
            int size = 10;
            int wait_timeout = 100;       // milliseconds
            int connection_lifetime = 10; // minutes
            // sentinel_connection_options
            int sentinel_connect_timeout = 200; // milliseconds
            int sentinel_socket_timeout = 200;  // milliseconds
        };

        template <typename RedisInstance>
        class RedisWrapper
        {
        private:
            bool isRedisConnect = false;
            Redis_Connection_Params redis_connection_params;
            SentinelOptions sentinel_opts;
            ConnectionOptions conn_opts;
            ConnectionPoolOptions pool_opts;

        public:
            static std::shared_ptr<RedisInstance> redis_conn;

        private:
            static std::shared_ptr<RedisWrapper<RedisInstance>> instance_ptr; //for the hungry singleton mode
            RedisWrapper();                                                   // In singleton mode, classes should not be initialized through constructor

        public:
            ~RedisWrapper();

            std::shared_ptr<RedisInstance> start_conn();
            auto conn();

            RedisWrapper(const RedisWrapper &) = delete;
            RedisWrapper &operator=(const RedisWrapper &) = delete;

            //for singleton mode
            static std::shared_ptr<RedisWrapper<RedisInstance>> get_instance(struct Redis_Connection_Params &conn_params_input);
        };
    } // namespace redis_lookup
} // namespace sw::redis