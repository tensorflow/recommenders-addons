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
	namespace redis_connection
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
			int socket_timeout = 100;	 // milliseconds
			// connection_pool_options
			int size = 10;
			int wait_timeout = 100;				// milliseconds
			int connection_lifetime = 10; // minutes
			// sentinel_connection_options
			int sentinel_connect_timeout = 200; // milliseconds
			int sentinel_socket_timeout = 200;	// milliseconds

			Redis_Connection_Params &operator=(const Redis_Connection_Params &x)
			{
				connection_mode = x.connection_mode;
				master_name = x.master_name;
				host_name = x.host_name;
				host_port = x.host_port;
				password = x.password;
				db = x.db;
				connect_timeout = x.connect_timeout; // milliseconds
				socket_timeout = x.socket_timeout;	 // milliseconds
				size = x.size;
				wait_timeout = x.wait_timeout;												 // milliseconds
				connection_lifetime = x.connection_lifetime;					 // minutes
				sentinel_connect_timeout = x.sentinel_connect_timeout; // milliseconds
				sentinel_socket_timeout = x.sentinel_socket_timeout;	 // milliseconds
				return *this;
			}
		};

		template <typename RedisInstance, typename = void>
		class RedisWrapper
		{
		};

		template <typename RedisInstance>
		class RedisWrapper<RedisInstance, typename std::enable_if<std::is_same<RedisInstance, RedisCluster>::value>::type>
		{
		private:
			bool isRedisConnect = false;
			Redis_Connection_Params redis_connection_params;
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
				conn_opts.host = redis_connection_params.host_name;
				conn_opts.port = redis_connection_params.host_port;
				// Redis connection options
				conn_opts.password = redis_connection_params.password; // Optional. No password by default.
				conn_opts.db = redis_connection_params.db;
				conn_opts.connect_timeout = std::chrono::milliseconds(redis_connection_params.connect_timeout);
				conn_opts.socket_timeout = std::chrono::milliseconds(redis_connection_params.socket_timeout);
				// Redis connection pool options
				pool_opts.size = redis_connection_params.size;
				pool_opts.wait_timeout = std::chrono::milliseconds(redis_connection_params.wait_timeout);
				pool_opts.connection_lifetime = std::chrono::minutes(redis_connection_params.connection_lifetime);

				static std::shared_ptr<RedisInstance> redis_client = nullptr;
				try
				{
					redis_client.reset(new RedisInstance(conn_opts, pool_opts));
				}
				catch (const ReplyError &err)
				{
					printf("RedisHandler-- ReplyError：%s \n", err.what());
					return redis_client;
				}
				catch (const TimeoutError &err)
				{
					printf("RedisHandler-- TimeoutError%s \n", err.what());
					return redis_client;
				}
				catch (const ClosedError &err)
				{
					printf("RedisHandler-- ClosedError%s \n", err.what());
					return redis_client;
				}
				catch (const IoError &err)
				{
					printf("RedisHandler-- IoError%s \n", err.what());
					return redis_client;
				}
				catch (const Error &err)
				{
					printf("RedisHandler-- other%s \n", err.what());
					return redis_client;
				}
				return redis_client;
			}

			std::shared_ptr<RedisInstance> &conn()
			{
				if (isRedisConnect == false)
				{
					for (short i = 0; i < 10; i++)
					{
						redis_conn = start_conn();
						if (redis_conn)
						{
							isRedisConnect = true;
							return redis_conn;
						}
					}
					std::cerr << "Can not connect to the Redis servers." << std::endl;
					throw(std::runtime_error("Exit without any Redis connection."));
				}
				return redis_conn;
			}

			static std::shared_ptr<RedisWrapper<RedisInstance>> get_instance()
			{
				static std::shared_ptr<RedisWrapper<RedisInstance>> instance_ptr = std::make_shared<RedisWrapper<RedisInstance>>(); // for the Meyer's Singleton mode
				return instance_ptr;
			}

			void set_params(struct Redis_Connection_Params &conn_params_input)
			{
				redis_connection_params = conn_params_input;
			}
		};

		template <typename RedisInstance>
		class RedisWrapper<RedisInstance, typename std::enable_if<std::is_same<RedisInstance, Redis>::value>::type>
		{
		private:
			bool isRedisConnect = false;
			Redis_Connection_Params redis_connection_params;
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
				sentinel_opts.nodes = {{redis_connection_params.host_name, redis_connection_params.host_port}};
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
				pool_opts.size = redis_connection_params.size;
				pool_opts.wait_timeout = std::chrono::milliseconds(redis_connection_params.wait_timeout);
				pool_opts.connection_lifetime = std::chrono::minutes(redis_connection_params.connection_lifetime);

				auto sentinel = std::make_shared<Sentinel>(sentinel_opts);

				static std::shared_ptr<RedisInstance> redis_client = nullptr;
				try
				{
					redis_client.reset(new RedisInstance(sentinel, redis_connection_params.master_name, Role::MASTER, conn_opts, pool_opts));
				}
				catch (const ReplyError &err)
				{
					printf("RedisHandler-- ReplyError：%s \n", err.what());
					return redis_client;
				}
				catch (const TimeoutError &err)
				{
					printf("RedisHandler-- TimeoutError%s \n", err.what());
					return redis_client;
				}
				catch (const ClosedError &err)
				{
					printf("RedisHandler-- ClosedError%s \n", err.what());
					return redis_client;
				}
				catch (const IoError &err)
				{
					printf("RedisHandler-- IoError%s \n", err.what());
					return redis_client;
				}
				catch (const Error &err)
				{
					printf("RedisHandler-- other%s \n", err.what());
					return redis_client;
				}
				return redis_client;
			}

			std::shared_ptr<RedisInstance> &conn()
			{
				if (isRedisConnect == false)
				{
					for (short i = 0; i < 10; i++)
					{
						redis_conn = start_conn();
						if (redis_conn)
						{
							isRedisConnect = true;
							return redis_conn;
						}
					}
					std::cerr << "Can not connect to the Redis servers." << std::endl;
					throw(std::runtime_error("Exit without any Redis connection."));
				}
				return redis_conn;
			}

			static std::shared_ptr<RedisWrapper<RedisInstance>> get_instance()
			{
				static std::shared_ptr<RedisWrapper<RedisInstance>> instance_ptr = std::make_shared<RedisWrapper<RedisInstance>>(); // for the Meyer's Singleton mode
				return instance_ptr;
			}

			void set_params(struct Redis_Connection_Params &conn_params_input)
			{
				redis_connection_params = conn_params_input;
			}
		};

	} // namespace redis_lookup
} // namespace sw::redis
