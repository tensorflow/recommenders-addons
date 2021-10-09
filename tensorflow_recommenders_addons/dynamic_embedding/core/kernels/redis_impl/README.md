# Description

Addding Redis as the backend for KV storage and changed the Python interface by calling the Config class 
and creating/passing the KvCreator class to fit the different backend. Also setting backend parameters with json file.

The implementation of TFRA backend relies on the Redis-plus-plus and hiredis component. 
How to install Redis-plus-plus please visit the following page:
https://github.com/sewenew/redis-plus-plus

Redis-plus-plus is a Redis client, based on hiredis and written in C++11. 
It supports scritpting, pub/sub, pipeline, transaction, Redis Cluster, Redis Sentinel, connection pool, ACL, SSL and thread safety.

the method for parsing json file was created by James McLaughlin et al.  

For performance reasons, TFRA-Redis uses the method of exponentiating the remainder of 2 to calculate the bucket 
division of key in embedding table. **<u>So never use a key with an artificial ending,</u> for example 341445_1 and 341445_0. 
Put your custom edit to the key at the beginning, for example 1_341445 and 0_341445.**  

You can solve this problem by hashing the keys with more out-of-order results, such as SHA or MD5, rather than 
CRC32 / CRC16. But even with the addition of Intel's instruction set, it still affects performance too much.

# How To Create a Config File?
Below is an example of a JSON file, along with comments on the corresponding project. Please use UTF-8 encoding.
**Attention! Json files cannot contain comments when actually used!**
```json
{
      "redis_connection_mode": 1,  // ClusterMode = 0, SentinelMode = 1, StreamMode = 2
      "redis_master_name": "master",

      // connection_options
      "redis_host_ip": ["127.0.0.1"],
      "redis_host_port": [6379],
      "redis_user": "default",
      "redis_password": "",
      "redis_db": 0,
      "redis_read_access_slave": False, // set True in infer or train mode if you like
      "redis_connect_keep_alive": False,  // keep TCP alive
      "redis_connect_timeout": 1000,  // milliseconds
      "redis_socket_timeout": 1000,  // milliseconds

      // connection_pool_options
      "redis_conn_pool_size": 20,
      "redis_wait_timeout": 100000000,  // milliseconds
      "redis_connection_lifetime": 100,  // minutes

      // sentinel_connection_options
      "redis_sentinel_connect_timeout": 1000,  // milliseconds
      "redis_sentinel_socket_timeout": 1000,  // milliseconds

      // Below there is user-defined parameters in this custom op, not Redis setting parameters
      "storage_slice": 2,  // For deciding bucket number, which usually is how many Redis instance may be used in the trainning.
      "keys_sending_size": 1024,  // Determines how many keys to send at a time for performance tuning
      "using_md5_prefix_name": False,  // 1=true, 0=false
      "model_tag_import": "test",  // model_tag_import for version and any other information from last time.
      "redis_hash_tags_import": ["{6379}","{26379}"], // Deciding hash tag for every bucket from last time, Note that the hash tag must be wrapped in curly braces {}.
      "model_tag_runtime": "test",  // model_tag_runtime for version and any other information for now.
      "redis_hash_tags_runtime": ["{3560}","{120}"], // Deciding hash tag for every bucket for now, Note that the hash tag must be wrapped in curly braces {}.
      "expire_model_tag_in_seconds": 604800,  // To eliminate unwanted model versions in Redis to ensure sufficient storage space.
      "table_store_mode": 1,  // Saving and restoring table into ensor in TF savedmodel variable file, table_store_mode = 0; Saving and restoring table into redis rdb file in model_lib_abs_dir, table_store_mode = 1; Saving and restoring nothing, keeping data in redis servers, table_store_mode = 2.
      "model_lib_abs_dir": "/tmp/"  // if table_store_mode equals 1, then it will try to save or resoter table from model_lib_abs_dir which has been mounted in system
  }
```
If you creat a new model, then "model_tag_import" equals "model_tag_runtime". If you want to import the embedding table 
from a old model, then keep then "model_tag_import" as the old tag, and assign a new "model_tag_runtime".  
Also if you want to change the Redis hash tag for "storage_slice", you could assign "redis_hash_tags_import" and 
"redis_hash_tags_runtime" in the same way. Or you could only change "redis_hash_tags_runtime" without modifying "model_tag_runtime".

**Remember! the tag numbers in "redis_hash_tags_import" and "redis_hash_tags_runtime" arrays should equal to the "storage_slice"!**

**Attention! TFRA-Redis will run with "model_tag_runtime"!**

When TFRA-Redis is loaded for the first time, it checks whether the current "model_tag_runtime" bucket partition 
in Redis is the same as the "storage_slice" parameter. If not, The buckets of "model_tag_runtime" are re-bucketized 
to the number of buckets set by the "storage_slice" parameter. 

Generally, "storage_slice "should be equal to the number of nodes in the Redis cluster, but you can still change this parameter 
to any other number. Also there is a table inside the program to generate the Redis hash tag sequentially. So for a particular "storage_slice" parameter, 
the target Redis node and slot number is fixed. Of course you can set it with the "redis_hash_tags_runtime" parameter by yourself rather than generated by the program.
  
# How To Use in TensorFlow?
By default, TFRA-Redis reads the JSON file pointed to by the path in the OP attribute redis_config_abs_dir_env, which is an environment variable.
If the environment variable described in redis_config_ABS_dir_env does not exist or the path to which the value of the environment variable points does not have a corresponding JSON file, 
the next step is to look for the path of the JSON file pointed to in the TFRA_REDIS_CONFIG_PATH environment variable.The last step will read the json file path configured in the Python operator attribute redis_config_abs_dir.  

So when you deploy inference, you can set the environment variable TFRA_REDIS_CONFIG_PATH to configure the redis service of all model on the inference side. For example, **"$export TFRA_REDIS_CONFIG_PATH=/tem/redis.config"**.  

Or to specify a particular Redis service for a particular model in a container, you can agree on a common environment variable redis_config_abs_dir_env, both online and offline, from which to read the path to the config file.
For example, set **"redis_config_abs_dir_env="model1_tfra_redis_config_path"**, and then   
**"$export model1_tfra_redis_config_path=/tem/redis_offline.config"** in the offline,  
**"$export model1_tfra_redis_config_path=/tem/redis_online.config"** in the online

### priority: redis_config_abs_dir_env > TFRA_REDIS_CONFIG_PATH > redis_config_abs_dir
  
Besides, the default value of redis_config_abs_dir_env is "TFRA_REDIS_CONFIG_PATH".

The following Python code describes how you should introduce the Redis backend 
for training under the TFRA framework:

```python
redis_config1=tfra.dynamic_embedding.RedisTableConfig(
    redis_config_abs_dir_env="model1_tfra_redis_config_path",
    redis_config_abs_dir=“/tmp/test/”
)
redis_creator1=tfra.dynamic_embedding.RedisTableCreator(redis_config1)

self.user_embeddings = tfra.dynamic_embedding.get_variable(
            name="user_dynamic_embeddings",
            dim=self.embedding_size,
            initializer=tf.keras.initializers.RandomNormal(-1, 1),
            checkpoint=False,
            KVCreator=redis_creator1
            )
```

or you can do it like this:

```python
redis_config_dir = os.path.join(tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR')), "save_restore")
redis_config_path = os.path.join(tempfile.mkdtemp(prefix=redis_config_dir), "hash")
os.makedirs(redis_config_path)
redis_config_path = os.path.join(redis_config_path, "redis_config.json")
redis_config_params = {
  "redis_host_ip":["127.0.0.1"],
  "redis_host_port":[6379],
  "using_model_lib":False
}
with open(redis_config_path, 'w', encoding='utf-8') as f:
  f.write(json.dumps(redis_config_params, indent=2, ensure_ascii=True))
redis_config = de.RedisTableConfig(
  redis_config_abs_dir_env="model1_tfra_redis_config_path",
  redis_config_abs_dir=redis_config_path
)
redis_creator=tfra.dynamic_embedding.RedisTableCreator(redis_config)

self.user_embeddings = tfra.dynamic_embedding.get_variable(
            name="user_dynamic_embeddings",
            dim=self.embedding_size,
            initializer=tf.keras.initializers.RandomNormal(-1, 1),
            checkpoint=False,
            KVCreator= redis_creator
            )
```