# Description

Addding Redis as the backend for KV storage and changed the Python interface by calling the Config class 
and creating/passing the KvCreator class to fit the different backend. Also setting backend parameters with json file.

The implementation of TFRA backend relies on the Redis-plus-plus and hiredis component. 
How to install Redis-plus-plus please visit the following page:
https://github.com/sewenew/redis-plus-plus

Redis-plus-plus is a Redis client, based on hiredis and written in C++11. 
It supports scritpting, pub/sub, pipeline, transaction, Redis Cluster, Redis Sentinel, connection pool, ACL, SSL and thread safety.

the method for parsing json file was created by James McLaughlin et al.  

# How To Create a Config File?
Below is an example of a JSON file, along with comments on the corresponding project. 
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
      "storage_slice": 1,  // For deciding hash tag, which usually is how many Redis instance may be used in the trainning.
      "keys_sending_size": 1024,  // Determines how many keys to send at a time for performance tuning
      "using_md5_prefix_name": False,  // 1=true, 0=false
      "model_tag_old": "test",  // model_tag_old for version and any other information from last time.
      "model_tag_new": "test",  // model_tag_new for version and any other information for now.
      "expire_model_tag_in_seconds": 604800,  // To eliminate unwanted model versions in Redis to ensure sufficient storage space.
      "table_store_mode": 1,  // Saving and restoring table into ensor in TF savedmodel variable file, table_store_mode = 0; Saving and restoring table into redis rdb file in model_lib_abs_dir, table_store_mode = 1; Saving and restoring nothing, keeping data in redis servers, table_store_mode = 2.
      "model_lib_abs_dir": "/tmp/"  // if table_store_mode equals 1, then it will try to save or resoter table from model_lib_abs_dir which has been mounted in system
  }
```
  
# How To Use in TensorFlow?

```python
redis_config1=tfra.dynamic_embedding.RedisTableConfig(
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