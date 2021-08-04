# Description
# Description

Addding Redis as the backend for KV storage and changed the Python interface by calling the Config class 
and creating/passing the KvCreator class to fit the different backend. Also setting backend parameters with json file.

The implementation of TFRA backend relies on the Redis-plus-plus and hiredis component. 
How to install Redis-plus-plus please visit the following page:
https://github.com/sewenew/redis-plus-plus

Redis-plus-plus is a Redis client, based on hiredis and written in C++11. 
It supports scritpting, pub/sub, pipeline, transaction, Redis Cluster, Redis Sentinel, connection pool, ACL, SSL and thread safety.

# How Has This Been Used?

```
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

```
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