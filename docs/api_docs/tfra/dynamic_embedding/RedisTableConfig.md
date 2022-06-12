<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.RedisTableConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tfra.dynamic_embedding.RedisTableConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



## Class `RedisTableConfig`

RedisTableConfig config json file for connecting Redis service and 



<!-- Placeholder for "Used in" -->
assign the embedding table starage properties.
An example of a configuration file is shown below:
```python
{
  "redis_connection_mode": 2,
  "redis_master_name": "master",
  "redis_host_ip": [
    "127.0.0.1"
  ],
  "redis_host_port": 6379,
  "redis_user": "default",
  "redis_password": "",
  "redis_db": 0,
  "redis_read_access_slave": False,
  "redis_connect_keep_alive": False,
  "redis_connect_timeout": 1000,
  "redis_socket_timeout": 1000,
  "redis_conn_pool_size": 20,
  "redis_wait_timeout": 100000000,
  "redis_connection_lifetime": 100,
  "redis_sentinel_user": "default",
  "redis_sentinel_password": "",
  "redis_sentinel_connect_timeout": 1000,
  "redis_sentinel_socket_timeout": 1000,
  "storage_slice_import": 1,
  "storage_slice": 1,
  "using_hash_storage_slice": False,
  "keys_sending_size": 1024,
  "using_md5_prefix_name": False,
  "redis_hash_tags_hypodispersion": False,
  "model_tag_import": "test",
  "redis_hash_tags_import": [],
  "model_tag_runtime": "test",
  "redis_hash_tags_runtime": [],
  "expire_model_tag_in_seconds": 604800,
  "table_store_mode": 1,
  "model_lib_abs_dir": "/tmp/"
}
```
Refer to the [Redis table config guide](https://github.com/tensorflow/recommenders-addons/blob/master/docs/api_docs/tfra/dynamic_embedding/RedisBackend.md)
and default_redis_params variable in RedisTable class 
to learn the description of the JSON configuration file

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">View source</a>

``` python
__init__(
    redis_config_abs_dir=None,
    redis_config_abs_dir_env=None
)
```

Initialize self.  See help(type(self)) for accurate signature.




