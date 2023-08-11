<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.HkvTableCreator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfra.dynamic_embedding.HkvHashTableCreator

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



## Class `HkvHashTableCreator`

  A generic KV table creator.



<!-- Placeholder for "Used in" -->

  KV table instance will be created by the create function with config.
And also a config class for specific table instance backend should be
inited before callling the creator function.
  And then, the KVCreator class instance will be passed to the Variable
class for creating the real KV table backend(TF resource).

#### Example usage:



```python
hkv_config=tfra.dynamic_embedding.HkvHashTableConfig(
  init_capacity=1024 * 1024,
  max_capacity=1024 * 1024,
  max_hbm_for_values=0,
)
hkv_creator=tfra.dynamic_embedding.HkvHashTableCreator(config=hkv_config)
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">View source</a>

``` python
__init__(config=None)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">View source</a>

``` python
create(
    key_dtype=None,
    value_dtype=None,
    default_value=None,
    name=None,
    checkpoint=None,
    init_size=KHkvHashTableInitCapacity,
    config=None,
    device=None,
    shard_saveable_object_fn=None,
)
```




<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">View source</a>

``` python
get_config()

#return as follow

config = {
    'key_dtype': self.key_dtype,
    'value_dtype': self.value_dtype,
    'default_value': self.default_value.numpy(),
    'name': self.name,
    'checkpoint': self.checkpoint,
    'init_capacity': self.init_capacity,
    'max_capacity': self.max_capacity,
    'max_hbm_for_values': self.max_hbm_for_values
    'config': self.config,
    'device': self.device,
}
```






