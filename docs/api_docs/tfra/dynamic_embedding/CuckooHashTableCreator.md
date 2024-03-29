<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.CuckooHashTableCreator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfra.dynamic_embedding.CuckooHashTableCreator

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



## Class `CuckooHashTableCreator`

  A generic KV table creator.



<!-- Placeholder for "Used in" -->

  KV table instance will be created by the create function with config.
And also a config class for specific table instance backend should be
inited before callling the creator function.
  And then, the KVCreator class instance will be passed to the Variable
class for creating the real KV table backend(TF resource).

#### Example usage:

Due to CuckooHashTableConfig include nothing for parameter default satisfied. Just setting the parameter saver is enough.

```python
cuckoo_creator=tfra.dynamic_embedding.CuckooHashTableCreator(saver=de.FileSystemSaver())
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
    init_size=None,
    config=None
)
```




<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">View source</a>

``` python
get_config()
```






