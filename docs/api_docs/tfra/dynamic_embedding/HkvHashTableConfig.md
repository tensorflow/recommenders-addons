<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.HkvHashTableConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tfra.dynamic_embedding.HkvHashTableConfig

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



## Class `HkvHashTableConfig`





<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_creator.py">View source</a>

``` python

KHkvHashTableInitCapacity = 1024 * 1024
KHkvHashTableMaxCapacity = 1024 * 1024
KHkvHashTableMaxHbmForValuesByBytes = 1024 * 1024 * 1024

__init__(
    init_capacity=KHkvHashTableInitCapacity,
    max_capacity=KHkvHashTableMaxCapacity,
    max_hbm_for_values=KHkvHashTableMaxHbmForValuesByBytes
):
```

HkvHashTableConfig contains three parameters to configure the HashTable, They all have default values.

#### Args:


* <b>`init_capacity`</b>: The initial capacity of the hash table.
* <b>`max_capacity`</b>: The maximum capacity of the hash table.
* <b>`max_hbm_for_values`</b>: The maximum HBM for values, in bytes.

#### Configuration Suggestion

* <b>`Pure HBM mode`</b>: set the max_hbm_for_values >= sizeof(V) * dim * max_capacity
* <b>`HBM + HMEM mode`</b>: set the max_hbm_for_values < sizeof(V) * dim * max_capacity
* <b>`Pure HMEM mode`</b>: set the max_hbm_for_values = 0
* if max_capacity == init_capacity, the HBM + HMEM consumption = sizeof(V) * dim * max_capacity
