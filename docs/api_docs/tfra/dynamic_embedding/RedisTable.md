<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.RedisTable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="key_dtype"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="resource_handle"/>
<meta itemprop="property" content="value_dtype"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="accum"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="insert"/>
<meta itemprop="property" content="lookup"/>
<meta itemprop="property" content="remove"/>
<meta itemprop="property" content="size"/>
<meta itemprop="property" content="default_redis_params"/>
</div>

# tfra.dynamic_embedding.RedisTable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



## Class `RedisTable`

A generic mutable hash table implementation.



<!-- Placeholder for "Used in" -->

Data can be inserted by calling the insert method and removed by calling the
remove method. It does not support initialization via the init method.

#### Example usage:



```python
table = tfra.dynamic_embedding.RedisTable(key_dtype=tf.string,
                                               value_dtype=tf.int64,
                                               default_value=-1)
sess.run(table.insert(keys, values))
out = table.lookup(query_keys)
print(out.eval())
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
__init__(
    key_dtype,
    value_dtype,
    default_value,
    name='RedisTable',
    checkpoint=(False),
    config=None
)
```

Creates an empty `RedisTable` object.

Creates a redis table through OS envionment variables,
the type of its keys and values are specified by key_dtype
and value_dtype, respectively.

#### Args:


* <b>`key_dtype`</b>: the type of the key tensors.
* <b>`value_dtype`</b>: the type of the value tensors.
* <b>`default_value`</b>: The value to use if a key is missing in the table.
* <b>`name`</b>: A name for the operation (optional, usually it's embedding table name).
* <b>`checkpoint`</b>: if True, the contents of the table are saved to and restored
  from a Redis binary dump files according to the directory "[model_lib_abs_dir]/[model_tag]/[name].rdb".
  If `shared_name` is empty for a checkpointed table, it is shared using the table node name.


#### Returns:

A `RedisTable` object.



#### Raises:


* <b>`ValueError`</b>: If checkpoint is True and no name was specified.



## Properties

<h3 id="key_dtype"><code>key_dtype</code></h3>

The table key dtype.


<h3 id="name"><code>name</code></h3>

The name of the table.


<h3 id="resource_handle"><code>resource_handle</code></h3>

Returns the resource handle associated with this Resource.


<h3 id="value_dtype"><code>value_dtype</code></h3>

The table value dtype.




## Methods

<h3 id="__getitem__"><code>__getitem__</code></h3>

``` python
__getitem__(keys)
```

Looks up `keys` in a table, outputs the corresponding values.


<h3 id="accum"><code>accum</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
accum(
    keys,
    values_or_deltas,
    exists,
    name=None
)
```

Associates `keys` with `values`.


#### Args:


* <b>`keys`</b>: Keys to accmulate. Can be a tensor of any shape.
  Must match the table's key type.
* <b>`values_or_deltas`</b>: values to be associated with keys. Must be a tensor of
  the same shape as `keys` and match the table's value type.
* <b>`exists`</b>: A bool type tensor indicates if keys already exist or not.
  Must be a tensor of the same shape as `keys`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.



#### Raises:


* <b>`TypeError`</b>: when `keys` or `values` doesn't match the table data
  types.

<h3 id="clear"><code>clear</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
clear(name=None)
```

clear all keys and values in the table.


#### Args:


* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.


<h3 id="export"><code>export</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
export(name=None)
```

Returns nothing in Redis Implement. It will dump some binary files
to model_lib_abs_dir.

#### Args:


* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A pair of tensors with the first tensor containing all keys and the
  second tensors containing all values in the table.


<h3 id="insert"><code>insert</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
insert(
    keys,
    values,
    name=None
)
```

Associates `keys` with `values`.


#### Args:


* <b>`keys`</b>: Keys to insert. Can be a tensor of any shape. Must match the table's
  key type.
* <b>`values`</b>: Values to be associated with keys. Must be a tensor of the same
  shape as `keys` and match the table's value type.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.



#### Raises:


* <b>`TypeError`</b>: when `keys` or `values` doesn't match the table data
  types.

<h3 id="lookup"><code>lookup</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
lookup(
    keys,
    dynamic_default_values=None,
    return_exists=(False),
    name=None
)
```

Looks up `keys` in a table, outputs the corresponding values.

The `default_value` is used for keys not present in the table.

#### Args:


* <b>`keys`</b>: Keys to look up. Can be a tensor of any shape. Must match the
  table's key_dtype.
* <b>`dynamic_default_values`</b>: The values to use if a key is missing in the
  table. If None (by default), the static default_value
  `self._default_value` will be used.
* <b>`return_exists`</b>: if True, will return a additional Tensor which indicates
  if or not keys are existing in the table.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A tensor containing the values in the same shape as `keys` using the
  table's value type.

* <b>`exists`</b>:   A bool type Tensor of the same shape as `keys` which indicates
    if keys are existing in the table.
    Only provided if `return_exists` is True.


#### Raises:


* <b>`TypeError`</b>: when `keys` do not match the table data types.

<h3 id="remove"><code>remove</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
remove(
    keys,
    name=None
)
```

Removes `keys` and its associated values from the table.

If a key is not present in the table, it is silently ignored.

#### Args:


* <b>`keys`</b>: Keys to remove. Can be a tensor of any shape. Must match the table's
  key type.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.



#### Raises:


* <b>`TypeError`</b>: when `keys` do not match the table data types.

<h3 id="size"><code>size</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/redis_table_ops.py">View source</a>

``` python
size(name=None)
```

Compute the number of elements in this table.


#### Args:


* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A scalar tensor containing the number of elements in this table.




## Class Members

* `default_redis_params` <a id="default_redis_params"></a>
