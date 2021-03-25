<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.Variable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="resource_handle"/>
<meta itemprop="property" content="tables"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="lookup"/>
<meta itemprop="property" content="remove"/>
<meta itemprop="property" content="size"/>
<meta itemprop="property" content="upsert"/>
</div>

# tfra.dynamic_embedding.Variable

## Class `Variable`



A Distributed version of HashTable
It is designed to dynamically store the Sparse Weights(Parameters) of DLRMs.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    key_dtype=tf.dtypes.int64,
    value_dtype=tf.dtypes.float32,
    dim=1,
    devices=None,
    partitioner=default_partition_fn,
    shared_name=None,
    name='DynamicEmbedding_Variable',
    initializer=None,
    trainable=True,
    checkpoint=True,
    init_size=0
)
```

Creates an empty `Variable` object.

Creates a group of tables placed on devices,
the type of its keys and values are specified by key_dtype
and value_dtype, respectively.

The environment variables 'TF_HASHTABLE_INIT_SIZE' can be used to set the
inital size of each tables, which can help reduce rehash times.
The default initial table size : 1,048,576 for CPU, 16,777,216 for GPU.

#### Args:

* <b>`key_dtype`</b>: the type of the key tensors.
* <b>`value_dtype`</b>: the type of the value tensors.
* <b>`dim`</b>: the length of the value array for each key.
* <b>`devices`</b>: the list of devices holding the tables.
    One table will be created on each device.
* <b>`partitioner`</b>: partition function of keys,
    return the partition index for each key.

  Example partition func:
  ```python
  def default_partition_fn(keys, shard_num):
    return tf.cast(keys % shard_num, dtype=tf.int32)
  ```
* <b>`shared_name`</b>: No used.
* <b>`name`</b>: A name for the operation (optional).
* <b>`initializer`</b>: The value to use if a key is missing in the hash table.
    which can be a python number, numpy array or `tf.initializer` instances.
    If initializer is `None` (the default), `0` will be taken.
* <b>`trainable`</b>: True, will be treated as a trainable Variable, and add to
    to the list of variables collected in the graph under the key
    `GraphKeys.TRAINABLE_VARIABLES`.
* <b>`checkpoint`</b>: if True, the contents of the SparseVariable are
    saved to and restored from checkpoints.
    If `shared_name` is empty for a checkpointed table,
    it is shared using the table node name.
* <b>`init_size`</b>: initial size for the Variable and initial size of each
    hash tables will be `int(init_size / N)`, `N` is the number of
    the devices.


#### Returns:

A `Variable` object.



## Properties

<h3 id="resource_handle"><code>resource_handle</code></h3>

Returns the resource handle associated with this Resource.

<h3 id="tables"><code>tables</code></h3>





## Methods

<h3 id="export"><code>export</code></h3>

``` python
export(name=None)
```

Returns tensors of all keys and values in the table.

#### Args:

* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A pair of tensors with the first tensor containing all keys and the
  second tensors containing all values in the table.

<h3 id="lookup"><code>lookup</code></h3>

``` python
lookup(
    keys,
    name=None
)
```

Looks up `keys` in a Variable, outputs the corresponding values.

The `default_value` is used for keys not present in the table.

#### Args:

* <b>`keys`</b>: Keys to look up. Can be a tensor of any shape. Must match the
    table's key_dtype.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A tensor containing the values in the same shape as `keys` using the
  table's value type.

<h3 id="remove"><code>remove</code></h3>

``` python
remove(
    keys,
    name=None
)
```

Removes `keys` and its associated values from the variable.

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

``` python
size(
    index=None,
    name=None
)
```

Compute the number of elements in the index-th table of this Variable.

If index is none, the total size of the Variable wil be return.

#### Args:

* <b>`index`</b>: The index of table (optional)
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A scalar tensor containing the number of elements in this Variable.

<h3 id="upsert"><code>upsert</code></h3>

``` python
upsert(
    keys,
    values,
    name=None
)
```

Insert or Update `keys` with `values`.

If key exists already, value will be updated.

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



