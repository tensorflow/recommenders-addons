<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.get_variable" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.get_variable

``` python
tfra.dynamic_embedding.get_variable(
    name,
    key_dtype=tf.dtypes.int64,
    value_dtype=tf.dtypes.float32,
    dim=1,
    devices=None,
    partitioner=default_partition_fn,
    shared_name='get_variable',
    initializer=None,
    trainable=True,
    checkpoint=True,
    init_size=0
)
```

Gets an `Variable` object with this name if it exists,
     or create a new one.

#### Args:

* <b>`name`</b>: A unique name for the `Variable`.
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
* <b>`initializer`</b>: The value to use if a key is missing in the hash table.
    which can a python number, numpy array or `tf.initializer` instances.
    If initializer is `None` (the default), `0` will be used.
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