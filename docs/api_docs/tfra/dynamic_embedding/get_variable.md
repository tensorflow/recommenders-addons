<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.get_variable" />
<meta itemprop="path" content="Stable" />
</div>

# tfra.dynamic_embedding.get_variable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



Gets an `Variable` object with this name if it exists,

``` python
tfra.dynamic_embedding.get_variable(
    name,
    key_dtype=dtypes.int64,
    value_dtype=dtypes.float32,
    dim=1,
    devices=None,
    partitioner=default_partition_fn,
    shared_name='get_variable',
    initializer=None,
    trainable=True,
    checkpoint=True,
    init_size=0,
    restrict_policy=None,
    bp_v2=False,
)
```



<!-- Placeholder for "Used in" -->
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
* <b>`init_size`</b>: initial size for the Variable and initial size of each hash 
  tables will be int(init_size / N), N is the number of the devices.
* <b>`restrict_policy`</b>: a restrict policy to specify the rule to restrict the
  size of variable. If in training program, the variable is updated by
  optimizer, then the sparse slot variables in optimizer are also be
  restricted.
* <b>`bp_v2`</b>:update parameters by *updating* instead of *setting*, which solves
  the race condition problem among workers during backpropagation in large-scale
  distributed asynchronous training.


#### Returns:

A `Variable` object.
