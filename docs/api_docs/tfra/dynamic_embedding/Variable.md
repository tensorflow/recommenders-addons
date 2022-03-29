<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.Variable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="restrict_policy"/>
<meta itemprop="property" content="tables"/>
<meta itemprop="property" content="trainable_store"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="accum"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="get_slot_variables"/>
<meta itemprop="property" content="get_trainable_by_name"/>
<meta itemprop="property" content="lookup"/>
<meta itemprop="property" content="remove"/>
<meta itemprop="property" content="restrict"/>
<meta itemprop="property" content="size"/>
<meta itemprop="property" content="upsert"/>
</div>

# tfra.dynamic_embedding.Variable

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



## Class `Variable`

A Distributed version of HashTable(reference from lookup_ops.MutableHashTable)



<!-- Placeholder for "Used in" -->
It is designed to dynamically store the Sparse Weights(Parameters) of DLRMs.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
__init__(
    key_dtype=dtypes.int64,
    value_dtype=dtypes.float32,
    dim=1,
    devices=None,
    partitioner=default_partition_fn,
    shared_name=None,
    name='DynamicEmbedding_Variable',
    initializer=None,
    trainable=(True),
    checkpoint=(True),
    init_size=0,
    kv_creator=None,
    restrict_policy=None,
    bp_v2=(False)
)
```

Creates an empty `Variable` object.

Creates a group of tables placed on devices specified by `devices`,
and the device placement mechanism of TensorFlow will be ignored,
the type of its keys and values are specified by key_dtype
and value_dtype, respectively.
The environment variables 'TF_HASHTABLE_INIT_SIZE' can be used to set the
inital size of each tables, which can help reduce rehash times.
The default initial table size is 8,192

#### Args:


* <b>`key_dtype`</b>: the type of the key tensors.
* <b>`value_dtype`</b>: the type of the value tensors.
* <b>`dim`</b>: the length of the value array for each key,
  on GPUs, `dim` should be less or equal to 200.
* <b>`devices`</b>: the list of devices holding the tables.
  One table will be created on each device. By default, `devices` is
  ['/CPU:0'] and when GPU is available, `devices` is ['/GPU:0']
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
* <b>`trainable`</b>: Bool. If true, the variable will be treated as a trainable.
  Default is true.
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
* <b>`bp_v2`</b>: By default with `bp_v2=False`, the optimizer will update
  dynamic embedding values by *setting* (key, value) after
  `optimizer.apply_gradient`. If one key is used by multiple workers
  at the same time, only one of them will be seen, while the others are
  overwritten. By setting `bp_v2=True`, the optimizer will update
  parameters by *adding delta* instead of *setting*, which solves the
  race condition problem among workers during backpropagation in
  large-scale distributed asynchronous training.


#### Returns:

A `Variable` object.




## Properties

<h3 id="restrict_policy"><code>restrict_policy</code></h3>




<h3 id="tables"><code>tables</code></h3>




<h3 id="trainable_store"><code>trainable_store</code></h3>






## Methods

<h3 id="accum"><code>accum</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
accum(
    keys,
    old_values,
    new_values,
    exists,
    name=None
)
```

Insert `keys` with `values` if not exist, or accumulate a delta value
  `new_values - old_values` to 'keys'.
This API will help relieve stale gradient problem in asynchronous training.

#### Args:


* <b>`keys`</b>: Keys to insert. Can be a tensor of any shape. Must match
  the table's key type.
* <b>`old_values`</b>: old values to be associated with keys. Must be a tensor of
  arrays with same shape as `keys` and match the table's value type.
* <b>`new_values`</b>: new values to be associated with keys. Must be a tensor of
  arrays with same shape as `keys` and match the table's value type.
* <b>`exists`</b>: A bool type tensor indicates if keys existed or not.
  Must be a tensor of the same shape as `keys`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.



#### Raises:


* <b>`TypeError`</b>: when `keys` or `values` doesn't match the table data types.

<h3 id="clear"><code>clear</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
clear(name=None)
```

clear all keys and values in the table.


#### Args:


* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.


<h3 id="export"><code>export</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
export(name=None)
```

Returns tensors of all keys and values in the table.


#### Args:


* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A pair of tensors with the first tensor containing all keys and the
  second tensors containing all values in the table.


<h3 id="get_slot_variables"><code>get_slot_variables</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
get_slot_variables(optimizer)
```

Get slot variables from optimizer. If Variable is trained by optimizer,
then it returns the variables in slots of optimizer, else return an empty
list.

#### Args:


* <b>`optimizer`</b>: An optimizer under `tf.keras.optimizers` or `tf.compat.v1.train`.


#### Returns:

List of slot `Variable`s in optimizer.


<h3 id="get_trainable_by_name"><code>get_trainable_by_name</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
get_trainable_by_name(name)
```

Get trainable shadow variable when using eager execution.


#### Example:


```python
from tensorflow_recommenders_addons import dynamic_embedding as de
init = tf.keras.initializers.RandomNormal()
params = de.get_variable('foo', dim=4, initializer=init)
optimizer = tf.keras.optimizers.Adam(1E-3)
optimizer = de.DynamicEmbeddingOptimizer(optimizer)

@tf.function
def loss_fn(ids):
  emb = de.embedding_lookup(params, ids, name='user_embedding')
  emb = tf.math.reduce_sum(emb, axis=1)
  loss = tf.reduce_mean(emb)
  return loss

for i in range(10):
  optimizer.minimize(lambda: loss_fn(ids),
                     var_list=[params.get_eager_trainable_by_name('user_embedding')])
```

#### Args:


* <b>`name`</b>: str. Name used to get the trainable shadow to the Variable.


#### Returns:

A ShadowVariable object refers to the specific name.



#### Raises:


* <b>`RuntimeError`</b>: if not in eager mode.

<h3 id="lookup"><code>lookup</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
lookup(
    keys,
    return_exists=(False),
    name=None
)
```

Looks up `keys` in a Variable, outputs the corresponding values.

The `default_value` is used for keys not present in the table.

#### Args:


* <b>`keys`</b>: Keys to look up. Can be a tensor of any shape. Must match the
  table's key_dtype.
* <b>`return_exists`</b>: if True, will return a additional Tensor which indicates
  if keys are existing in the table.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A tensor containing the values in the same shape as `keys` using the
  table's value type.

* <b>`exists`</b>:   A bool type Tensor of the same shape as `keys` which indicates
    if keys are existing in the table.
    Only provided if `return_exists` is True.

<h3 id="remove"><code>remove</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

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

<h3 id="restrict"><code>restrict</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

``` python
restrict(
    num_reserved,
    **kwargs
)
```

Restrict the size of self, also including features reside in commensal
slots, and the policy status. The restriction rule follow the setting
in `restrict_policy`.

#### Args:


* <b>`num_reserved`</b>: int. Number of remaining features after restriction.
* <b>`**kwargs`</b>: keyword arguments passing to `restrict_policy.apply_restriction`.


#### Returns:

An operation to restrict size of the variable itself. Return None if
the restrict policy is not set.


<h3 id="size"><code>size</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_variable.py">View source</a>

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
* <b>`values`</b>: Values to be associated with keys.Must be a tensor of
  arrays with same shape as `keys` and match the table's value type.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.



#### Raises:


* <b>`TypeError`</b>: when `keys` or `values` doesn't match the table data
  types.



