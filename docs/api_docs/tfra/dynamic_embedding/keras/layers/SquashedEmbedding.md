<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfra.dynamic_embedding.keras.layers.SquashedEmbedding" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="compute_dtype"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="dtype_policy"/>
<meta itemprop="property" content="dynamic"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="metrics"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="supports_masking"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="variable_dtype"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_metric"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfra.dynamic_embedding.keras.layers.SquashedEmbedding

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/keras/layers/embedding.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>
<br/>
<br/>
<br/>
<br/>



## Class `SquashedEmbedding`

The SquashedEmbedding layer allow arbirary input shape of feature ids, and get

Inherits From: [`BasicEmbedding`](../../../../tfra/dynamic_embedding/keras/layers/BasicEmbedding.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tfra.dynamic_embedding.keras.layers.embedding.SquashedEmbedding`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->
(shape(ids) + embedding_size) lookup result. Finally the output of the
layer with be squashed to (batch_size, embedding_size) if the input is
batched, or (embedding_size) if it's a single example.

### Example
```python
embedding = SquashedEmbedding(8) # embedding size 8
ids = tf.constant([[15,2], [4,92], [22,4]], dtype=tf.int64) # (3, 2)
out = embedding(ids) # (3, 8)

ids = tf.constant([2, 7, 4, 1, 5], dtype=tf.int64) # (5,)
out = embedding(ids) # (8,)
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/keras/layers/embedding.py">View source</a>

``` python
__init__(
    embedding_size,
    key_dtype=tf.int64,
    value_dtype=tf.float32,
    combiner='sum',
    initializer=None,
    devices=None,
    name='DynamicEmbeddingLayer',
    **kwargs
)
```

Creates a BasicEmbedding layer.


#### Args:


* <b>`embedding_size`</b>: An object convertible to int. Length of embedding vector
  to every feature id.
* <b>`key_dtype`</b>: Dtype of the embedding keys to weights. Default is int64.
* <b>`value_dtype`</b>: Dtype of the embedding weight values. Default is float32
* <b>`combiner`</b>: A string or a function to combine the lookup result. It's value
  could be 'sum', 'mean', 'min', 'max', 'prod', 'std', etc. whose are
  one of tf.math.reduce_xxx.
* <b>`initializer`</b>: Initializer to the embedding values. Default is RandomNormal.
* <b>`devices`</b>: List of devices to place the embedding layer parameter.
* <b>`name`</b>: Name of the embedding layer.

* <b>`**kwargs`</b>:   trainable: Bool. Whether if the layer is trainable. Default is True.
  bp_v2: Bool. If true, the embedding layer will be updated by incremental
    amount. Otherwise, it will be updated by value directly. Default is
    False.
  restrict_policy: A RestrictPolicy class to restrict the size of
    embedding layer parameter since the dynamic embedding supports
    nearly infinite embedding space capacity.
  init_capacity: Integer. Initial number of kv-pairs in an embedding
    layer. The capacity will growth if the used space exceeded current
    capacity.
  partitioner: A function to route the keys to specific devices for
    distributed embedding parameter.
  kv_creator: A KVCreator object to create external KV storage as
    embedding parameter.
  max_norm: If not `None`, each values is clipped if its l2-norm is larger
  distribute_strategy: Used when creating ShadowVariable.
  keep_distribution: Bool. If true, save and restore python object with
    devices information. Default is false.



## Properties

<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.


<h3 id="compute_dtype"><code>compute_dtype</code></h3>

The dtype of the layer's computations.

This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
mixed precision is used, this is the same as `Layer.dtype`, the dtype of
the weights.

Layers automatically cast their inputs to the compute dtype, which causes
computations and the output to be in the compute dtype as well. This is done
by the base Layer class in `Layer.__call__`, so you do not have to insert
these casts if implementing your own layer.

Layers often perform certain internal computations in higher precision when
`compute_dtype` is float16 or bfloat16 for numeric stability. The output
will still typically be float16 or bfloat16 in such cases.

#### Returns:

The layer's compute dtype.


<h3 id="dtype"><code>dtype</code></h3>

The dtype of the layer weights.

This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
mixed precision is used, this is the same as `Layer.compute_dtype`, the
dtype of the layer's computations.

<h3 id="dtype_policy"><code>dtype_policy</code></h3>

The dtype policy associated with this layer.

This is an instance of a `tf.keras.mixed_precision.Policy`.

<h3 id="dynamic"><code>dynamic</code></h3>

Whether the layer is dynamic (eager-only); set in the constructor.


<h3 id="input"><code>input</code></h3>

Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.

#### Returns:

Input tensor or list of input tensors.



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.
* <b>`AttributeError`</b>: If no inbound nodes are found.

<h3 id="input_spec"><code>input_spec</code></h3>

`InputSpec` instance(s) describing the input format for this layer.

When you create a layer subclass, you can set `self.input_spec` to enable
the layer to run input compatibility checks when it is called.
Consider a `Conv2D` layer: it can only be called on a single input tensor
of rank 4. As such, you can set, in `__init__()`:

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
```

Now, if you try to call the layer on an input that isn't rank 4
(for instance, an input of shape `(2,)`, it will raise a nicely-formatted
error:

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
```

Input checks that can be specified via `input_spec` include:
- Structure (e.g. a single input, a list of 2 inputs, etc)
- Shape
- Rank (ndim)
- Dtype

For more information, see `tf.keras.layers.InputSpec`.

#### Returns:

A `tf.keras.layers.InputSpec` instance, or nested structure thereof.


<h3 id="losses"><code>losses</code></h3>

List of losses added using the `add_loss()` API.

Variable regularization tensors are created when this property is accessed,
so it is eager safe: accessing `losses` under a `tf.GradientTape` will
propagate gradients back to the corresponding variables.

#### Examples:



```
>>> class MyLayer(tf.keras.layers.Layer):
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
>>> l = MyLayer()
>>> l(np.ones((10, 1)))
>>> l.losses
[1.0]
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> x = tf.keras.layers.Dense(10)(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Activity regularization.
>>> len(model.losses)
0
>>> model.add_loss(tf.abs(tf.reduce_mean(x)))
>>> len(model.losses)
1
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
>>> x = d(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Weight regularization.
>>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
>>> model.losses
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
```

#### Returns:

A list of tensors.


<h3 id="metrics"><code>metrics</code></h3>

List of metrics added using the `add_metric()` API.


#### Example:



```
>>> input = tf.keras.layers.Input(shape=(3,))
>>> d = tf.keras.layers.Dense(2)
>>> output = d(input)
>>> d.add_metric(tf.reduce_max(output), name='max')
>>> d.add_metric(tf.reduce_min(output), name='min')
>>> [m.name for m in d.metrics]
['max', 'min']
```

#### Returns:

A list of `Metric` objects.


<h3 id="name"><code>name</code></h3>

Name of the layer (string), set in the constructor.


<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="non_trainable_weights"><code>non_trainable_weights</code></h3>

List of all non-trainable weights tracked by this layer.

Non-trainable weights are *not* updated during training. They are expected
to be updated manually in `call()`.

#### Returns:

A list of non-trainable variables.


<h3 id="output"><code>output</code></h3>

Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.

#### Returns:

Output tensor or list of output tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to more than one incoming
  layers.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> list(a.submodules) == [b, c]
True
>>> list(b.submodules) == [c]
True
>>> list(c.submodules) == []
True
```

#### Returns:

A sequence of all submodules.


<h3 id="supports_masking"><code>supports_masking</code></h3>

Whether this layer supports computing a mask using `compute_mask`.


<h3 id="trainable"><code>trainable</code></h3>




<h3 id="trainable_weights"><code>trainable_weights</code></h3>

List of all trainable weights tracked by this layer.

Trainable weights are updated via gradient descent during training.

#### Returns:

A list of trainable variables.


<h3 id="variable_dtype"><code>variable_dtype</code></h3>

Alias of `Layer.dtype`, the dtype of the weights.


<h3 id="weights"><code>weights</code></h3>

Returns the list of all layer variables/weights.


#### Returns:

A list of variables.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    *args,
    **kwargs
)
```

Wraps `call`, applying pre- and post-processing steps.


#### Args:


* <b>`*args`</b>: Positional arguments to be passed to `self.call`.
* <b>`**kwargs`</b>: Keyword arguments to be passed to `self.call`.


#### Returns:

Output tensor(s).



#### Note:

- The following optional keyword arguments are reserved for specific uses:
  * `training`: Boolean scalar tensor of Python boolean indicating
    whether the `call` is meant for training or inference.
  * `mask`: Boolean input mask.
- If the layer's `call` method takes a `mask` argument (as some Keras
  layers do), its default value will be set to the mask generated
  for `inputs` by the previous layer (if `input` did come from
  a layer that generated a corresponding mask, i.e. if it came from
  a Keras layer with masking support.
- If the layer is not built, the method will call `build`.



#### Raises:


* <b>`ValueError`</b>: if the layer's `call` method returns None (an invalid value).
* <b>`RuntimeError`</b>: if `super().__init__()` was not called in the constructor.

<h3 id="add_loss"><code>add_loss</code></h3>

``` python
add_loss(
    losses,
    **kwargs
)
```

Add loss tensor(s), potentially dependent on layer inputs.

Some losses (for instance, activity regularization losses) may be dependent
on the inputs passed when calling a layer. Hence, when reusing the same
layer on different inputs `a` and `b`, some entries in `layer.losses` may
be dependent on `a` and some on `b`. This method automatically keeps track
of dependencies.

This method can be used inside a subclassed layer or model's `call`
function, in which case `losses` should be a Tensor or list of Tensors.

#### Example:



```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any loss Tensors passed to this Model must
be symbolic and be able to be traced back to the model's `Input`s. These
losses become part of the model's topology and are tracked in `get_config`.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Activity regularization.
model.add_loss(tf.abs(tf.reduce_mean(x)))
```

If this is not the case for your loss (if, for example, your loss references
a `Variable` of one of the model's layers), you can wrap your loss in a
zero-argument lambda. These losses are not tracked as part of the model's
topology since they can't be serialized.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
d = tf.keras.layers.Dense(10)
x = d(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Weight regularization.
model.add_loss(lambda: tf.reduce_mean(d.kernel))
```

#### Args:


* <b>`losses`</b>: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
  may also be zero-argument callables which create a loss tensor.
* <b>`**kwargs`</b>: Additional keyword arguments for backward compatibility.
  Accepted values:
    inputs - Deprecated, will be automatically inferred.

<h3 id="add_metric"><code>add_metric</code></h3>

``` python
add_metric(
    value,
    name=None,
    **kwargs
)
```

Adds metric tensor to the layer.

This method can be used inside the `call()` method of a subclassed layer
or model.

```python
class MyMetricLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

  def call(self, inputs):
    self.add_metric(self.mean(inputs))
    self.add_metric(tf.reduce_sum(inputs), name='metric_2')
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any tensor passed to this Model must
be symbolic and be able to be traced back to the model's `Input`s. These
metrics become part of the model's topology and are tracked when you
save the model via `save()`.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(math_ops.reduce_sum(x), name='metric_1')
```

Note: Calling `add_metric()` with the result of a metric object on a
Functional Model, as shown in the example below, is not supported. This is
because we cannot trace the metric result tensor back to the model's inputs.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
```

#### Args:


* <b>`value`</b>: Metric tensor.
* <b>`name`</b>: String metric name.
* <b>`**kwargs`</b>: Additional keyword arguments for backward compatibility.
  Accepted values:
  `aggregation` - When the `value` tensor provided is not the result of
  calling a `keras.Metric` instance, it will be aggregated by default
  using a `keras.Metric.Mean`.

<h3 id="build"><code>build</code></h3>

``` python
build(input_shape)
```

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model`
can override if they need a state-creation step in-between
layer instantiation and layer call.

This is typically used to create the weights of `Layer` subclasses.

#### Args:


* <b>`input_shape`</b>: Instance of `TensorShape`, or list of instances of
  `TensorShape` if the layer expects a list of inputs
  (one instance per input).

<h3 id="compute_mask"><code>compute_mask</code></h3>

``` python
compute_mask(
    inputs,
    mask=None
)
```

Computes an output mask tensor.


#### Args:


* <b>`inputs`</b>: Tensor or list of tensors.
* <b>`mask`</b>: Tensor or list of tensors.


#### Returns:

None or a tensor (or list of tensors,
    one per output tensor of the layer).


<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

``` python
compute_output_shape(input_shape)
```

Computes the output shape of the layer.

If the layer has not been built, this method will call `build` on the
layer. This assumes that the layer will later be used with inputs that
match the input shape provided here.

#### Args:


* <b>`input_shape`</b>: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.


#### Returns:

An input shape tuple.


<h3 id="count_params"><code>count_params</code></h3>

``` python
count_params()
```

Count the total number of scalars composing the weights.


#### Returns:

An integer count.



#### Raises:


* <b>`ValueError`</b>: if the layer isn't yet built
  (in which case its weights aren't yet defined).

<h3 id="from_config"><code>from_config</code></h3>

``` python
@classmethod
from_config(
    cls,
    config
)
```

Creates a layer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by `set_weights`).

#### Args:


* <b>`config`</b>: A Python dictionary, typically the
    output of get_config.


#### Returns:

A layer instance.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/recommenders-addons/tree/master/tensorflow_recommenders_addons/dynamic_embedding/python/keras/layers/embedding.py">View source</a>

``` python
get_config()
```

Returns the config of the layer.

A layer config is a Python dictionary (serializable)
containing the configuration of a layer.
The same layer can be reinstantiated later
(without its trained weights) from this configuration.

The config of a layer does not include connectivity
information, nor the layer class name. These are handled
by `Network` (one layer of abstraction above).

Note that `get_config()` does not guarantee to return a fresh copy of dict
every time it is called. The callers should make a copy of the returned dict
if they want to modify it.

#### Returns:

Python dictionary.


<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```

Returns the current weights of the layer, as NumPy arrays.

The weights of a layer represent the state of the layer. This function
returns both trainable and non-trainable weight values associated with this
layer as a list of NumPy arrays, which can in turn be used to load state
into similarly parameterized layers.

For example, a `Dense` layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
`Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

#### Returns:

Weights values as a list of NumPy arrays.


<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```

Sets the weights of the layer, from NumPy arrays.

The weights of a layer represent the state of the layer. This function
sets the weight values from numpy arrays. The weight values should be
passed in the order they are created by the layer. Note that the layer's
weights must be instantiated before calling this function, by calling
the layer.

For example, a `Dense` layer returns a list of two values: the kernel matrix
and the bias vector. These can be used to set the weights of another
`Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

#### Args:


* <b>`weights`</b>: a list of NumPy arrays. The number
  of arrays and their shape must match
  number of the dimensions of the weights
  of the layer (i.e. it should match the
  output of `get_weights`).


#### Raises:


* <b>`ValueError`</b>: If the provided weights list does not match the
  layer's specifications.

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
@classmethod
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
...     return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
>>> mod = MyModule()
>>> mod(tf.ones([1, 2]))
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
>>> mod.w
<tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
numpy=..., dtype=float32)>
```

#### Args:


* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.




