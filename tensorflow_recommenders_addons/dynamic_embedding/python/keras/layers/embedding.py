# Copyright 2021 The TensorFlow Recommenders-Addons Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# lint-as: python3
"""
Dynamic Embedding is designed for Large-scale Sparse Weights Training.
See [Sparse Domain Isolation](https://github.com/tensorflow/community/pull/237)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.ops import init_ops
from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import dynamic_embedding_variable as devar


def _choose_reduce_method(combiner, sparse=False, segmented=False):
  select = 'sparse' if sparse else 'math'
  try:
    module = getattr(tf, select)
  except:
    raise AttributeError('tensorflow has no attribute {}'.format(select))
  select = 'segment_' if segmented else 'reduce_'
  select += combiner
  try:
    method = getattr(module, select)
  except:
    raise AttributeError('Module [{}] has no attribute {}'.format(
        module, select))
  if not callable(method):
    raise ValueError('{}: {} in {} is not callable'.format(
        select, method, module))
  return method


@tf.function
def reduce_pooling(x, combiner='sum'):
  """
  Default combine_fn for Embedding layer. By assuming input
  ids shape is (batch_size, s1, ..., sn), it will get lookup result
  with shape (batch_size, s1, ..., sn, embedding_size). Every
  sample in a batch will be reduecd to a single vector, and thus
  the output shape is (batch_size, embedding_size)
  """

  ndims = x.shape.ndims
  combiner = _choose_reduce_method(combiner, sparse=False, segmented=False)

  with tf.name_scope('deep_squash_pooling'):

    if ndims == 1:
      raise ValueError("reduce_pooling need at least dim-2 input.")
    elif ndims == 2:
      return combiner(x, 0)

    for i in range(0, ndims - 2):
      x = combiner(x, 1)
    return x


class BasicEmbedding(tf.keras.layers.Layer):
  """
  A keras style Embedding layer. The `BasicEmbedding` layer acts same like
  [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding),
  except that the `BasicEmbedding` has dynamic embedding space so it does
  not need to set a static vocabulary size, and there will be no hash conflicts
  between features.

  The embedding layer allow arbirary input shape of feature ids, and get
  (shape(ids) + embedding_size) lookup result. Normally the first dimension
  is batch_size.

  ### Example
  ```python
  embedding = dynamic_embedding.keras.layers.BasicEmbedding(8) # embedding size 8
  ids = tf.constant([[15,2], [4,92], [22,4]], dtype=tf.int64) # (3, 2)
  out = embedding(ids) # lookup result, (3, 2, 8)
  ```

  You could inherit the `Embedding` class to implement a custom embedding
  layer with other fixed shape output.

  TODO(Lifann) Currently the BasicEmbedding only implemented in eager mode
  API, need to support graph mode also.
  """

  def __init__(self,
               embedding_size,
               key_dtype=tf.int64,
               value_dtype=tf.float32,
               combiner='sum',
               initializer=None,
               devices=None,
               name='DynamicEmbeddingLayer',
               **kwargs):
    """
    Creates a BasicEmbedding layer.

    Args:
      embedding_size: An object convertible to int. Length of embedding vector
        to every feature id.
      key_dtype: Dtype of the embedding keys to weights. Default is int64.
      value_dtype: Dtype of the embedding weight values. Default is float32
      combiner: A string or a function to combine the lookup result. It's value
        could be 'sum', 'mean', 'min', 'max', 'prod', 'std', etc. whose are
        one of tf.math.reduce_xxx.
      initializer: Initializer to the embedding values. Default is RandomNormal.
      devices: List of devices to place the embedding layer parameter.
      name: Name of the embedding layer.

      **kwargs:
        trainable: Bool. Whether if the layer is trainable. Default is True.
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
    """

    try:
      embedding_size = int(embedding_size)
    except:
      raise TypeError(
          'embeddnig size must be convertible to integer, but get {}'.format(
              type(embedding_size)))

    try:
      embedding_size = int(embedding_size)
    except:
      raise TypeError(
          'embeddnig size must be convertible to integer, but get {}'.format(
              type(embedding_size)))

    self.embedding_size = embedding_size
    self.combiner = combiner
    if initializer is None:
      initializer = tf.keras.initializers.RandomNormal()
    partitioner = kwargs.get('partitioner', devar.default_partition_fn)
    trainable = kwargs.get('trainable', True)
    self.max_norm = kwargs.get('max_norm', None)
    self.keep_distribution = kwargs.get('keep_distribution', False)

    parameter_name = name + '-parameter' if name else 'EmbeddingParameter'
    with tf.name_scope('DynamicEmbedding'):
      self.params = de.get_variable(parameter_name,
                                    key_dtype=key_dtype,
                                    value_dtype=value_dtype,
                                    dim=self.embedding_size,
                                    devices=devices,
                                    partitioner=partitioner,
                                    shared_name='layer_embedding_variable',
                                    initializer=initializer,
                                    trainable=trainable,
                                    checkpoint=True,
                                    init_size=kwargs.get('init_capacity', 0),
                                    kv_creator=kwargs.get('kv_creator', None),
                                    restrict_policy=kwargs.get(
                                        'restrict_policy', None),
                                    bp_v2=kwargs.get('bp_v2', False))

      self.distribute_strategy = kwargs.get('distribute_strategy', None)
      shadow_name = name + '-shadow' if name else 'ShadowVariable'
      self.shadow = de.shadow_ops.ShadowVariable(
          self.params,
          name=shadow_name,
          max_norm=self.max_norm,
          trainable=trainable,
          distribute_strategy=self.distribute_strategy)
    self._current_ids = self.shadow.ids
    self._current_exists = self.shadow.exists
    self.optimizer_vars = self.shadow._optimizer_vars
    super(BasicEmbedding, self).__init__(name=name,
                                         trainable=trainable,
                                         dtype=value_dtype)

  def call(self, ids):
    """
    Compute embedding output for feature ids. The output shape will be (shape(ids), 
    embedding_size).

    Args:
      ids: feature ids of the input. It should be same dtype as the key_dtype
        of the layer.

    Returns:
      A embedding output with shape (shape(ids), embedding_size).
    """
    ids = tf.convert_to_tensor(ids)
    input_shape = tf.shape(ids)
    lookup_result = de.shadow_ops.embedding_lookup(self.shadow, ids)
    lookup_result = tf.reshape(
        lookup_result, tf.concat([input_shape, [self.embedding_size]], 0))
    return lookup_result

  def get_config(self):
    _initializer = self.params.initializer
    if _initializer is None:
      _initializer = tf.keras.initializers.Zeros()
    _max_norm = None
    if isinstance(self.max_norm, tf.keras.constraints.Constraint):
      _max_norm = tf.keras.constraints.serialize(self.max_norm)

    if self.params.restrict_policy:
      _restrict_policy = self.params.restrict_policy.__class__
    else:
      _restrict_policy = None

    config = {
        'embedding_size':
            self.embedding_size,
        'key_dtype':
            self.params.key_dtype,
        'value_dtype':
            self.params.value_dtype,
        'combiner':
            self.combiner,
        'initializer':
            tf.keras.initializers.serialize(_initializer),
        'devices':
            self.params.devices if self.keep_distribution else None,
        'name':
            self.name,
        'trainable':
            self.trainable,
        'bp_v2':
            self.params.bp_v2,
        'restrict_policy':
            _restrict_policy,
        'init_capacity':
            self.params.init_size,
        'partitioner':
            self.params.partition_fn,
        'kv_creator':
            self.params.kv_creator if self.keep_distribution else None,
        'max_norm':
            _max_norm,
        'distribute_strategy':
            self.distribute_strategy,
    }
    return config


class SquashedEmbedding(BasicEmbedding):
  """
  The SquashedEmbedding layer allow arbirary input shape of feature ids, and get
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
  """

  def call(self, ids):
    ids = tf.convert_to_tensor(ids)
    input_shape = tf.shape(ids)
    lookup_result = de.shadow_ops.embedding_lookup(self.shadow, ids)
    lookup_result = tf.reshape(
        lookup_result, tf.concat([input_shape, [self.embedding_size]], 0))
    embedding = reduce_pooling(lookup_result, combiner=self.combiner)
    return embedding


class FieldWiseEmbedding(BasicEmbedding):
  """
  An embedding layer, which feature ids are mapped into fields.

  A field means the category of a feature id. Assume we have N fields, then
  fields are counted from 0 to N-1. Every feature id belongs to a field
  slot. And features ids which belong to the same field will be reduced
  into a embedding vector. And the output of FieldWiseEmbedding will be
  (batch_size, N, embedding_size).

  Example:

  ```python
  nslots = 3
  @tf.function
  def feature_to_slot(feature_id):
    field_id = tf.math.mod(feature_id, nslots)
    return field_id

  ids = tf.constant([[23, 12, 0], [9, 13, 10]], dtype=tf.int64)
  embedding = de.layers.FieldWiseEmbedding(2,
                                           nslots,
                                           slot_map_fn=feature_to_slot,
                                           initializer=tf.keras.initializer.Zeros())

  out = embedding(ids)
  # [[[0., 0.], [0., 0.], [0., 1.]]
  #  [[0., 0.], [0., 0.], [0., 1.]]]

  prepared_keys = tf.range(0, 100, dtype=tf.int64)
  prepared_values = tf.ones((100, 2), dtype=tf.float32)
  embedding.params.upsert(prepared_keys, prepared_values)
  out = embedding(ids)
  # [[2., 2.], [0., 0.], [1., 1.]]
  # [[1., 1.], [2., 2.], [0., 0.]]
  ```
  """

  def __init__(self,
               embedding_size,
               nslots,
               slot_map_fn=None,
               key_dtype=tf.int64,
               value_dtype=tf.float32,
               combiner='sum',
               initializer=None,
               devices=None,
               name='SlotDynamicEmbeddingLayer',
               **kwargs):
    """
    Create a embedding layer with weights aggregated into feature related slots.

    Args:
      embedding_size: An object convertible to int. Length of embedding vector
        to every feature id.
      nslots: Number of fields. It should be convertible to int.
      slot_map_fn: A element-wise tf.function to map feature id to a field slot.
      key_dtype: Dtype of the embedding keys to weights. Default is int64.
      value_dtype: Dtype of the embedding weight values. Default is float32
      combiner: A string or a function to combine the lookup result. It's value
        could be 'sum', 'mean', 'min', 'max', 'prod', 'std', etc. whose are
        one of tf.math.reduce_xxx.
      initializer: Initialilizer to get the embedding values. Default is
        RandomNormal.
      devices: List of devices to place the embedding layer parameter.
      name: Name of the embedding layer.

      **kwargs:
        trainable: Bool. Whether if the layer is trainable. Default is True.
        bp_v2: Bool. If true, the embedding layer will be updated by incremental
          amount. Otherwise, it will be updated by value directly. Default is
          True.
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
    """

    if not callable(slot_map_fn):
      raise ValueError('slot_map_fn is not callable.')
    self.slot_map_fn = slot_map_fn

    try:
      nslots = int(nslots)
    except:
      raise TypeError('nslots should be convertible to int, but get {}'.format(
          type(nslots)))
    self.nslots = nslots

    super(FieldWiseEmbedding, self).__init__(embedding_size,
                                             key_dtype=key_dtype,
                                             value_dtype=value_dtype,
                                             combiner=combiner,
                                             initializer=initializer,
                                             devices=devices,
                                             name=name,
                                             **kwargs)

  def call(self, ids):
    ids = tf.convert_to_tensor(ids)
    if ids.shape.rank > 2:
      raise NotImplementedError(
          'Input dimension higher than 2 is not implemented yet.')
    flat_ids = tf.reshape(ids, (-1,))
    lookup_result = de.shadow_ops.embedding_lookup(self.shadow, flat_ids)
    embedding = self._pooling_by_slots(lookup_result, ids)
    return embedding

  def _pooling_by_slots(self, lookup_result, ids):
    input_shape = tf.shape(ids)
    batch_size = input_shape[0]
    num_per_sample = input_shape[1]
    slots = self.slot_map_fn(ids)
    bias = tf.reshape(
        tf.range(batch_size, dtype=ids.dtype) * self.nslots, (batch_size, 1))
    bias = tf.tile(bias, (1, num_per_sample))
    slots += bias

    segment_ids = tf.reshape(slots, (-1,))
    sorted_index = tf.argsort(segment_ids)
    segment_ids = tf.sort(segment_ids)
    chosen = tf.range(tf.size(ids), dtype=ids.dtype)
    chosen = tf.gather(chosen, sorted_index)

    combiner = _choose_reduce_method(self.combiner, sparse=True, segmented=True)
    latent = combiner(lookup_result,
                      chosen,
                      segment_ids,
                      num_segments=self.nslots * batch_size)
    latent = tf.reshape(latent, (batch_size, self.nslots, self.embedding_size))
    return latent

  def get_config(self):
    _initializer = self.params.initializer
    if _initializer is None:
      _initializer = tf.keras.initializers.Zeros()
    _max_norm = None
    if isinstance(self.max_norm, tf.keras.constraints.Constraint):
      _max_norm = tf.keras.constraints.serialize(self.max_norm)

    config = {
        'embedding_size': self.embedding_size,
        'nslots': self.nslots,
        'slot_map_fn': self.slot_map_fn,
        'combiner': self.combiner,
        'key_dtype': self.params.key_dtype,
        'value_dtype': self.params.value_dtype,
        'initializer': tf.keras.initializers.serialize(_initializer),
        'devices': self.params.devices,
        'name': self.name,
        'trainable': self.trainable,
        'bp_v2': self.params.bp_v2,
        'restrict_policy': self.params.restrict_policy.__class__,
        'init_capacity': self.params.init_size,
        'partitioner': self.params.partition_fn,
        'kv_creator': self.params.kv_creator,
        'max_norm': _max_norm,
        'distribute_strategy': self.distribute_strategy,
    }
    return config
