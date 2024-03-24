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

import pickle

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.ops import init_ops
from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops import dynamic_embedding_variable as devar

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.keras.utils import tf_utils
try:  # tf version >= 2.14.0
  from tensorflow.python.distribute import distribute_lib as distribute_ctx
  assert hasattr(distribute_ctx, 'has_strategy')
except:
  from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import values_util
from tensorflow.python.framework import ops
from tensorflow.python.eager import tape
from tensorflow.python.ops.variables import VariableAggregation
from tensorflow.python.platform import tf_logging
try:  # The data_structures has been moved to the new package in tf 2.11
  from tensorflow.python.trackable import data_structures
except:
  from tensorflow.python.training.tracking import data_structures

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import DistributedVariableWrapper, TrainableWrapperDistributedPolicy
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import make_partition
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.tf_save_restore_patch import de_fs_saveable_class_names


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


class Embedding(tf.keras.layers.Layer):
  """
  A keras style Embedding layer. The `Embedding` layer acts same like
  [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding),
  except that the `Embedding` has dynamic embedding space so it does
  not need to set a static vocabulary size, and there will be no hash conflicts
  between features.

  The embedding layer allows arbitrary input shape of feature ids, and get
  (shape(ids) + embedding_size) lookup result. Normally the first dimension
  is batch_size.

  ### Example
  ```python
  embedding = dynamic_embedding.keras.layers.Embedding(8) # embedding size 8
  ids = tf.constant([[15,2], [4,92], [22,4]], dtype=tf.int64) # (3, 2)
  out = embedding(ids) # lookup result, (3, 2, 8)
  ```

  You could inherit the `Embedding` class to implement a custom embedding
  layer with other fixed shape output.

  TODO(Lifann) Currently the Embedding only implemented in eager mode
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
               with_unique=True,
               **kwargs):
    """
    Creates an Embedding layer.

    Args:
      embedding_size: An object convertible to int. Length of embedding vector
        to every feature id.
      key_dtype: Dtype of the embedding keys to weights. Default is int64.
      value_dtype: Dtype of the embedding weight values. Default is float32
      combiner: A string or a function to combine the lookup result. Its value
        could be 'sum', 'mean', 'min', 'max', 'prod', 'std', etc. whose are
        one of tf.math.reduce_xxx.
      initializer: Initializer to the embedding values. Default is RandomNormal.
      devices: List of devices to place the embedding layer parameter.
      name: Name of the embedding layer.
      with_unique: Bool. Whether if the layer does unique on `ids`. Default is True.

      **kwargs:
        trainable: Bool. Whether if the layer is trainable. Default is True.
        bp_v2: Bool. If true, the embedding layer will be updated by incremental
          amount. Otherwise, it will be updated by value directly. Default is
          False.
        restrict_policy: A RestrictPolicy class to restrict the size of
          embedding layer parameter since the dynamic embedding supports
          nearly infinite embedding space capacity.
        init_capacity: Integer. Initial number of kv-pairs in an embedding
          layer. The capacity will grow if the used space exceeded current
          capacity.
        partitioner: A function to route the keys to specific devices for
          distributed embedding parameter.
        kv_creator: A KVCreator object to create external KV storage as
          embedding parameter.
        max_norm: If not `None`, each value is clipped if its l2-norm is larger
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

    self.embedding_size = embedding_size
    self.combiner = combiner
    if initializer is None:
      initializer = tf.keras.initializers.RandomNormal()
    partitioner = kwargs.get('partitioner', devar.default_partition_fn)
    trainable = kwargs.get('trainable', True)
    self.max_norm = kwargs.get('max_norm', None)
    self.keep_distribution = kwargs.get('keep_distribution', False)
    self.with_unique = with_unique

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
                                    checkpoint=kwargs.get('checkpoint', True),
                                    init_size=kwargs.get('init_capacity', 0),
                                    kv_creator=kwargs.get('kv_creator', None),
                                    restrict_policy=kwargs.get(
                                        'restrict_policy', None),
                                    bp_v2=kwargs.get('bp_v2', False))

      self.distribute_strategy = kwargs.get('distribute_strategy', None)
      shadow_name = name + '-shadow' if name else 'ShadowVariable'
      if distribute_ctx.has_strategy():
        self.distribute_strategy = distribute_ctx.get_strategy()
      if self.distribute_strategy:
        strategy_devices = self.distribute_strategy.extended.worker_devices
        self.shadow_impl = tf_utils.ListWrapper([])
        for i, strategy_device in enumerate(strategy_devices):
          with ops.device(strategy_device):
            shadow_name_replica = shadow_name
            if i > 0:
              shadow_name_replica = "%s/replica_%d" % (shadow_name, i)
            with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
              self.shadow_impl.as_list().append(
                  de.shadow_ops.ShadowVariable(
                      self.params,
                      name=shadow_name_replica,
                      max_norm=self.max_norm,
                      trainable=trainable,
                      distribute_strategy=self.distribute_strategy))
      else:
        self.shadow_impl = tf_utils.ListWrapper([
            de.shadow_ops.ShadowVariable(self.params,
                                         name=shadow_name,
                                         max_norm=self.max_norm,
                                         trainable=trainable)
        ])
    if len(self.shadow_impl.as_list()) > 1:
      self._current_ids = data_structures.NoDependency(
          [shadow_i.ids for shadow_i in self.shadow_impl.as_list()])
      self._current_exists = data_structures.NoDependency(
          [shadow_i.exists for shadow_i in self.shadow_impl.as_list()])
      self.optimizer_vars = data_structures.NoDependency(
          [shadow_i._optimizer_vars for shadow_i in self.shadow_impl.as_list()])
    else:
      self._current_ids = data_structures.NoDependency(
          self.shadow_impl.as_list()[0].ids)
      self._current_exists = data_structures.NoDependency(
          self.shadow_impl.as_list()[0].exists)
      self.optimizer_vars = self.shadow_impl.as_list()[0]._optimizer_vars
    if distribute_ctx.has_strategy(
    ) and self.distribute_strategy and 'OneDeviceStrategy' not in str(
        self.distribute_strategy) and not values_util.is_saving_non_distributed(
        ) and values_util.get_current_replica_id_as_int() is not None:
      self.shadow = DistributedVariableWrapper(
          self.distribute_strategy, self.shadow_impl.as_list(),
          VariableAggregation.NONE,
          TrainableWrapperDistributedPolicy(VariableAggregation.NONE))
    else:
      self.shadow = self.shadow_impl.as_list()[0]
    self.params._created_in_class = self  # To facilitate access to the primitive class through params
    super(Embedding, self).__init__(name=name,
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
    embeddings_shape = tf.concat([input_shape, [self.embedding_size]], 0)
    ids_flat = tf.reshape(ids, (-1,))
    if self.with_unique:
      with tf.name_scope(self.name + "/EmbeddingWithUnique"):
        unique_ids, idx = tf.unique(ids_flat)
        unique_embeddings = de.shadow_ops.embedding_lookup(
            self.shadow, unique_ids)
        embeddings_flat = tf.gather(unique_embeddings, idx)
    else:
      embeddings_flat = de.shadow_ops.embedding_lookup(self.shadow, ids_flat)
    embeddings = tf.reshape(embeddings_flat, embeddings_shape)
    return embeddings

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
    }
    return config


"""
  For matching the original name space of `tf.keras.layers.BasicEmbedding`.
"""
BasicEmbedding = Embedding


class SquashedEmbedding(Embedding):
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
    lookup_result = super(SquashedEmbedding, self).call(ids)
    embedding = reduce_pooling(lookup_result, combiner=self.combiner)
    return embedding


class FieldWiseEmbedding(Embedding):
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
               with_unique=True,
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
      with_unique: Bool. Whether if the layer does unique on `ids`. Default is True.

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
                                             with_unique=with_unique,
                                             **kwargs)

  def call(self, ids):
    ids = tf.convert_to_tensor(ids)
    if ids.shape.rank > 2:
      raise NotImplementedError(
          'Input dimension higher than 2 is not implemented yet.')
    flat_ids = tf.reshape(ids, (-1,))
    lookup_result = super(FieldWiseEmbedding, self).call(flat_ids)
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


class HvdAllToAllEmbedding(BasicEmbedding):
  """
  This embedding layer will dispatch keys to all corresponding Horovod workers and receive its own keys for distributed training before embedding_lookup.
  """

  def __init__(self,
               with_unique=True,
               with_secondary_unique=True,
               mpi_size=None,
               batch_size=None,
               *args,
               **kwargs):
    try:
      import horovod.tensorflow as hvd
    except ImportError:
      raise ValueError(
          "Please install Horovod properly first if you want to use distributed synchronous training based on Horovod"
      )
    self.hvd = hvd
    self.with_unique = with_unique
    self.with_secondary_unique = with_secondary_unique
    self.batch_size = batch_size
    if mpi_size is None:
      self._mpi_size = self.hvd.size()
    else:
      self._mpi_size = mpi_size
    super(HvdAllToAllEmbedding, self).__init__(*args, **kwargs)
    try:
      assert type(self.params.saveable).__name__ in de_fs_saveable_class_names
    except:
      tf_logging.warning(
          "Please use FileSystemSaver in KVCreator when use HvdAllToAllEmbedding. "
          "It will allow TFRA save and restore KV files when Embedding tensor parallel in distributed training. "
      )

  def __relocate_dense_feature__(self, ids, batch_size=None):
    """
    Args:
      ids: A 2-D Tensor with shape: (batch_size, sequence_length) or a 1-D Tensor with shape: (batch_size,). 
        If batch_size is provided, then it trust the batch_size argument, to avoid new an OP instead.
      batch_size: Integer or a int32/int64 scalar. All ranks must have same batch_size.
        Otherwise will make undefined behavior.

    Returns:
      flat_reloc_ids: a flat ids partitioned to each rank.
    """
    if ids.dtype not in (tf.int32, tf.int64):
      raise NotImplementedError

    if ids.shape.rank > 2:
      raise NotImplementedError(
          'Input ids must be shape '
          f'(batch_size, sequence_length) or (batch_size,), but get {ids.shape}'
      )

    if batch_size is None:
      input_shape = tf.shape(ids)
      batch_size = input_shape[0]

    partition_index = self.params.partition_fn(ids, self._mpi_size)
    ids_partitions, ids_indices = make_partition(ids, partition_index,
                                                 self._mpi_size)
    partitions_sizes = tf.stack([tf.size(p) for p in ids_partitions], axis=0)
    relocs_tensor = tf.concat(ids_partitions, axis=0)
    flat_reloc_ids, remote_sizes = self.hvd.alltoall(relocs_tensor,
                                                     splits=partitions_sizes)
    return flat_reloc_ids, remote_sizes, ids_indices

  def __alltoall_embedding_lookup__(self, ids):
    if self._mpi_size == 1:
      return de.shadow_ops.embedding_lookup(self.shadow, ids)
    if isinstance(ids, tf.sparse.SparseTensor):
      raise NotImplementedError('SparseTensor is not supported yet.')

    input_shape = tf.shape(ids)
    if self.batch_size is None:
      batch_size_runtime = input_shape[0]

    reloc_ids, remote_sizes, gather_indices = self.__relocate_dense_feature__(
        ids, batch_size=batch_size_runtime)

    if self.with_secondary_unique:
      with tf.name_scope(self.name + "/EmbeddingWithUnique"):
        reloc_unique_ids, reloc_unique_idx = tf.unique(reloc_ids)
        reloc_unique_embeddings = de.shadow_ops.embedding_lookup(
            self.shadow, reloc_unique_ids)
        lookup_result = tf.gather(reloc_unique_embeddings, reloc_unique_idx)
    else:
      lookup_result = de.shadow_ops.embedding_lookup(self.shadow, reloc_ids)
    lookup_result, _ = self.hvd.alltoall(lookup_result, splits=remote_sizes)

    recover_shape = tf.concat((input_shape, (self.embedding_size,)), axis=0)
    gather_indices = tf.expand_dims(tf.concat(gather_indices, axis=0), axis=-1)
    lookup_result = tf.scatter_nd(gather_indices, lookup_result, recover_shape)
    return lookup_result

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
    ids_flat = tf.reshape(ids, (-1,))
    if self.with_unique:
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = self.__alltoall_embedding_lookup__(unique_ids)
      lookup_result = tf.gather(unique_embeddings, idx)
    else:
      lookup_result = self.__alltoall_embedding_lookup__(ids_flat)
    lookup_result = tf.reshape(
        lookup_result, tf.concat([input_shape, [self.embedding_size]], 0))
    return lookup_result

  def get_config(self):
    config = super(HvdAllToAllEmbedding, self).get_config()
    config.update({"with_unique": self.with_unique})
    config.update({"mpi_size": self._mpi_size})
    config.update({"batch_size": self.batch_size})
    return config
