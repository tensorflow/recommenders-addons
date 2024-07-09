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

`The file will be introduced as `shadow_ops` under `dynamic_embedding`.
It is a submodule of `dynamic_embedding`.

In TensorFlow 2.x, tf.function is introduced to speedup the computation.
And also modular programming based on [tf.Module](https://www.tensorflow.org/guide/intro_to_modules)
are recommended because of the Pythonic style APIs. But APIs like
`embedding_lookup`, `embedding_lookup_unique`, `embedding_lookup_sparse`, and
`safe_embedding_lookup_sparse` in `dynamic_embedding`, are wrappers of
`embedding_lookup`. And it will create a TrainableWrapper object inside
the function, which doesn't meet the requirements of
[tf.function](https://www.tensorflow.org/guide/function)

The `shadow_ops` submodule is designed to support usage on `tf.function`
and modular style development, like keras.
"""

import tensorflow as tf

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.embedding_weights import EmbeddingWeights, \
  TrainableWrapper

try:  # tf version >= 2.10.0
  from tensorflow.python.trackable import base as trackable
except:
  from tensorflow.python.training.tracking import base as trackable

from tensorflow.python.distribute import distribute_utils


class ShadowVariable(EmbeddingWeights, TrainableWrapper):
  """
  ShadowVariable is a eager persistent twin of TrainableWrapper.

  ShadowVariable maps the sparse domain, which may reside cross multiple
  devices, as a projection on current device. Its value represents the activated
  part of the sparse domain. When lookup on sparse domain, it will fetch the
  lookup result to local, and could be regarded as trainable object to
  optimizers, like an ordinary variable. It supports the modular programming
  and [tf.function](https://www.tensorflow.org/guide/function).
  """

  def __init__(self,
               params,
               name='ShadowVariable',
               max_norm=None,
               trainable=True,
               distribute_strategy=None,
               **kwargs):
    """
    Create a ShadowVariable object.

    Args:
      params: A dynamic_embedding.Variable object represents the sparse domain.
      ids: If set, it needs to be a ResourceVariable, to keep the
        ids for backward computations. Otherwise the ShadowVariable will
        create ids variable buffer itself.
      name: Name of the ShadowVariable.
      max_norm: If not `None`, each values is clipped if its l2-norm is larger
        than this value.
      trainable: Bool. If true, the variable will be treated as trainable.
        Default is true.
      distribute_strategy: DistributeStrategy.

      **kwargs:
        model_mode: ModelMode of the option. Default is ModelMode.CURRENT_SETTING.
          often used internally.
        ids: A Buffer to store the feature ids. If None, it use a private one.
        exists: A Buffer to indicate whether the feature ids exist in sparse domain.
          If None, it use a private one.
    """
    if not context.executing_eagerly():
      raise NotImplementedError('Currently ShadowVariable is only allowed'
                                ' in eager mode.')

    self._name = name
    if not isinstance(params, de.Variable):
      raise TypeError('params must be de.Variable, but get %s' % type(params))
    self.params = params
    collections = kwargs.get('collections', None)
    ids = kwargs.get('ids', None)
    if ids is not None:
      kwargs.pop('ids')
    ids_name = self._name + '-ids'
    if ids is None:
      self.ids = DEResourceVariable((),
                                    trainable=False,
                                    collections=collections,
                                    name=ids_name,
                                    dtype=self.params.key_dtype,
                                    distribute_strategy=distribute_strategy,
                                    shape=tensor_shape.TensorShape(None))
    else:
      if not isinstance(ids, resource_variable_ops.ResourceVariable):
        raise TypeError('If ids is set, it needs to be a ResourceVariable')
      self.ids = ids

    model_mode = kwargs.get('model_mode', None)
    if model_mode:
      kwargs.pop('model_mode')
    else:
      model_mode = de.ModelMode.CURRENT_SETTING
    initial_value = array_ops.zeros(shape=(0, self.params.dim),
                                    dtype=self.params.value_dtype)

    if (distribute_strategy is not None) and (not isinstance(
        distribute_strategy, distribute_lib.StrategyBase)):
      raise TypeError('distribute_strategy must inherit from StrategyBase.')

    super(ShadowVariable,
          self).__init__(self.params,
                         self.ids,
                         max_norm=max_norm,
                         initial_value=initial_value,
                         dtype=self.params.value_dtype,
                         trainable=trainable,
                         collections=collections,
                         model_mode=model_mode,
                         distribute_strategy=distribute_strategy,
                         name=name)
    exists = kwargs.get('exists', None)
    exists_name = self._name + '-exists'
    if exists is None:
      self.exists = DEResourceVariable((),
                                       trainable=False,
                                       collections=collections,
                                       name=exists_name,
                                       dtype=dtypes.bool,
                                       distribute_strategy=distribute_strategy,
                                       shape=tensor_shape.TensorShape(None))
      self._track_trackable(self.exists, exists_name, overwrite=False)
    else:
      self.exists = exists
    self.params._trainable_store[name] = self

  def verify_embedding_weights(self, sparse_ids, sparse_weights=None):
    EmbeddingWeights.verify_embedding_param_weights(self.params, sparse_ids,
                                                    sparse_weights)

  def embedding_lookup(self,
                       ids,
                       name=None,
                       max_norm=None) -> (tf.Tensor, EmbeddingWeights):
    return embedding_lookup(self, ids, name), self

  def prefetch_values(self, update=False):
    if self.params.bp_v2:
      with ops.device(self._handle.device):
        r, exists = self.params.lookup(self.ids, return_exists=True)
        self.exists.assign(exists)
        self.prefetch_values_op = self.transform(r)
    else:
      with ops.device(self._handle.device):
        self.prefetch_values_op = self.transform(self.params.lookup(self.ids))
    return self.prefetch_values_op

  def value(self, do_prefetch=False):
    """A cached operation which reads the value of this variable."""
    if self._cached_value is not None:
      return self._cached_value
    with ops.colocate_with(None, ignore_existing=True):
      return self._read_variable_op(do_prefetch=do_prefetch)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    """
    Assigns a new value to this variable.
    To discriminate with ResourceVariable, the shadow always uses a
    variant space to hold the temporary embedding lookup buffer.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name to use for the assignment.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    # Note: not depending on the cached value here since this can be used to
    # initialize the variable.
    with resource_variable_ops._handle_graph(self.handle):
      value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
      assign_op = gen_resource_variable_ops.assign_variable_op(self.handle,
                                                               value_tensor,
                                                               name=name)
      if read_value:
        return self._lazy_read(assign_op)
    return assign_op

  def _reset_ids(self, ids):
    return self.ids.assign(ids, use_locking=True)

  def _gather_saveables_for_checkpoint(self):
    self._reset_ids(array_ops.zeros((0,), dtype=self.params.key_dtype))
    self.read_value(do_prefetch=True)
    for s in self._tracked_slots:
      s.read_value(do_prefetch=True)
    return {trackable.VARIABLE_VALUE_KEY: self}


def embedding_lookup(
    shadow: ShadowVariable,
    ids,
    partition_strategy=None,  # pylint: disable=unused-argument
    name=None,
    validate_indices=None,  # pylint: disable=unused-argument
):
  """
  Shadow version of dynamic_embedding.embedding_lookup. It use existed shadow
  variable to to embedding lookup, and store the result. No by-product will
  be introduced in this call. So it can be decorated by `tf.function`.

  Args:
    shadow: A ShadowVariable object.
    ids: A tensor with any shape as same dtype of params.key_dtype.
    partition_strategy: No used, for API compatiblity with `nn.emedding_lookup`.
    name: A name for the operation.
    validate_indices: No used, just for compatible with nn.embedding_lookup .

  Returns:
    A tensor with shape [shape of ids] + [dim],
      dim is equal to the value dim of params.
      containing the values from the params tensor(s) for keys in ids.
  """
  ids = ops.convert_to_tensor(ids)

  if distribute_utils.is_distributed_variable(shadow):
    shadow_ = shadow._get_on_device_or_primary()
  else:
    shadow_ = shadow
  if shadow_.ids.dtype != ids.dtype:
    raise ValueError('{} ids is not matched with ShadowVariable with ids'
                     ' {},'.format(ids.dtype, shadow_.ids.dtype))

  with ops.name_scope(name, "shadow_embedding_lookup"):
    with ops.colocate_with(None, ignore_existing=True):
      if de.ModelMode.CURRENT_SETTING == de.ModelMode.TRAIN:
        with ops.control_dependencies([shadow_._reset_ids(ids)]):
          result = shadow_.read_value(do_prefetch=True)
      else:
        result = shadow_.params.lookup(ids)

      return result


def embedding_lookup_unique_base(ids,
                                 embedding_size,
                                 lookup_function,
                                 with_unique=True,
                                 name=None):
  """
  Helper function to perform embedding lookup with optional uniqueness and ragged tensor support.

  Args:
    ids: A tensor or a tf.RaggedTensor containing the ids for which to lookup embeddings.
    embedding_size: Size of each embedding.
    lookup_function: Function to be used for the lookup, must accept a single argument (ids).
    with_unique: Whether to use unique ids to lookup embeddings.
    name: Optional name for the operation.

  Returns:
    A tensor or a tf.RaggedTensor containing the embeddings corresponding to ids.
  """
  is_ragged = isinstance(ids, tf.RaggedTensor)

  if is_ragged:
    original_structure = ids
    ids = ids.flat_values
  else:
    ids = tf.convert_to_tensor(ids)

  input_shape = tf.shape(ids)
  embeddings_shape = tf.concat([input_shape, [embedding_size]], 0)

  ids_flat = tf.reshape(ids, (-1,))
  if with_unique:
    with ops.name_scope(name, "EmbeddingWithUnique"):
      unique_ids, idx = tf.unique(ids_flat)
      unique_embeddings = lookup_function(unique_ids)
      embeddings_flat = tf.gather(unique_embeddings, idx)
  else:
    embeddings_flat = lookup_function(ids_flat)

  embeddings = tf.reshape(embeddings_flat, embeddings_shape)

  if is_ragged:
    embeddings = tf.RaggedTensor.from_row_lengths(
        embeddings, original_structure.row_lengths())

  return embeddings


def embedding_lookup_unique(
    shadow,
    ids,
    embedding_size,
    with_unique=True,
    name=None,
):
  """
  unify version of embedding_lookup. It handles ragged tensor, unique and shape. No by-product will
  be introduced in this call. So it can be decorated by `tf.function`.

  Args:
    shadow: A ShadowVariable object.
    ids: A tensor with any shape as same dtype of params.key_dtype.
    embedding_size: The size of embedding, used in shape the output
    with_unique: If True, it will use unique ids to lookup embedding.
    name: A name for the operation.

  Returns:
    A tensor with shape [shape of ids] + [embedding_size],
      containing the values from the params tensor(s) for keys in ids.
  """

  return embedding_lookup_unique_base(ids, embedding_size,
                                      lambda x: embedding_lookup(shadow, x),
                                      with_unique, name)


class DEResourceVariable(resource_variable_ops.ResourceVariable):

  def __init__(self, *args, **kwargs):
    super(DEResourceVariable, self).__init__(*args, **kwargs)


class HvdVariable(EmbeddingWeights):

  def __init__(
      self,
      name,
      shadow,
      embedding_size,
      with_unique=True,
      with_secondary_unique=True,
      mpi_size=None,
  ):
    self.name = name
    self.embedding_size = embedding_size
    self.shadow = shadow
    try:
      import horovod.tensorflow as hvd
    except ImportError:
      raise ValueError(
          "Please install Horovod properly first if you want to use distributed synchronous training based on Horovod"
      )
    self.hvd = hvd
    self.with_unique = with_unique
    self.with_secondary_unique = with_secondary_unique
    if mpi_size is None:
      self._mpi_size = self.hvd.size()
    else:
      self._mpi_size = mpi_size

  def verify_embedding_weights(self, sparse_ids, sparse_weights=None):
    EmbeddingWeights.verify_embedding_param_weights(self.shadow.params,
                                                    sparse_ids, sparse_weights)

  def __relocate_dense_feature__(self, ids):
    """
    Args:
      ids: A 2-D Tensor with shape: (batch_size, sequence_length) or a 1-D Tensor with shape: (batch_size,).
        If batch_size is provided, then it trust the batch_size argument, to avoid new an OP instead.
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

    partition_index = self.shadow.params.partition_fn(ids, self._mpi_size)
    from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import make_partition
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

    reloc_ids, remote_sizes, gather_indices = self.__relocate_dense_feature__(
        ids)

    if self.with_secondary_unique:
      with tf.name_scope(self.name + "/EmbeddingWithUnique"):
        reloc_unique_ids, reloc_unique_idx = tf.unique(reloc_ids)
        reloc_unique_embeddings = de.shadow_ops.embedding_lookup(
            self.shadow, reloc_unique_ids)
        lookup_result = tf.gather(reloc_unique_embeddings, reloc_unique_idx)
    else:
      lookup_result = de.shadow_ops.embedding_lookup(self.shadow, reloc_ids)
    lookup_result, _ = self.hvd.alltoall(lookup_result, splits=remote_sizes)

    input_shape = tf.shape(ids)
    recover_shape = tf.concat((input_shape, (self.embedding_size,)), axis=0)
    gather_indices = tf.expand_dims(tf.concat(gather_indices, axis=0), axis=-1)
    lookup_result = tf.scatter_nd(gather_indices, lookup_result, recover_shape)
    return lookup_result

  def embedding_lookup(self,
                       ids,
                       name=None,
                       max_norm=None) -> (tf.Tensor, EmbeddingWeights):
    return self.__alltoall_embedding_lookup__(ids), self.shadow
