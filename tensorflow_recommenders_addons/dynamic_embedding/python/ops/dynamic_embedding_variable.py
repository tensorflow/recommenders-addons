# Copyright 2020 The TensorFlow Recommenders-Addons Authors.
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

import functools
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.utils.check_platform import is_macos, is_arm64

try:
  from tensorflow.python.util import _pywrap_util_port as pywrap
except:
  try:
    from tensorflow.python import _pywrap_util_port as pywrap
  except:
    from tensorflow.python import pywrap_tensorflow as pywrap

try:
  from tensorflow.python.keras.initializers import initializers_v2 as kinit2
except ImportError:
  kinit2 = None
  pass  # for compatible with TF < 2.3.x

from tensorflow.python.client import device_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import python_state
from tensorflow.python.util.tf_export import tf_export


def make_partition(data, partition_index, shard_num):
  """
  Shard keys to shard_num partitions

  Args:
    data: keys or values, usually the IDs of dynamic features.
    partition_index: partitions index.
    shard_num: partition number
  Returns:
    a pair of tensor: (partition result, partition indices)
  """
  if shard_num <= 1:
    return [
        data,
    ], None
  with ops.colocate_with(data, ignore_existing=True):
    partitions = data_flow_ops.dynamic_partition(data, partition_index,
                                                 shard_num)
    indices = data_flow_ops.dynamic_partition(
        math_ops.range(array_ops.shape(data)[0]),
        math_ops.cast(partition_index, dtypes.int32),
        shard_num,
    )
  return partitions, indices


def _stitch(values, indices):
  if len(values) == 1:
    return values[0]
  with ops.colocate_with(indices[0], ignore_existing=True):
    all_values = data_flow_ops.dynamic_stitch(indices, values)
  return all_values


def default_partition_fn(keys, shard_num):
  """The default partition function.
    partition keys by "mod" strategy.

    keys: a tensor presents the keys to be partitioned.
    shard_num: the num of partitions
  Returns:
    a tensor with same shape as keys with type of `tf.int32`,
      represents the corresponding partition-ids of keys.
  """
  keys_op = ops.convert_to_tensor(keys, name="keys")
  gpu_mode = pywrap.IsGoogleCudaEnabled()

  with ops.colocate_with(keys_op):
    if keys_op.dtype == dtypes.int64 and gpu_mode:
      # This branch has low performance on some multi-CPU scenario,
      # so we try to use default branch when GPUs are not available.
      mask = constant_op.constant(0x7fffffff, dtypes.int64)
      keys_int32 = math_ops.cast(bitwise_ops.bitwise_and(keys_op, mask),
                                 dtypes.int32)
      mod = math_ops.mod(keys_int32,
                         constant_op.constant(shard_num, dtypes.int32))
      ids = math_ops.cast(mod, dtype=dtypes.int32)
    elif keys_op.dtype == dtypes.string:
      ids = string_ops.string_to_hash_bucket_fast(keys_op, shard_num)
      mask = constant_op.constant(0x7fffffff, dtypes.int64)
      ids = math_ops.cast(bitwise_ops.bitwise_and(ids, mask), dtypes.int32)
    else:
      ids = math_ops.cast(math_ops.mod(keys_op, shard_num), dtype=dtypes.int32)
  return ids


class GraphKeys(object):
  """
  (Deprecated) extended standard names related to
  `dynamic_embedding_ops.Variable` to use for graph collections.
  The following standard keys are defined:
  * `DYNAMIC_EMBEDDING_VARIABLES`: the default collection of
    all `dynamic_embedding_ops.Variable` objects.
  * `TRAINABLE_DYNAMIC_EMBEDDING_VARIABLES`: the subset of
    `dynamic_embedding_ops.Variable` that is trainable.
  """
  tf_logging.warn(
      'dynamic_embedding.GraphKeys has already been deprecated. '
      'The Variable will not be added to collections because it '
      'does not actully own any value, but only a holder of tables, '
      'which may lead to import_meta_graph failed since non-valued '
      'object has been added to collection. If you need to use '
      '`tf.compat.v1.train.Saver` and access all Variables from '
      'collection, you could manually add it to the collection by '
      'tf.compat.v1.add_to_collections(names, var) instead.')
  # Dynamic embedding variables.
  DYNAMIC_EMBEDDING_VARIABLES = "dynamic_embedding_variables"
  # Trainable dynamic embedding variables.
  TRAINABLE_DYNAMIC_EMBEDDING_VARIABLES = "trainable_dynamic_embedding_variables"


class Variable(base.Trackable):
  """
  A Distributed version of HashTable(reference from lookup_ops.MutableHashTable)
  It is designed to dynamically store the Sparse Weights(Parameters) of DLRMs.
  """

  def __init__(
      self,
      key_dtype=dtypes.int64,
      value_dtype=dtypes.float32,
      dim=1,
      devices=None,
      partitioner=default_partition_fn,
      shared_name=None,
      name="DynamicEmbedding_Variable",
      initializer=None,
      trainable=True,
      checkpoint=True,
      init_size=0,
      kv_creator=None,
      restrict_policy=None,
      bp_v2=False,
  ):
    """Creates an empty `Variable` object.

        Creates a group of tables placed on devices specified by `devices`,
        and the device placement mechanism of TensorFlow will be ignored,
        the type of its keys and values are specified by key_dtype
        and value_dtype, respectively.
        The environment variables 'TF_HASHTABLE_INIT_SIZE' can be used to set the
        inital size of each tables, which can help reduce rehash times.
        The default initial table size is 8,192

        Args:
          key_dtype: the type of the key tensors.
          value_dtype: the type of the value tensors.
          dim: the length of the value array for each key,
            on GPUs, `dim` should be less or equal to 200.
          devices: the list of devices holding the tables.
            One table will be created on each device. By default, `devices` is
            ['/CPU:0'] and when GPU is available, `devices` is ['/GPU:0']
          partitioner: partition function of keys,
            return the partition index for each key.

          Example partition func:
          ```python
          def default_partition_fn(keys, shard_num):
            return tf.cast(keys % shard_num, dtype=tf.int32)
          ```
          shared_name: No used.
          name: A name for the operation (optional).
          initializer: The value to use if a key is missing in the hash table.
            which can be a python number, numpy array or `tf.initializer` instances.
            If initializer is `None` (the default), `0` will be taken.
          trainable: Bool. If true, the variable will be treated as a trainable.
            Default is true.
          checkpoint: if True, the contents of the SparseVariable are
            saved to and restored from checkpoints.
            If `shared_name` is empty for a checkpointed table,
            it is shared using the table node name.
          init_size: initial size for the Variable and initial size of each hash
            tables will be int(init_size / N), N is the number of the devices.
          restrict_policy: a restrict policy to specify the rule to restrict the
            size of variable. If in training program, the variable is updated by
            optimizer, then the sparse slot variables in optimizer are also be
            restricted.
          bp_v2: By default with `bp_v2=False`, the optimizer will update
            dynamic embedding values by *setting* (key, value) after
            `optimizer.apply_gradient`. If one key is used by multiple workers
            at the same time, only one of them will be seen, while the others are
            overwritten. By setting `bp_v2=True`, the optimizer will update
            parameters by *adding delta* instead of *setting*, which solves the
            race condition problem among workers during backpropagation in
            large-scale distributed asynchronous training.

        Returns:
          A `Variable` object.
    """
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.dim = dim
    self.bp_v2 = bp_v2

    def _get_default_devices():
      gpu_list = [
          x.name
          for x in device_lib.list_local_devices()
          if x.device_type == "GPU"
      ]
      return gpu_list[0:1] or [
          "/CPU:0",
      ]

    devices_ = devices or _get_default_devices()
    self.devices = (devices_ if isinstance(devices_, list) else [
        devices,
    ])
    self.partition_fn = partitioner
    self.name = name
    self.shared_name = shared_name or "shared_name.{}".format(name)

    self.initializer = None

    self.trainable = trainable
    self.checkpoint = checkpoint

    self._tables = data_structures.ListWrapper([])
    self._track_trackable(self._tables,
                          'tables_of_{}'.format(self.name),
                          overwrite=True)
    self.size_ops = []
    self._trainable_store = {}
    self.kv_creator = kv_creator if kv_creator else de.CuckooHashTableCreator()

    self.shard_num = len(self.devices)

    self.init_size = int(init_size)

    if restrict_policy is not None:
      if not issubclass(restrict_policy, de.RestrictPolicy):
        raise TypeError('restrict_policy must be subclass of RestrictPolicy.')
      self._restrict_policy = restrict_policy(self)
    else:
      self._restrict_policy = None

    valid_dtype_list = [[dtypes.int64, dtypes.float32],
                        [dtypes.int64, dtypes.half],
                        [dtypes.int64, dtypes.bfloat16],
                        [dtypes.int64,
                         dtypes.int32], [dtypes.int64, dtypes.int8],
                        [dtypes.int64, dtypes.int64],
                        [dtypes.int64, dtypes.float64],
                        [dtypes.int64, dtypes.string],
                        [dtypes.int32, dtypes.float32],
                        [dtypes.int32, dtypes.int32],
                        [dtypes.int32, dtypes.float64],
                        [dtypes.string, dtypes.float32],
                        [dtypes.string, dtypes.half],
                        [dtypes.string, dtypes.bfloat16],
                        [dtypes.string, dtypes.int32],
                        [dtypes.string, dtypes.int8],
                        [dtypes.string, dtypes.int64],
                        [dtypes.string, dtypes.float64],
                        [dtypes.string, dtypes.bool]]
    if "GPU" in self.devices[0].upper():
      valid_dtype_list = [
          [dtypes.int64, dtypes.float32],
          [dtypes.int64, dtypes.half],
          [dtypes.int64, dtypes.int32],
          [dtypes.int64, dtypes.int8],
          [dtypes.int64, dtypes.int64],
          [dtypes.int32, dtypes.float32],
      ]
    if is_macos() and is_arm64():
      if value_dtype == dtypes.half or value_dtype == dtypes.bfloat16:
        raise TypeError("""
          float16 and bfloat16 value dtypes are not supported on macOS with ARM64 architecture. Please try another type.
          """)
    if [key_dtype, value_dtype] not in valid_dtype_list:
      raise TypeError(
          "key-value dtype ({}-{}) is not support! The valid dtypes are \n{}\n".
          format(key_dtype, value_dtype, valid_dtype_list))

    _initializer = initializer
    if _initializer is None:
      _initializer = init_ops.zeros_initializer(dtype=self.value_dtype)
    static_default_value = self._convert_anything_to_init(_initializer, dim)
    scope_name = self.name.split("/")[-1]
    with ops.name_scope(scope_name, "DynamicEmbedding_Variable"):
      with ops.colocate_with(None, ignore_existing=True):
        for idx in range(len(self.devices)):
          with ops.device(self.devices[idx]):
            mht = None
            if not issubclass(self.kv_creator.__class__, de.KVCreator):
              raise TypeError("config should be instance of 'config', but got ",
                              str(type(self.kv_creator)))
            mht = self.kv_creator.create(
                key_dtype=self.key_dtype,
                value_dtype=self.value_dtype,
                default_value=static_default_value,
                name=self._make_name(idx),
                checkpoint=self.checkpoint,
                init_size=int(self.init_size / self.shard_num),
                device=self.devices[idx],
            )
            self._tables.append(mht)

  @property
  def tables(self):
    return self._tables

  @property
  def restrict_policy(self):
    return self._restrict_policy

  def _convert_anything_to_init(self, raw_init, dim):
    init = raw_init
    valid_list = [
        init_ops.Initializer, init_ops_v2.Initializer,
        tf.keras.initializers.Initializer
    ]
    if kinit2 is not None:
      valid_list.append(kinit2.Initializer)
    valid_list = tuple(valid_list)
    while callable(init):
      if isinstance(init, valid_list):
        self.initializer = init
        init = init(shape=[dim])
      else:
        try:
          init = init(shape=[1])
        except:
          init = init()
    try:
      init_dims = init.get_shape().as_list()
      init_dims_mul = 1
      for init_d in init_dims:
        init_dims_mul = init_dims_mul * init_d
      if init_dims_mul == dim:
        init = array_ops.reshape(init, [dim])
      else:
        raise ValueError
    except:
      init = array_ops.fill([dim], array_ops.reshape(init, [-1])[0])
    init = math_ops.cast(init, dtype=self.value_dtype)
    return init

  def _make_name(self, table_idx):
    return "{}_mht_{}of{}".format(self.name.replace("/", "_"), table_idx + 1,
                                  self.shard_num)

  def upsert(self, keys, values, name=None):
    """Insert or Update `keys` with `values`.

        If key exists already, value will be updated.

        Args:
          keys: Keys to insert. Can be a tensor of any shape. Must match the table's
            key type.
          values: Values to be associated with keys.Must be a tensor of
            arrays with same shape as `keys` and match the table's value type.
          name: A name for the operation (optional).

        Returns:
          The created Operation.

        Raises:
          TypeError: when `keys` or `values` doesn't match the table data
            types.
    """

    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, _ = make_partition(keys, partition_index, self.shard_num)
    values_partitions, _ = make_partition(values, partition_index,
                                          self.shard_num)

    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.append(self._tables[idx].insert(keys_partitions[idx],
                                             values_partitions[idx],
                                             name=name))

    return control_flow_ops.group(ops_)

  def accum(self, keys, old_values, new_values, exists, name=None):
    """
    Insert `keys` with `values` if not exist, or accumulate a delta value
      `new_values - old_values` to 'keys'.
    This API will help relieve stale gradient problem in asynchronous training.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match
        the table's key type.
      old_values: old values to be associated with keys. Must be a tensor of
        arrays with same shape as `keys` and match the table's value type.
      new_values: new values to be associated with keys. Must be a tensor of
        arrays with same shape as `keys` and match the table's value type.
      exists: A bool type tensor indicates if keys existed or not.
        Must be a tensor of the same shape as `keys`.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data types.
    """
    exists = ops.convert_to_tensor(exists, dtypes.bool, name="original_exists")
    exists = array_ops.reshape(exists, shape=[-1, 1])
    exists_expanded = array_ops.repeat(exists, axis=-1, repeats=self.dim)
    exists_expanded = array_ops.reshape(exists_expanded,
                                        shape=array_ops.shape(old_values))
    values_or_deltas = array_ops.where(exists_expanded,
                                       new_values - old_values,
                                       new_values,
                                       name="values_or_deltas")
    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, _ = make_partition(keys, partition_index, self.shard_num)
    values_or_deltas_partitions, _ = make_partition(values_or_deltas,
                                                    partition_index,
                                                    self.shard_num)
    exists_partitions, _ = make_partition(exists, partition_index,
                                          self.shard_num)

    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.append(self._tables[idx].accum(keys_partitions[idx],
                                            values_or_deltas_partitions[idx],
                                            exists_partitions[idx],
                                            name=name))

    return control_flow_ops.group(ops_)

  def restrict(self, num_reserved, **kwargs):
    """
    Restrict the size of self, also including features reside in commensal
    slots, and the policy status. The restriction rule follow the setting
    in `restrict_policy`.

    Args:
      num_reserved: int. Number of remaining features after restriction.
      **kwargs: keyword arguments passing to `restrict_policy.apply_restriction`.

    Returns:
      An operation to restrict size of the variable itself. Return None if
      the restrict policy is not set.
    """
    if self._restrict_policy is not None:
      return self._restrict_policy.apply_restriction(num_reserved, **kwargs)
    else:
      tf_logging.warning('Call restrict without setting restrict policy.')
      return None

  def remove(self, keys, name=None):
    """Removes `keys` and its associated values from the variable.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, _ = make_partition(keys, partition_index, self.shard_num)

    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.append(self._tables[idx].remove(keys_partitions[idx], name=name))

    return control_flow_ops.group(ops_)

  def clear(self, name=None):
    """clear all keys and values in the table.

        Args:
          name: A name for the operation (optional).

        Returns:
          The created Operation.
        """
    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.append(self._tables[idx].clear(name=name))
    return control_flow_ops.group(ops_)

  def _create_default_values_by_initializer(self, keys):
    if self.initializer is None:
      return None
    try:
      keys_shape = array_ops.shape(array_ops.reshape(keys, [-1]))
      vals_shape = [keys_shape[0], self.dim]
      init_op = self.initializer(vals_shape)
    except Exception as e:  # constant.initializer
      init_op = self.initializer([self.dim])
      tf_logging.warn(
          "Variable [{}] is not running on full-size initialization mode: {}".
          format(str(self.name), str(e)))
    return init_op

  def lookup(self, keys, return_exists=False, name=None):
    """
    Looks up `keys` in a Variable, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      return_exists: if True, will return a additional Tensor which indicates
        if keys are existing in the table.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.
      exists:
        A bool type Tensor of the same shape as `keys` which indicates
          if keys are existing in the table.
          Only provided if `return_exists` is True.
    """
    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, keys_indices = make_partition(keys, partition_index,
                                                   self.shard_num)

    _values = []
    _exists = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        dynamic_default_values = self._create_default_values_by_initializer(
            keys_partitions[idx])
        if dynamic_default_values is not None:
          dynamic_default_values = math_ops.cast(dynamic_default_values,
                                                 self.value_dtype)

        ops_ = None
        ops_ = self._tables[idx].lookup(
            keys_partitions[idx],
            dynamic_default_values=dynamic_default_values,
            return_exists=return_exists,
            name=name,
        )
        if return_exists:
          _values.append(ops_[0])
          _exists.append(ops_[1])
        else:
          _values.append(ops_)

    if return_exists:
      result = (_stitch(_values, keys_indices), _stitch(_exists, keys_indices))
    else:
      result = _stitch(_values, keys_indices)
    return result

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    full_keys = []
    full_values = []
    for idx in range(len(self.devices)):
      keys_ = None
      vals_ = None
      with ops.device(self.devices[idx]):
        keys_, vals_ = self._tables[idx].export(name=name)
        full_keys.append(keys_)
        full_values.append(vals_)
    return array_ops.concat(full_keys, 0), array_ops.concat(full_values, 0)

  def size(self, index=None, name=None):
    """Compute the number of elements in the index-th table of this Variable.

    If index is none, the total size of the Variable wil be return.

    Args:
      index: The index of table (optional)
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this Variable.
    """
    if context.executing_eagerly():
      self.size_ops = []
    if not self.size_ops:
      for idx in range(len(self.devices)):
        with ops.device(self.devices[idx]):
          self.size_ops.append(self._tables[idx].size(name=name))

    return (self.size_ops[index]
            if index is not None else math_ops.add_n(self.size_ops))

  def get_slot_variables(self, optimizer):
    """
    Get slot variables from optimizer. If Variable is trained by optimizer,
    then it returns the variables in slots of optimizer, else return an empty
    list.

    Args:
      optimizer: An optimizer under `tf.keras.optimizers` or `tf.compat.v1.train`.

    Returns:
      List of slot `Variable`s in optimizer.
    """
    if not isinstance(optimizer,
                      (Optimizer, OptimizerV2, tf.keras.optimizers.Optimizer)):
      raise TypeError('Expect an optimizer, but get {}'.format(type(optimizer)))
    slots = []
    snames = optimizer.get_slot_names()
    for tw in self._trainable_store.values():
      for name in snames:
        try:
          s = optimizer.get_slot(tw, name)
          slots.append(s.params)
        except:
          continue
    return slots

  def get_trainable_by_name(self, name):
    """
    Get trainable shadow variable when using eager execution.

    Example:
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

    Args:
      name: str. Name used to get the trainable shadow to the Variable.

    Returns:
      A ShadowVariable object refers to the specific name.

    Raises:
      RuntimeError: if not in eager mode.
    """
    if not isinstance(name, str):
      raise TypeError('name should be a string')
    return self._trainable_store.get(name, None)

  def _gather_saveables_for_checkpoint(self):
    g = ops.get_default_graph()
    if context.executing_eagerly() or g._functions:
      return {
          "py_state_de_var":
              functools.partial(base.PythonStringStateSaveable,
                                name=self.name,
                                state_callback=lambda: self.name,
                                restore_callback=lambda name: None)
      }
    else:
      saveables = dict()
      for table in self._tables:
        saveable_dict = table._gather_saveables_for_checkpoint()
        for (_, saveable) in saveable_dict.items():
          # merge all tables saveable to one dict with their own name.
          saveables[saveable.keywords["name"]] = saveable
      return saveables

  @property
  def trainable_store(self):
    return self._trainable_store


@tf_export("dynamic_embedding.get_variable")
def get_variable(
    name,  # unique
    key_dtype=dtypes.int64,
    value_dtype=dtypes.float32,
    dim=1,
    devices=None,
    partitioner=default_partition_fn,
    shared_name="get_variable",
    initializer=None,
    trainable=True,
    checkpoint=True,
    init_size=0,
    kv_creator=None,
    restrict_policy=None,
    bp_v2=False,
):
  """Gets an `Variable` object with this name if it exists,
         or create a new one.

    Args:
      name: A unique name for the `Variable`.
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      dim: the length of the value array for each key.
      devices: the list of devices holding the tables.
        One table will be created on each device.
      partitioner: partition function of keys,
        return the partition index for each key.

      Example partition func:
      ```python
      def default_partition_fn(keys, shard_num):
        return tf.cast(keys % shard_num, dtype=tf.int32)
      ```
      shared_name: No used.
      initializer: The value to use if a key is missing in the hash table.
        which can a python number, numpy array or `tf.initializer` instances.
        If initializer is `None` (the default), `0` will be used.
      trainable: Bool. If true, the variable will be treated as a trainable.
        Default is true.
      checkpoint: if True, the contents of the SparseVariable are
        saved to and restored from checkpoints.
        If `shared_name` is empty for a checkpointed table,
        it is shared using the table node name.
      init_size: initial size for the Variable and initial size of each hash
        tables will be int(init_size / N), N is the number of the devices.
      restrict_policy: a restrict policy to specify the rule to restrict the
        size of variable. If in training program, the variable is updated by
        optimizer, then the sparse slot variables in optimizer are also be
        restricted.
      bp_v2: By default with `bp_v2=False`, the optimizer will update
        dynamic embedding values by *setting* (key, value) after
        `optimizer.apply_gradient`. If one key is used by multiple workers
        at the same time, only one of them will be seen, while the others are
        overwritten. By setting `bp_v2=True`, the optimizer will update
        parameters by *adding delta* instead of *setting*, which solves the
        race condition problem among workers during backpropagation in
        large-scale distributed asynchronous training.

    Returns:
      A `Variable` object.
    """
  var_ = None
  scope = variable_scope.get_variable_scope()
  scope_store = variable_scope._get_default_variable_store()
  full_name = scope.name + "/" + name if scope.name else name
  if full_name in scope_store._vars:
    if scope.reuse is False:
      err_msg = ("Variable %s already exists, disallowed."
                 " Did you mean to set reuse=True or "
                 "reuse=tf.AUTO_REUSE in VarScope?" % full_name)

      raise ValueError(err_msg)
  else:
    var_ = Variable(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        dim=dim,
        devices=devices,
        partitioner=partitioner,
        shared_name=shared_name,
        name=full_name,
        initializer=initializer,
        trainable=trainable,
        checkpoint=checkpoint,
        init_size=init_size,
        kv_creator=kv_creator,
        restrict_policy=restrict_policy,
        bp_v2=bp_v2,
    )
    scope_store._vars[full_name] = var_
  return scope_store._vars[full_name]
