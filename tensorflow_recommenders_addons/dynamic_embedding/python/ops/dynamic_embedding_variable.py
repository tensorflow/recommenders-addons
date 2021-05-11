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

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.python import _pywrap_util_port
from tensorflow.python.client import device_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
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
from tensorflow.python.training.tracking import tracking as trackable
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
  gpu_mode = _pywrap_util_port.IsGoogleCudaEnabled()

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
  """Extended standard names related to `dynamic_embedding_ops.Variable` to use
  for graph collections.

  The following standard keys are defined:

  * `DYNAMIC_EMBEDDING_VARIABLES`: the default collection of
    all `dynamic_embedding_ops.Variable` objects.
  * `TRAINABLE_DYNAMIC_EMBEDDING_VARIABLES`: the subset of
    `dynamic_embedding_ops.Variable` that is trainable.
  """
  # Dynamic embedding variables.
  DYNAMIC_EMBEDDING_VARIABLES = "dynamic_embedding_variables"
  # Trainable dynamic embedding variables.
  TRAINABLE_DYNAMIC_EMBEDDING_VARIABLES = "trainable_dynamic_embedding_variables"


class Variable(trackable.TrackableResource):
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
      restrict_policy=None,
  ):
    """Creates an empty `Variable` object.

        Creates a group of tables placed on devices,
        the type of its keys and values are specified by key_dtype
        and value_dtype, respectively.
        The environment variables 'TF_HASHTABLE_INIT_SIZE' can be used to set the
        inital size of each tables, which can help reduce rehash times.
        The default initial table size : 1,048,576 for CPU, 16,777,216 for GPU.

        Args:
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
          name: A name for the operation (optional).
          initializer: The value to use if a key is missing in the hash table.
            which can be a python number, numpy array or `tf.initializer` instances.
            If initializer is `None` (the default), `0` will be taken.
          trainable: True, will be treated as a trainable Variable, and add to
            to the list of variables collected in the graph under the key
            `GraphKeys.TRAINABLE_VARIABLES`.
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

        Returns:
          A `Variable` object.
        """
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.dim = dim

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

    self._tables = []
    self.size_ops = []
    self.shard_num = len(self.devices)
    self.init_size = int(init_size)
    if restrict_policy is not None:
      if not issubclass(restrict_policy, de.RestrictPolicy):
        raise TypeError('restrict_policy must be subclass of RestrictPolicy.')
      self._restrict_policy = restrict_policy(self)
    else:
      self._restrict_policy = None

    key_dtype_list = [dtypes.int32, dtypes.int64, dtypes.string]
    value_dtype_list = [
        dtypes.int32, dtypes.int64, dtypes.bool, dtypes.float32, dtypes.float64,
        dtypes.half, dtypes.int8, dtypes.string
    ]
    if "GPU" in self.devices[0].upper():
      key_dtype_list = [dtypes.int64]
      value_dtype_list = [
          dtypes.int32, dtypes.float32, dtypes.half, dtypes.int8
      ]
    if key_dtype not in key_dtype_list:
      raise TypeError("key_dtype should be ", key_dtype_list)
    if value_dtype not in value_dtype_list:
      raise TypeError("value_dtype should be ", value_dtype_list)

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
            mht = de.CuckooHashTable(
                key_dtype=self.key_dtype,
                value_dtype=self.value_dtype,
                default_value=static_default_value,
                name=self._make_name(idx),
                checkpoint=self.checkpoint,
                init_size=int(self.init_size / self.shard_num),
            )

            self._tables.append(mht)
    super(Variable, self).__init__()

    ops.add_to_collection(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES, self)
    if trainable:
      ops.add_to_collections(de.GraphKeys.TRAINABLE_DYNAMIC_EMBEDDING_VARIABLES,
                             self)

  @property
  def tables(self):
    return self._tables

  @property
  def restrict_policy(self):
    return self._restrict_policy

  def _convert_anything_to_init(self, raw_init, dim):
    init = raw_init
    while callable(init):
      if isinstance(init, (init_ops.Initializer, init_ops_v2.Initializer)):
        self.initializer = init
        init = init(shape=[1])
      else:
        init = init()
    try:
      init = array_ops.reshape(init, [dim])
    except:
      init = array_ops.fill([dim], array_ops.reshape(init, [-1])[0])
    init = math_ops.cast(init, dtype=self.value_dtype)
    return init

  def _create_resource(self):
    raise NotImplementedError

  def _make_name(self, table_idx):
    return "{}_mht_{}of{}".format(self.name.replace("/", "_"), table_idx + 1,
                                  self.shard_num)

  def upsert(self, keys, values, name=None):
    """Insert or Update `keys` with `values`.

        If key exists already, value will be updated.

        Args:
          keys: Keys to insert. Can be a tensor of any shape. Must match the table's
            key type.
          values: Values to be associated with keys. Must be a tensor of the same
            shape as `keys` and match the table's value type.
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

  def lookup(self, keys, name=None):
    """Looks up `keys` in a Variable, outputs the corresponding values.

        The `default_value` is used for keys not present in the table.

        Args:
          keys: Keys to look up. Can be a tensor of any shape. Must match the
            table's key_dtype.
          name: A name for the operation (optional).

        Returns:
          A tensor containing the values in the same shape as `keys` using the
            table's value type.
        """
    partition_index = self.partition_fn(keys, self.shard_num)
    keys_partitions, keys_indices = make_partition(keys, partition_index,
                                                   self.shard_num)

    ops_ = []
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        dynamic_default_values = self._create_default_values_by_initializer(
            keys_partitions[idx])
        if dynamic_default_values is not None:
          dynamic_default_values = math_ops.cast(dynamic_default_values,
                                                 self.value_dtype)
        ops_.append(self._tables[idx].lookup(
            keys_partitions[idx],
            dynamic_default_values=dynamic_default_values,
            name=name,
        ))
    result = _stitch(ops_, keys_indices)

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

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    saveables = dict()
    for table in self._tables:
      # pylint: disable=protected-access
      saveable_dict = table._gather_saveables_for_checkpoint()
      for (_, saveable) in saveable_dict.items():
        # merge all tables saveable to one dict with their own name.
        saveables[saveable.keywords["name"]] = saveable
    return saveables


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
    restrict_policy=None,
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
      trainable: True, will be treated as a trainable Variable, and add to
        to the list of variables collected in the graph under the key
        `GraphKeys.TRAINABLE_VARIABLES`.
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
        restrict_policy=restrict_policy,
    )
    scope_store._vars[full_name] = var_
  return scope_store._vars[full_name]
