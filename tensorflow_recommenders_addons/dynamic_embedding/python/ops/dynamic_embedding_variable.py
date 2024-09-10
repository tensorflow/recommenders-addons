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

import functools
import re
import typing
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.embedding_weights import EmbeddingWeights
from tensorflow_recommenders_addons.utils.check_platform import is_macos, is_arm64

try:  # tf version >= 2.14.0
  from tensorflow.python.distribute import distribute_lib as distribute_ctx
  assert hasattr(distribute_ctx, 'has_strategy')
except:
  from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values as distribute_values_lib
from tensorflow.python.training.saving import saveable_object

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

try:  # tf version >= 2.16
  from tf_keras.initializers import Initializer
  from tf_keras.optimizers.legacy import Optimizer as keras_OptimizerV2_legacy
  from tf_keras.optimizers import Optimizer as keras_OptimizerV2
except:
  from tensorflow.keras.initializers import Initializer
  try:  # Keras version >= 2.12.0
    from tensorflow.keras.optimizers.legacy import Optimizer as keras_OptimizerV2_legacy
    from tensorflow.keras.optimizers import Optimizer as keras_OptimizerV2
  except:
    from tensorflow.keras.optimizers import Optimizer as keras_OptimizerV2_legacy
    keras_OptimizerV2 = keras_OptimizerV2_legacy

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.data.ops import readers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
try:  # tf version >= 2.14.0
  from tensorflow.python.ops.control_flow_assert import Assert
except:
  from tensorflow.python.ops.control_flow_ops import Assert
try:  # tf version >= 2.14.0
  from tensorflow.python.ops.cond import cond
except:
  from tensorflow.python.ops.control_flow_ops import cond
try:  # tf version >= 2.14.0
  from tensorflow.python.ops.while_loop import while_loop
except:
  from tensorflow.python.ops.control_flow_ops import while_loop
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.optimizer import Optimizer
try:  # tf version >= 2.10.0
  from tensorflow.python.trackable import base
except:
  from tensorflow.python.training.tracking import base
try:  # tf version >= 2.10.0
  from tensorflow.python.training.saving.saveable_object_util import _PythonStringStateSaveable as TF_PythonStringStateSaveable
except:
  from tensorflow.python.training.tracking.base import PythonStringStateSaveable as TF_PythonStringStateSaveable
try:  # The data_structures has been moved to the new package in tf 2.11
  from tensorflow.python.trackable import data_structures
except:
  from tensorflow.python.training.tracking import data_structures
try:  # tf version >= 2.14.0
  from tensorflow.python.trackable import python_state
except:
  from tensorflow.python.training.tracking import python_state
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.eager import tape as tape_record
if not hasattr(tape_record, 'record_operation'):
  # tf version >= 2.13.0
  from tensorflow.python.eager import record as tape_record


def make_partition(data, partition_index, shard_num, name=None):
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
    with ops.name_scope("data_partitions"):
      partitions = de.data_flow.dynamic_partition(data, partition_index,
                                                  shard_num, name)
    with ops.name_scope("make_indices"):
      indices = de.data_flow.dynamic_partition(
          math_ops.range(array_ops.shape(data)[0]), partition_index, shard_num,
          name)
  return partitions, indices


def _stitch(values, indices, use_fast=True, name=None):
  if len(values) == 1:
    return values[0]
  with ops.colocate_with(indices[0], ignore_existing=True):
    all_values = de.data_flow.dynamic_stitch(indices, values, use_fast, name)
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
  if shard_num <= 1:
    return array_ops.zeros(shape=array_ops.shape(keys), dtype=dtypes.int32)

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


def _list_de_variable_saved_files_from_file_system(de_variable_name,
                                                   de_variable_folder_path,
                                                   proc_size: int = None):
  """Lists the keys and values files saved by the corresponding DE in the file path.

    de_variable_name: Name of specific DynamicEmbedding Variable.
    de_variable_folder_path: Directory of DynamicEmbedding Variable saved files.
    proc_size: Specify that files of a certain mpi size are filtered.

  Returns:
    shard_keys_file_list: A list of shard table saved keys files.
    shard_values_file_list: A list of shard table saved values files.
  """
  if proc_size is None:
    keys_pattern = '_mht_*of*_rank*_size*-keys'
    values_pattern = '_mht_*of*_rank*_size*-values'
  else:
    keys_pattern = '_mht_*of*_rank*_size{}-keys'.format(proc_size)
    values_pattern = '_mht_*of*_rank*_size{}-values'.format(proc_size)
  de_variable_saveable_name = string_ops.regex_replace(de_variable_name, "/",
                                                       "_")
  _shard_name_base_dir = string_ops.string_join(
      [de_variable_folder_path, de_variable_saveable_name], separator='/')
  _shard_name_keys_pattern = string_ops.string_join(
      [_shard_name_base_dir, keys_pattern], separator='')
  shard_keys_file_list = gen_io_ops.matching_files(_shard_name_keys_pattern)
  _shard_name_values_pattern = string_ops.string_join(
      [_shard_name_base_dir, values_pattern], separator='')
  shard_values_file_list = gen_io_ops.matching_files(_shard_name_values_pattern)
  return shard_keys_file_list, shard_values_file_list


def _insert_de_shard_from_file_system(
    shard_list,
    value_dim,
    shard_keys_file_list,
    shard_values_file_list,
    partition_fn: typing.Callable[[typing.Any, typing.Any],
                                  typing.Any] = default_partition_fn,
    proc_size: int = None,
    proc_rank: int = None,
    buffer_size: int = 4194304):
  """Load DE keys/values files from file system as tensor partitions which fitting number of table shards in present DE variable. 
    Only load files and insert to DE table when shards number of files is not equal to the present DE variable table shards number.

    shard_list: A list of local DynamicEmbedding Variable shard table.
    value_dim: Embedding dim of current DE. 
    shard_keys_file_list: A list of shard table saved keys files.
    shard_values_file_list: A list of shard table saved values files.
    partition_fn: Calltable partition function for insert keys to DE shards table.
    proc_size: Total shards number of current DE in entire nodes system.
    proc_rank: Global shards index of current DE in total nodes.
    buffer_size: Read files buffer size for FixedLengthRecordDataset.
  Returns:
    traverse_files_result: A tensor from loop result, return False if success.
  """
  check_size_op = Assert(
      math_ops.equal(array_ops.size(shard_keys_file_list),
                     array_ops.size(shard_values_file_list)),
      [
          "The number of keys files and values files must be equal when loading DynamicEmbedding Variable shard table saved files."
      ])

  _local_shard_num = len(shard_list)
  _proc_size = 1
  if proc_size:
    _proc_size = proc_size
  _proc_rank = 0
  if proc_rank:
    _proc_rank = proc_rank

  _key_dtype = shard_list[0]._key_dtype
  _value_dtype = shard_list[0]._value_dtype

  insert_num_once = int(buffer_size / _key_dtype.size * 0.5)
  _keys_tensor_dataset = readers.FixedLengthRecordDataset(
      shard_keys_file_list,
      record_bytes=_key_dtype.size,
      buffer_size=buffer_size).padded_batch(insert_num_once,
                                            drop_remainder=False)

  _values_tensor_dataset = readers.FixedLengthRecordDataset(
      shard_values_file_list,
      record_bytes=_value_dtype.size * value_dim,
      buffer_size=buffer_size * value_dim).padded_batch(insert_num_once,
                                                        drop_remainder=False)

  iterator_init_list = tf_utils.ListWrapper([])
  iterator_init_list.as_list().append(check_size_op)
  if context.executing_eagerly():
    keys_tensor_iterator = iter(_keys_tensor_dataset)
    values_tensor_iterator = iter(_values_tensor_dataset)
  else:
    keys_tensor_iterator = dataset_ops.make_initializable_iterator(
        _keys_tensor_dataset)
    values_tensor_iterator = dataset_ops.make_initializable_iterator(
        _values_tensor_dataset)
    keys_iterator_init = keys_tensor_iterator.initializer
    values_iterator_init = values_tensor_iterator.initializer
    iterator_init_list.as_list().append(keys_iterator_init)
    iterator_init_list.as_list().append(values_iterator_init)

  def _traverse_files_cond(loop_continue):
    return loop_continue

  def _traverse_files_body(loop_continue):
    with ops.device("CPU"):
      keys_get_next = keys_tensor_iterator.get_next_as_optional()
      values_get_next = values_tensor_iterator.get_next_as_optional()

      def _insert_table():
        keys_tensor_byte = keys_get_next.get_value()
        keys_tensor = parsing_ops.decode_raw(keys_tensor_byte, _key_dtype)
        keys_tensor = array_ops.reshape(keys_tensor, (-1,))
        values_tensor_byte = values_get_next.get_value()
        values_tensor = parsing_ops.decode_raw(values_tensor_byte, _value_dtype)
        values_tensor = array_ops.reshape(values_tensor, (-1, value_dim))
        if _proc_size == 1:
          local_keys = keys_tensor
          local_values = values_tensor
        else:
          mpi_partition_index = partition_fn(keys_tensor, _proc_size)
          mpi_keys_partitions, _ = make_partition(keys_tensor,
                                                  mpi_partition_index,
                                                  _proc_size)
          mpi_values_partitions, _ = make_partition(values_tensor,
                                                    mpi_partition_index,
                                                    _proc_size)
          local_keys = mpi_keys_partitions[_proc_rank]
          local_values = mpi_values_partitions[_proc_rank]
        _insert_ops = tf_utils.ListWrapper([])
        local_partition_index = partition_fn(local_keys, _local_shard_num)
        local_keys_partitions, _ = make_partition(local_keys,
                                                  local_partition_index,
                                                  _local_shard_num)
        local_values_partitions, _ = make_partition(local_values,
                                                    local_partition_index,
                                                    _local_shard_num)
        for local_idx, shard in enumerate(shard_list):
          with ops.name_scope("%s_table_restore" % shard._name):
            _insert_ops.as_list().append(
                shard.insert(local_keys_partitions[local_idx],
                             local_values_partitions[local_idx]))
        with ops.control_dependencies(_insert_ops.as_list()):
          return constant_op.constant(True, dtype=dtypes.bool)

      def _end_insert_loop():
        return constant_op.constant(False, dtype=dtypes.bool)

      _files_has_value = math_ops.logical_and(keys_get_next.has_value(),
                                              values_get_next.has_value())
      return cond(_files_has_value, _insert_table, _end_insert_loop)

  with ops.control_dependencies(iterator_init_list.as_list()):
    traverse_files_result = while_loop(
        _traverse_files_cond, _traverse_files_body,
        [constant_op.constant(True, dtype=dtypes.bool)])
  return traverse_files_result


def load_de_variable_from_file_system(de_variable,
                                      de_variable_folder_dir,
                                      proc_size: int = None,
                                      proc_rank: int = None,
                                      buffer_size: int = 4194304):
  """Load DE keys/values files from file system or tensor array 
      which generated from load_de_variable_from_file_system function. 
    Load files directly when _de_now_global_shard_num == _de_prev_global_shard_num or _de_now_global_shard_num == 1.
    Otherwith insert tensor from files.

    de_variable: A present DynamicEmbedding Variable obeject.
    shard: A present DynamicEmbedding Variable table obeject.
    de_variable_folder_path: A directory path which saved DynamicEmbedding table shards before.
    proc_size: Total node size of current DE in entire nodes system.
    proc_rank: Global node index of current DE in entire nodes.
    buffer_size: Read files buffer size for FixedLengthRecordDataset or load_from_file_system op.
  Returns:
    load_op: A TF Operation contains a flow to load shard table.
  """
  _proc_size = 1
  device_size = len(de_variable.devices)
  if proc_size:
    _proc_size = proc_size
  _global_shard_num = _proc_size * device_size
  _proc_rank = 0
  if proc_rank:
    _proc_rank = proc_rank
  _shard_list = tf_utils.ListWrapper([])
  for table in de_variable._tables:
    _shard_tmp = table._new_obj_trackable
    if _shard_tmp is None:
      _shard_tmp = table
    _shard_list.as_list().append(_shard_tmp)

  _de_now_global_shard_num = constant_op.constant(_global_shard_num,
                                                  dtypes.int32,
                                                  name=de_variable.name +
                                                  "-now_global_shard_num")
  _test_shard_keys_file_list, _ = _list_de_variable_saved_files_from_file_system(
      de_variable.name, de_variable_folder_dir, _proc_size)
  _de_prev_global_shard_num = array_ops.size(_test_shard_keys_file_list)

  def load_de_table_directly():
    _insert_ops = tf_utils.ListWrapper([])
    for idx, table in enumerate(de_variable._tables):
      shard = table._new_obj_trackable
      if shard is None:
        shard = table
      file_name = re.sub(
          r'_mht_([^/]*)of([^/]*)',
          '_mht_' + str(idx + 1) + 'of' + str(device_size) + '_rank' +
          str(_proc_rank) + '_size' + str(_proc_size), shard._name)
      with ops.name_scope(de_variable.name, "%s_table_restore" % shard._name):
        _insert_ops.as_list().append(
            shard.load_from_file_system(de_variable_folder_dir,
                                        file_name=file_name,
                                        buffer_size=buffer_size))
    return control_flow_ops.group(_insert_ops.as_list())

  def upsert_de_table():

    def load_all_de_table():
      shard = de_variable._tables[0]._new_obj_trackable
      if shard is None:
        shard = table
      with ops.name_scope(de_variable.name, "%s_table_restore" % shard._name):
        return shard.load_from_file_system(de_variable_folder_dir,
                                           file_name=shard._name,
                                           load_entire_dir=True,
                                           buffer_size=buffer_size)

    def insert_de_table():
      _shard_keys_file_list, _shard_values_file_list = _list_de_variable_saved_files_from_file_system(
          de_variable.name, de_variable_folder_dir)
      _partition_fn = de_variable.partition_fn
      with ops.name_scope(de_variable.name):
        return _insert_de_shard_from_file_system(_shard_list.as_list(),
                                                 de_variable.dim,
                                                 _shard_keys_file_list,
                                                 _shard_values_file_list,
                                                 _partition_fn, _proc_size,
                                                 _proc_rank, buffer_size)

    upsert_ops = cond(math_ops.equal(_de_now_global_shard_num, 1),
                      load_all_de_table, insert_de_table)
    return upsert_ops

  load_op = cond(
      math_ops.equal(_de_now_global_shard_num, _de_prev_global_shard_num),
      load_de_table_directly, upsert_de_table)
  return load_op


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


class Variable(EmbeddingWeights, base.Trackable):
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
      short_file_name=False,
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
          short_file_name: If True, the file name will not use scope name as prefix and create_slots will not
            use op_name to avoid file name over 255. the default is False to keep the same behavior as before.
        Returns:
          A `Variable` object.
    """
    self.key_dtype = key_dtype
    self.value_dtype = value_dtype
    self.dim = dim
    self.bp_v2 = bp_v2
    self.short_file_name = short_file_name

    def _get_default_devices():
      try:
        gpu_list = [
            x.name
            for x in device_lib.list_local_devices()
            if x.device_type == "GPU"
        ]
      except:
        gpu_list = []
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
    self._distribute_trainable_store = {}
    self.kv_creator = kv_creator if kv_creator else de.CuckooHashTableCreator()
    self._saveable_object_creator = self.kv_creator.saver

    self.shard_num = len(self.devices)

    self.init_size = int(init_size)

    if restrict_policy is not None:
      if not issubclass(restrict_policy, de.RestrictPolicy):
        raise TypeError('restrict_policy must be subclass of RestrictPolicy.')
      self._restrict_policy = restrict_policy(self)
      _restrict_var = self._restrict_policy._restrict_var
      self._track_trackable(_restrict_var, _restrict_var.name, overwrite=False)
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
                        [dtypes.int32, dtypes.bfloat16],
                        [dtypes.string, dtypes.float32],
                        [dtypes.string, dtypes.half],
                        [dtypes.string, dtypes.bfloat16],
                        [dtypes.string, dtypes.int32],
                        [dtypes.string, dtypes.int8],
                        [dtypes.string, dtypes.int64],
                        [dtypes.string, dtypes.float64],
                        [dtypes.string, dtypes.bool]]
    if "GPU" in self.devices[0].upper() or isinstance(self.kv_creator,
                                                      de.HkvHashTableCreator):
      valid_dtype_list = [
          [dtypes.int64, dtypes.float32],
          [dtypes.int64, dtypes.int8],
          [dtypes.int64, dtypes.int32],
          [dtypes.int64, dtypes.int64],
          [dtypes.int64, dtypes.half],
          [dtypes.int64, dtypes.bfloat16],
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
            if not issubclass(self.kv_creator.__class__, de.KVCreator):
              raise TypeError("config should be instance of 'config', but got ",
                              str(type(self.kv_creator)))
            if self._saveable_object_creator:
              shard_saveable_object_fn_i = functools.partial(
                  self._saveable_object_creator.create_shard_saveable_object,
                  variable=self,
                  shard_idx=idx)
            else:
              shard_saveable_object_fn_i = None
            mht = self.kv_creator.create(
                key_dtype=self.key_dtype,
                value_dtype=self.value_dtype,
                default_value=static_default_value,
                name=self._make_name(idx),
                checkpoint=self.checkpoint,
                init_size=int(self.init_size / self.shard_num),
                device=self.devices[idx],
                shard_saveable_object_fn=shard_saveable_object_fn_i)
            self._tables.append(mht)

    if self._saveable_object_creator:
      if self.checkpoint:
        if not context.executing_eagerly():
          self.op = control_flow_ops.no_op(name=self.name)
          self.saveable = self._saveable_object_creator.create_variable_saveable_object(
              self, self.op.name)
          ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self.saveable)
        else:
          self.saveable = self._saveable_object_creator.create_variable_saveable_object(
              self, self.name)

  def verify_embedding_weights(self, sparse_ids, sparse_weights=None):
    EmbeddingWeights.verify_embedding_param_weights(self, sparse_ids,
                                                    sparse_weights)

  def embedding_lookup(self,
                       ids,
                       name=None,
                       max_norm=None) -> (tf.Tensor, EmbeddingWeights):
    return embedding_lookup(
        self,
        ids,
        name=name + '/embedding_lookup',
        max_norm=max_norm,
        return_trainable=True,
    )

  @property
  def tables(self):
    return self._tables

  @property
  def restrict_policy(self):
    return self._restrict_policy

  def _convert_anything_to_init(self, raw_init, dim):
    init = raw_init
    valid_list = [init_ops.Initializer, init_ops_v2.Initializer, Initializer]
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

      def is_indexable_and_nonempty(obj):
        has_getitem = hasattr(obj, '__getitem__')
        is_nonempty = hasattr(obj, '__len__') and len(obj) > 0
        return has_getitem and is_nonempty

      if isinstance(init, int) or isinstance(init, float):
        first_element = init
      elif not isinstance(init, tf.Tensor) and is_indexable_and_nonempty(init):
        first_element = init[0]
      else:
        reshaped_init = array_ops.reshape(init, [-1])
        size_of_reshaped_init = tf.size(reshaped_init)

        def get_default_value():
          default_value = 0.0 if self.value_dtype.is_floating else 0
          return tf.constant(default_value, dtype=self.value_dtype)

        first_element = tf.cond(tf.greater(size_of_reshaped_init, 0),
                                lambda: reshaped_init[0], get_default_value)
      init = array_ops.fill([dim], first_element)
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

    ops_ = tf_utils.ListWrapper([])
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.as_list().append(self._tables[idx].insert(keys_partitions[idx],
                                                       values_partitions[idx],
                                                       name=name))

    return control_flow_ops.group(ops_.as_list())

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

    ops_ = tf_utils.ListWrapper([])
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.as_list().append(self._tables[idx].accum(
            keys_partitions[idx],
            values_or_deltas_partitions[idx],
            exists_partitions[idx],
            name=name))

    return control_flow_ops.group(ops_.as_list())

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

    ops_ = tf_utils.ListWrapper([])
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.as_list().append(self._tables[idx].remove(keys_partitions[idx],
                                                       name=name))

    return control_flow_ops.group(ops_.as_list())

  def clear(self, name=None):
    """clear all keys and values in the table.

        Args:
          name: A name for the operation (optional).

        Returns:
          The created Operation.
        """
    ops_ = tf_utils.ListWrapper([])
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        ops_.as_list().append(self._tables[idx].clear(name=name))
    return control_flow_ops.group(ops_.as_list())

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
      result = (_stitch(_values, keys_indices, use_fast=True),
                _stitch(_exists, keys_indices, use_fast=True))
    else:
      result = _stitch(_values, keys_indices, use_fast=True)
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

  def save_to_file_system(self,
                          dirpath,
                          proc_size=1,
                          proc_rank=0,
                          file_name_list=None,
                          dirpath_env='TFRA_SAVED_KV',
                          append_to_file=False,
                          buffer_size=4194304,
                          name=None):
    """
    Returns an operations list to save the keys and values in tables to dirpath. 
    The keys and values will be stored in FileSystem, rewrited or appended to the filepath.
    Args:
      dirpath: A directory path to save the table.
      proc_size: Number of nodes when using distributed trainning library (e.g. Horovod). 
      proc_rank: Rank of this node when using distributed trainning library (e.g. Horovod). 
      file_name_list: User custom file names list for key/value prefix file name, default is table._name.
      dirpath_env: A environment variable stored a path to save the table, which priority higher than dirpath.
      buffer_size: Number of keys in write buffer to file.
      append_to_file: If true, operation will append data to the file but not write a new one.
      name: Name for the operation.
    Returns:
      An operation to save the tables.
    """
    device_size = len(self.devices)
    ops_ = tf_utils.ListWrapper([])
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        if file_name_list is not None:
          save_file_name = file_name_list[idx]
        else:
          file_name = self._tables[idx]._name
          save_file_name = re.sub(
              r'_mht_([^/]*)of([^/]*)',
              '_mht_' + str(idx + 1) + 'of' + str(device_size) + '_rank' +
              str(proc_rank) + '_size' + str(proc_size), file_name)
        ops_.as_list().append(self._tables[idx].save_to_file_system(
            dirpath=dirpath,
            file_name=save_file_name,
            dirpath_env=dirpath_env,
            append_to_file=append_to_file,
            buffer_size=buffer_size,
            name=name))
    return control_flow_ops.group(ops_.as_list())

  def load_from_file_system(self,
                            dirpath,
                            proc_size=1,
                            proc_rank=0,
                            file_name_list=None,
                            dirpath_env='TFRA_SAVED_KV',
                            load_entire_dir=False,
                            buffer_size=4194304,
                            name=None):
    """
    Returns an operations list to load keys and values to table from
    FileSystem. The keys and values files are generated from `save_to_file_system`.
    Args:
      dirpath: A directory path stored the table keys and values files.
      proc_size: Number of nodes when using distributed trainning library (e.g. Horovod). 
      proc_rank: Rank of this node when using distributed trainning library (e.g. Horovod). 
      file_name_list: User custom file names list for key/value prefix file name, default is table._name.
      dirpath_env: A environment variable stored a path to load the table, which priority higher than dirpath.
      load_entire_dir: If true, operation will load all key value files in the dirpath regardless partition.
      buffer_size: Number of keys in read buffer from file.
      name: Name for the operation.
    Returns:
      An operation to load keys and values to table from FileSystem.
    """
    device_size = len(self.devices)
    ops_ = tf_utils.ListWrapper([])
    for idx in range(len(self.devices)):
      with ops.device(self.devices[idx]):
        if file_name_list is not None:
          load_file_name = file_name_list[idx]
        else:
          file_name = self._tables[idx]._name
          load_file_name = re.sub(
              r'_mht_([^/]*)of([^/]*)',
              '_mht_' + str(idx + 1) + 'of' + str(device_size) + '_rank' +
              str(proc_rank) + '_size' + str(proc_size), file_name)
        ops_.as_list().append(self._tables[idx].load_from_file_system(
            dirpath=dirpath,
            file_name=load_file_name,
            dirpath_env=dirpath_env,
            load_entire_dir=load_entire_dir,
            buffer_size=buffer_size,
            name=name))
    return control_flow_ops.group(ops_.as_list())

  def export_keys_and_scores(self, split_size, name=None):
    if not isinstance(self.kv_creator, de.HkvHashTableCreator):
      raise TypeError("Only hkv HashTable support export_keys_and_scores")
    full_keys = []
    full_scores = []
    for idx in range(len(self.devices)):
      keys_ = None
      scores_ = None
      with ops.device(self.devices[idx]):
        keys_, scores_ = self._tables[idx].export_keys_and_scores(
            split_size=split_size, name=name)
        full_keys.append(keys_)
        full_scores.append(scores_)
    return array_ops.concat(full_keys, 0), array_ops.concat(full_scores, 0)

  def load_from_file_system_with_restore_function(self,
                                                  dirpath,
                                                  proc_size=1,
                                                  proc_rank=0,
                                                  buffer_size=4194304):
    """
    Returns an operation to load the keys and values in tables from dirpath. 
    The keys and values will be loaded in FileSystem. Compatible with table shards scaling.
    Args:
      dirpath: A directory path stored the table keys and values files.
      proc_size: Number of nodes when using distributed trainning library (e.g. Horovod). 
      proc_rank: Rank of this node when using distributed trainning library (e.g. Horovod). 
      name: Name for the operation.
    Returns:
      An operation to save the tables.
    """
    return load_de_variable_from_file_system(self, dirpath, proc_size,
                                             proc_rank, buffer_size)

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
    if not isinstance(optimizer, (Optimizer, OptimizerV2, keras_OptimizerV2)):
      raise TypeError('Expect an optimizer, but get {}'.format(type(optimizer)))
    slots = []
    if hasattr(optimizer, 'get_slot_names'):
      snames = optimizer.get_slot_names()
      for tw in self._trainable_store.values():
        for name in snames:
          try:
            s = optimizer.get_slot(tw, name)
            slots.append(s.params)
          except:
            continue
    else:
      for tw in self._trainable_store.values():
        for s in optimizer._variables:
          try:
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

    def _get_saveable_object_creator(self, saveables):
      if self._saveable_object_creator:
        if context.executing_eagerly():
          op_name = self.name
        else:
          self.op = control_flow_ops.no_op(name=self.name)
          op_name = self.op.name
        saveables[self.name] = functools.partial(
            self._saveable_object_creator.create_variable_saveable_object,
            variable=self,
            name=op_name)

    g = ops.get_default_graph()
    if context.executing_eagerly() or g._functions:
      saveables = dict()
      saveables["py_state_de_var"] = functools.partial(
          TF_PythonStringStateSaveable,
          name=self.name,
          state_callback=lambda: self.name,
          restore_callback=lambda name: None)
      _get_saveable_object_creator(self, saveables)
      return saveables
    else:
      saveables = dict()
      for table in self._tables:
        saveable_dict = table._gather_saveables_for_checkpoint()
        for (_, saveable) in saveable_dict.items():
          # merge all tables saveable to one dict with their own name.
          saveables[saveable.keywords["name"]] = saveable
      _get_saveable_object_creator(self, saveables)
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
    short_file_name=False,
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
      short_file_name: If True, the file name will not use scope name as prefix and create_slots will not
          use op_name to avoid file name over 255. the default is False to keep the same behavior as before.
    Returns:
      A `Variable` object.
    """
  scope = variable_scope.get_variable_scope()
  scope_store = variable_scope._get_default_variable_store()

  full_name = scope.name + "/" + name if scope.name and not short_file_name else name
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
        short_file_name=short_file_name,
    )
    scope_store._vars[full_name] = var_
  return scope_store._vars[full_name]


def embedding_lookup(
    params,
    ids,
    partition_strategy=None,  # pylint: disable=unused-argument
    name=None,
    validate_indices=None,  # pylint: disable=unused-argument
    max_norm=None,
    return_trainable=False,
):
  """Provides a dynamic version of embedding_lookup
      similar with tf.nn.embedding_lookup.

    Ids are flattened to a 1d tensor before being passed to embedding_lookup
    then, they are unflattend to match the original ids shape plus an extra
    leading dimension of the size of the embeddings.
    ids must be unique or call safe_embedding_lookup_sparse in the GPU case
        if you use HKV hashtable since HKV requires unique key
    Args:
      params: A dynamic_embedding.Variable instance.
      ids: A tensor with any shape as same dtype of params.key_dtype.
      partition_strategy: No used, for API compatiblity with `nn.emedding_lookup`.
      name: A name for the operation. Name is optional in graph mode and required
        in eager mode.
      validate_indices: No used, just for compatible with nn.embedding_lookup .
      max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
        than this value.
      return_trainable: optional, If True, also return TrainableWrapper. If in
        eager mode, it will return a `ShadowVariable`, which is eager derivative of
        TrainableWrapper. If inside tf.function scope, then set return_trainable
        is disabled. Please use `dynamic_embedding.Variable.get_trainable_by_name` or
        `dynamic_embedding.Variable.trainable_store` to get the created trainable
        shadow inside tf.function scope.
    Returns:
      A tensor with shape [shape of ids] + [dim],
        dim is equal to the value dim of params.
        containing the values from the params tensor(s) for keys in ids.
      trainable_wrap:
        A TrainableWrapper object used to fill the Optimizers `var_list`
          Only provided if `return_trainable` is True. If in eager mode,
          it will be a `ShadowVariable`, which is eager derivative of TrainableWrapper.
    """
  if isinstance(params, (list, tuple)) and len(params) > 1:
    raise ValueError("Only one params is allowed.")
  if isinstance(params, (list, tuple)):
    params = params[0]
  if not isinstance(params, de.Variable):
    raise TypeError("params should be a Variable instance.")
  if params.key_dtype != ids.dtype:
    raise TypeError(
        "params.key_dtype should be same with ids.dtype: {} vs. {}".format(
            params.key_dtype, ids.dtype))
  if context.executing_eagerly() and (name is None):
    raise ValueError(
        'Must specify a name for dynamic_embedding.embedding_lookup when running '
        'eagerly. The `de.shadow_ops.embedding_lookup` is recommended in eager case.'
    )

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" if scope.name else ""
  full_name += (name + "/") if name else "embedding_lookup/"
  with ops.name_scope(full_name):
    ids = ops.convert_to_tensor(ids, name="ids")
    if ids.get_shape().is_fully_defined():
      # use static shape
      initial_shape = [ids.get_shape().num_elements(), params.dim]
      embeddings_shape = ids.get_shape().concatenate([params.dim])
    else:
      # use dynamic shape
      initial_shape = (1, params.dim)
      embeddings_shape = array_ops.concat([array_ops.shape(ids), [params.dim]],
                                          axis=0)
    initial_value = array_ops.zeros(shape=initial_shape,
                                    dtype=params.value_dtype)
    if (isinstance(initial_value, tf.Tensor)
        and hasattr(initial_value, "graph")
        and initial_value.graph.building_function):

      def initial_value():
        return array_ops.zeros(initial_shape, dtype=params.value_dtype)

    with ops.colocate_with(None, ignore_existing=True):
      collections = [ops.GraphKeys.LOCAL_VARIABLES]
      if params.trainable:
        collections += [ops.GraphKeys.TRAINABLE_VARIABLES]

      def _create_or_get_trainable(trainable_name):
        if trainable_name is None:
          if context.executing_eagerly():
            raise ValueError(
                'Must provide a name for embedding_lookup when using eager execution.'
            )
          _ANONYMOUS_TRAINABLE_STORE_KEY = '_anonymous_trainable_store_key'
          trainable_name = ops.get_default_graph().unique_name(
              _ANONYMOUS_TRAINABLE_STORE_KEY)
        if not context.executing_eagerly() and not ops.inside_function():
          wrapper = de.TrainableWrapper(params=params,
                                        ids=ids,
                                        max_norm=max_norm,
                                        initial_value=initial_value,
                                        dtype=params.value_dtype,
                                        trainable=params.trainable,
                                        collections=collections,
                                        model_mode=de.ModelMode.CURRENT_SETTING,
                                        name=trainable_name)
          params._trainable_store[trainable_name] = wrapper
          return wrapper
        else:
          with ops.init_scope():
            shadow = params._trainable_store.get(trainable_name, None)
            if shadow is None:
              shadow = de.shadow_ops.ShadowVariable(
                  params,
                  name=trainable_name,
                  max_norm=max_norm,
                  trainable=params.trainable,
                  model_mode=de.ModelMode.CURRENT_SETTING)
              params._trainable_store[trainable_name] = shadow
          return shadow

      with ops.colocate_with(ids, ignore_existing=True):
        if distribute_ctx.has_strategy():
          trainable_ = params._distribute_trainable_store.get(name, None)
          if trainable_ is None:
            strategy_devices = distribute_ctx.get_strategy(
            ).extended.worker_devices
            trainable_impl = tf_utils.ListWrapper([])
            for i, strategy_device in enumerate(strategy_devices):
              with ops.device(strategy_device):
                name_replica = name
                if i > 0:
                  name_replica = "%s/replica_%d" % (name, i)
                with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                  with tape_record.stop_recording():
                    trainable_impl.as_list().append(
                        _create_or_get_trainable(name_replica))

            trainable_ = de.DistributedVariableWrapper(
                distribute_ctx.get_strategy(), trainable_impl.as_list(),
                tf.VariableAggregation.NONE,
                TrainableWrapperDistributedPolicy(tf.VariableAggregation.NONE))
            params._distribute_trainable_store[name] = trainable_
        else:
          trainable_ = _create_or_get_trainable(name)

      if distribute_utils.is_distributed_variable(trainable_):
        trainable_device = trainable_._get_on_device_or_primary()
      else:
        trainable_device = trainable_
      if isinstance(trainable_device, de.shadow_ops.ShadowVariable):
        embeddings = de.shadow_ops.embedding_lookup(
            trainable_device,
            ids,
            partition_strategy=partition_strategy,
            name=name,
            validate_indices=validate_indices)
        if return_trainable:
          if not context.executing_eagerly():
            raise NotImplementedError(
                'return_trainable currently is not implemented when using tf.function.'
                ' Please use `Variable.trainable_store` or `Variable.get_trainable_by_name`'
                ' to access the shadow trainable variable if call `embedding_lookup` series'
                ' APIs inside tf.function scope.')
          return embeddings, trainable_
        return embeddings

    embeddings = trainable_
    embeddings = array_ops.reshape(embeddings, shape=embeddings_shape)

  return (embeddings, trainable_) if return_trainable else embeddings


class TrainableWrapperDistributedPolicy(distribute_values_lib.OnWritePolicy):

  def get_saveable(self, var, primary_var, name):
    for v in var.values:
      v._gather_saveables_for_checkpoint()

    def tensor():
      return array_ops.zeros((
          0,
          primary_var.params.dim,
      ),
                             dtype=primary_var.params.value_dtype)  # pylint: disable=protected-access

    spec = saveable_object.SaveSpec(tensor=tensor,
                                    slice_spec="",
                                    name=name,
                                    dtype=primary_var.params.value_dtype,
                                    device=primary_var.device)
    return tensor, [spec]
