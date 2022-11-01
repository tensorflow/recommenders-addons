# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""CuckooHash Lookup operations."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.lookup_ops import LookupInterface
from tensorflow.python.training.saver import BaseSaverBuilder

from tensorflow_recommenders_addons.utils.resource_loader import LazySO
from tensorflow_recommenders_addons.utils.resource_loader import prefix_op_name

cuckoo_ops = LazySO("dynamic_embedding/core/_cuckoo_hashtable_ops.so").ops


class CuckooHashTable(LookupInterface):
  """A generic mutable hash table implementation.

    Data can be inserted by calling the insert method and removed by calling the
    remove method. It does not support initialization via the init method.

    Example usage:

    ```python
    table = tfra.dynamic_embedding.CuckooHashTable(key_dtype=tf.string,
                                                   value_dtype=tf.int64,
                                                   default_value=-1)
    sess.run(table.insert(keys, values))
    out = table.lookup(query_keys)
    print(out.eval())
    ```
    """

  def __init__(
      self,
      key_dtype,
      value_dtype,
      default_value,
      name="CuckooHashTable",
      checkpoint=True,
      init_size=0,
      config=None,
      device='',
  ):
    """Creates an empty `CuckooHashTable` object.

        Creates a table, the type of its keys and values are specified by key_dtype
        and value_dtype, respectively.

        Args:
          key_dtype: the type of the key tensors.
          value_dtype: the type of the value tensors.
          default_value: The value to use if a key is missing in the table.
          name: A name for the operation (optional).
          checkpoint: if True, the contents of the table are saved to and restored
            from checkpoints. If `shared_name` is empty for a checkpointed table, it
            is shared using the table node name.
          init_size: initial size for the Variable and initial size of each hash
            tables will be int(init_size / N), N is the number of the devices.

        Returns:
          A `CuckooHashTable` object.

        Raises:
          ValueError: If checkpoint is True and no name was specified.
        """
    self._default_value = ops.convert_to_tensor(default_value,
                                                dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._device = device
    self._init_size = init_size
    self._name = name

    self._shared_name = None
    if context.executing_eagerly():
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      # TODO(rohanj): Use context.shared_name() instead.
      self._shared_name = "table_%d" % (ops.uid(),)
    super(CuckooHashTable, self).__init__(key_dtype, value_dtype)

    self._resource_handle = self._create_resource()
    if checkpoint:
      _ = CuckooHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        self.saveable = CuckooHashTable._Saveable(
            self,
            name=self._resource_handle.op.name,
            full_name=self._resource_handle.op.name,
        )
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self.saveable)
      else:
        self.saveable = CuckooHashTable._Saveable(self,
                                                  name=name,
                                                  full_name=name)

  def _create_resource(self):
    # The table must be shared if checkpointing is requested for multi-worker
    # training to work correctly. Use the node name if no shared_name has been
    # explicitly specified.
    use_node_name_sharing = self._checkpoint and self._shared_name is None

    with ops.device(self._device):
      table_ref = cuckoo_ops.tfra_cuckoo_hash_table_of_tensors(
          shared_name=self._shared_name,
          use_node_name_sharing=use_node_name_sharing,
          key_dtype=self._key_dtype,
          value_dtype=self._value_dtype,
          value_shape=self._default_value.get_shape(),
          init_size=self._init_size,
          name=self._name,
      )

    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref

  @property
  def name(self):
    return self._table_name

  def size(self, name=None):
    """Compute the number of elements in this table.

        Args:
          name: A name for the operation (optional).

        Returns:
          A scalar tensor containing the number of elements in this table.
        """
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        return cuckoo_ops.tfra_cuckoo_hash_table_size(self.resource_handle)

  def remove(self, keys, name=None):
    """Removes `keys` and its associated values from the table.

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
    if keys.dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    with ops.name_scope(
        name,
        "%s_lookup_table_remove" % self.name,
        (self.resource_handle, keys, self._default_value),
    ):
      op = cuckoo_ops.tfra_cuckoo_hash_table_remove(self.resource_handle, keys)

    return op

  def clear(self, name=None):
    """clear all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    """
    with ops.name_scope(name, "%s_lookup_table_clear" % self.name,
                        (self.resource_handle, self._default_value)):
      op = cuckoo_ops.tfra_cuckoo_hash_table_clear(
          self.resource_handle,
          key_dtype=self._key_dtype,
          value_dtype=self._value_dtype)

    return op

  def lookup(self,
             keys,
             dynamic_default_values=None,
             return_exists=False,
             name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

      The `default_value` is used for keys not present in the table.

      Args:
        keys: Keys to look up. Can be a tensor of any shape. Must match the
          table's key_dtype.
        dynamic_default_values: The values to use if a key is missing in the
          table. If None (by default), the static default_value
          `self._default_value` will be used.
        return_exists: if True, will return a additional Tensor which indicates
          if or not keys are existing in the table.
        name: A name for the operation (optional).

      Returns:
        A tensor containing the values in the same shape as `keys` using the
          table's value type.
        exists:
          A bool type Tensor of the same shape as `keys` which indicates
            if keys are existing in the table.
            Only provided if `return_exists` is True.

      Raises:
        TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(
        name,
        "%s_lookup_table_find" % self.name,
        (self.resource_handle, keys, self._default_value),
    ):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle, ignore_existing=True):
        if return_exists:
          values, exists = cuckoo_ops.tfra_cuckoo_hash_table_find_with_exists(
              self.resource_handle,
              keys,
              dynamic_default_values
              if dynamic_default_values is not None else self._default_value,
          )
        else:
          values = cuckoo_ops.tfra_cuckoo_hash_table_find(
              self.resource_handle,
              keys,
              dynamic_default_values
              if dynamic_default_values is not None else self._default_value,
          )
    return (values, exists) if return_exists else values

  def insert(self, keys, values, name=None):
    """Associates `keys` with `values`.

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
    with ops.name_scope(
        name,
        "%s_lookup_table_insert" % self.name,
        [self.resource_handle, keys, values],
    ):
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle, ignore_existing=True):
        # pylint: disable=protected-access
        op = cuckoo_ops.tfra_cuckoo_hash_table_insert(self.resource_handle,
                                                      keys, values)
    return op

  def accum(self, keys, values_or_deltas, exists, name=None):
    """Associates `keys` with `values`.

      Args:
        keys: Keys to accmulate. Can be a tensor of any shape.
          Must match the table's key type.
        values_or_deltas: values to be associated with keys. Must be a tensor of
          the same shape as `keys` and match the table's value type.
        exists: A bool type tensor indicates if keys already exist or not.
          Must be a tensor of the same shape as `keys`.
        name: A name for the operation (optional).

      Returns:
        The created Operation.

      Raises:
        TypeError: when `keys` or `values` doesn't match the table data
          types.
    """
    with ops.name_scope(
        name,
        "%s_lookup_table_accum" % self.name,
        [self.resource_handle, keys, values_or_deltas],
    ):
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values_or_deltas = ops.convert_to_tensor(values_or_deltas,
                                               self._value_dtype,
                                               name="values_or_deltas")
      exists = ops.convert_to_tensor(exists, dtypes.bool, name="exists")
      with ops.colocate_with(self.resource_handle, ignore_existing=True):
        # pylint: disable=protected-access
        op = cuckoo_ops.tfra_cuckoo_hash_table_accum(self.resource_handle, keys,
                                                     values_or_deltas, exists)
    return op

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

        Args:
          name: A name for the operation (optional).

        Returns:
          A pair of tensors with the first tensor containing all keys and the
            second tensors containing all values in the table.
        """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self.name,
                        [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        keys, values = cuckoo_ops.tfra_cuckoo_hash_table_export(
            self.resource_handle, self._key_dtype, self._value_dtype)
    return keys, values

  def save_to_file_system(self,
                          dirpath,
                          file_name=None,
                          dirpath_env='TFRA_SAVED_KV',
                          append_to_file=False,
                          buffer_size=4194304,
                          name=None):
    """
    Returns an operation to save the keys and values in table to dirpath. 
    The keys and values will be stored in FileSystem, rewrited or appended to the filepath.
    Args:
      dirpath: A directory path to save the table.
      dirpath_env: A environment variable stored a path to save the table, which priority higher than dirpath.
      file_name: User custom file name for key/value prefix file name, default is self._name.
      buffer_size: Number of keys in write buffer to file.
      append_to_file: If true, operation will append data to the file but not write a new one.
      name: Name for the operation.
    Returns:
      An operation to save the table.
    """
    with ops.name_scope(name, "%s_save_table" % self.name,
                        [self.resource_handle]):
      with ops.colocate_with(None, ignore_existing=True):
        return cuckoo_ops.tfra_cuckoo_hash_table_save_to_file_system(
            self.resource_handle,
            dirpath=dirpath,
            file_name=file_name if file_name else self._name,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            dirpath_env=dirpath_env,
            append_to_file=append_to_file,
            buffer_size=buffer_size)

  def load_from_file_system(self,
                            dirpath,
                            file_name=None,
                            dirpath_env='TFRA_SAVED_KV',
                            load_entire_dir=False,
                            buffer_size=4194304,
                            name=None):
    """
    Returns an operation to load keys and values to table from
    FileSystem. The keys and values files are generated from `save_to_file_system`.
    Args:
      dirpath: A directory path stored the table keys and values.
      dirpath_env: A environment variable stored a path to load the table, which priority higher than dirpath.
      file_name: User custom file name for key/value prefix file name, default is self._name.
      buffer_size: Number of keys in read buffer from file.
      load_entire_dir: If true, operation will load all key value files in the dirpath regardless partition.
      name: Name for the operation.
    Returns:
      An operation to load keys and values to table from FileSystem.
    """
    with ops.name_scope(name, "%s_load_table" % self.name,
                        [self.resource_handle]):
      with ops.colocate_with(None, ignore_existing=True):
        return cuckoo_ops.tfra_cuckoo_hash_table_load_from_file_system(
            self.resource_handle,
            dirpath=dirpath,
            file_name=file_name if file_name else self._name,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            dirpath_env=dirpath_env,
            load_entire_dir=load_entire_dir,
            buffer_size=buffer_size)

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    # full_name helps to figure out the name-based Saver's name for this saveable.
    full_name = self._table_name
    return {
        "table":
            functools.partial(
                CuckooHashTable._Saveable,
                table=self,
                name=self._name,
                full_name=full_name,
            )
    }

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for CuckooHashTable."""

    def __init__(self, table, name, full_name=""):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values"),
      ]
      # pylint: disable=protected-access
      super(CuckooHashTable._Saveable, self).__init__(table, specs, name)
      self._restore_name = table._name

    def restore(self, restored_tensors, restored_shapes, name=None):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope(name, "%s_table_restore" % self._restore_name):
        with ops.colocate_with(self.op.resource_handle):
          return cuckoo_ops.tfra_cuckoo_hash_table_import(
              self.op.resource_handle,
              restored_tensors[0],
              restored_tensors[1],
          )


ops.NotDifferentiable(prefix_op_name("CuckooHashTableOfTensors"))
