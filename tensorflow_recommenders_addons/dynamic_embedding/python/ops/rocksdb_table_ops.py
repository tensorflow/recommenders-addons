# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""RocksDB Lookup operations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops.lookup_ops import LookupInterface
from tensorflow.python.training.saver import BaseSaverBuilder

from tensorflow_recommenders_addons.utils.resource_loader import LazySO
from tensorflow_recommenders_addons.utils.resource_loader import prefix_op_name

rocksdb_table_ops = LazySO("dynamic_embedding/core/_rocksdb_table_ops.so").ops


class RocksDBTable(LookupInterface):
  """
    Transparently redirects the lookups to a RocksDB database.

    Data can be inserted by calling the insert method and removed by calling the
    remove method. Initialization via the init method is not supported.

    Example usage:

    ```python
    table = tfra.dynamic_embedding.RocksDBTable(key_dtype=tf.string,
                                                value_dtype=tf.int64,
                                                default_value=-1)
    sess.run(table.insert(keys, values))
    out = table.lookup(query_keys)
    print(out.eval())
    ```
  """

  default_rocksdb_params = {"model_lib_abs_dir": "/tmp/"}

  def __init__(
      self,
      key_dtype,
      value_dtype,
      default_value,
      database_path,
      embedding_name=None,
      read_only=False,
      estimate_size=False,
      export_path=None,
      name="RocksDBTable",
      checkpoint=False,
  ):
    """
      Creates an empty `RocksDBTable` object.

      Creates a RocksDB table through OS environment variables, the type of its keys and values
      are specified by key_dtype and value_dtype, respectively.

      Args:
          key_dtype: the type of the key tensors.
          value_dtype: the type of the value tensors.
          default_value: The value to use if a key is missing in the table.
          name: A name for the operation (optional, usually it's embedding table name).
          checkpoint: if True, the contents of the table are saved to and restored
              from a RocksDB binary dump files according to the directory "[model_lib_abs_dir]/[model_tag]/[name].rdb".
          If `shared_name` is empty for a checkpointed table, it is shared using the table node name.

      Returns:
          A `RocksDBTable` object.

      Raises:
          ValueError: If checkpoint is True and no name was specified.
    """

    self._default_value = ops.convert_to_tensor(default_value,
                                                dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._name = name
    self._database_path = database_path
    self._embedding_name = embedding_name if embedding_name else self._name.split(
        '_mht_', 1)[0]
    self._read_only = read_only
    self._estimate_size = estimate_size
    self._export_path = export_path

    self._shared_name = None
    if context.executing_eagerly():
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      # TODO(rohanj): Use context.shared_name() instead.
      self._shared_name = "table_%d" % (ops.uid(),)
    super().__init__(key_dtype, value_dtype)

    self._resource_handle = self._create_resource()
    if checkpoint:
      _ = self._Saveable(self, name)
      if not context.executing_eagerly():
        self.saveable = self._Saveable(
            self,
            name=self._resource_handle.op.name,
            full_name=self._resource_handle.op.name,
        )
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self.saveable)
      else:
        self.saveable = self._Saveable(self, name=name, full_name=name)

  def _create_resource(self):
    # The table must be shared if checkpointing is requested for multi-worker
    # training to work correctly. Use the node name if no shared_name has been
    # explicitly specified.
    use_node_name_sharing = self._checkpoint and self._shared_name is None

    table_ref = rocksdb_table_ops.tfra_rocksdb_table_of_tensors(
        shared_name=self._shared_name,
        use_node_name_sharing=use_node_name_sharing,
        key_dtype=self._key_dtype,
        value_dtype=self._value_dtype,
        value_shape=self._default_value.get_shape(),
        database_path=self._database_path,
        embedding_name=self._embedding_name,
        read_only=self._read_only,
        estimate_size=self._estimate_size,
        export_path=self._export_path,
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
    """
        Compute the number of elements in this table.

        Args:
            name: A name for the operation (optional).

        Returns:
            A scalar tensor containing the number of elements in this table.
        """
    print('SIZE CALLED')
    with ops.name_scope(name, f"{self.name}_Size", (self.resource_handle,)):
      with ops.colocate_with(self.resource_handle):
        size = rocksdb_table_ops.tfra_rocksdb_table_size(self.resource_handle)

    return size

  def remove(self, keys, name=None):
    """
        Removes `keys` and its associated values from the table.

        If a key is not present in the table, it is silently ignored.

        Args:
            keys: Keys to remove. Can be a tensor of any shape. Must match the table's key type.
            name: A name for the operation (optional).

        Returns:
            The created Operation.

        Raises:
            TypeError: when `keys` do not match the table data types.
        """
    print('REMOVE CALLED')
    if keys.dtype != self._key_dtype:
      raise TypeError(
          f"Signature mismatch. Keys must be dtype {self._key_dtype}, got {keys.dtype}."
      )

    with ops.name_scope(
        name,
        f"{self.name}_lookup_table_remove",
        (self.resource_handle, keys, self._default_value),
    ):
      op = rocksdb_table_ops.tfra_rocksdb_table_remove(self.resource_handle,
                                                       keys)

    return op

  def clear(self, name=None):
    """
        Clear all keys and values in the table.

        Args:
            name: A name for the operation (optional).

        Returns:
            The created Operation.
        """
    print('CLEAR CALLED')
    with ops.name_scope(name, f"{self.name}_lookup_table_clear",
                        (self.resource_handle, self._default_value)):
      op = rocksdb_table_ops.tfra_rocksdb_table_clear(
          self.resource_handle,
          key_dtype=self._key_dtype,
          value_dtype=self._value_dtype)

    return op

  def lookup(self, keys, dynamic_default_values=None, name=None):
    """
        Looks up `keys` in a table, outputs the corresponding values.

        The `default_value` is used for keys not present in the table.

        Args:
            keys: Keys to look up. Can be a tensor of any shape. Must match the
                table's key_dtype.
            dynamic_default_values: The values to use if a key is missing in the table. If None (by
                default), the static default_value `self._default_value` will be used.
            name: A name for the operation (optional).

        Returns:
            A tensor containing the values in the same shape as `keys` using the table's value type.

        Raises:
            TypeError: when `keys` do not match the table data types.
        """
    print('LOOKUP CALLED')
    with ops.name_scope(name, f"{self.name}_lookup_table_find",
                        (self.resource_handle, keys, self._default_value)):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = rocksdb_table_ops.tfra_rocksdb_table_find(
            self.resource_handle,
            keys,
            dynamic_default_values
            if dynamic_default_values is not None else self._default_value,
        )

    return values

  def insert(self, keys, values, name=None):
    """
        Associates `keys` with `values`.

        Args:
            keys: Keys to insert. Can be a tensor of any shape. Must match the table's key type.
        values: Values to be associated with keys. Must be a tensor of the same shape as `keys` and
            match the table's value type.
        name: A name for the operation (optional).

        Returns:
            The created Operation.

        Raises:
            TypeError: when `keys` or `values` doesn't match the table data types.
        """
    print('INSERT CALLED')
    with ops.name_scope(name, f"{self.name}_lookup_table_insert",
                        (self.resource_handle, keys, values)):
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")

      with ops.colocate_with(self.resource_handle):
        op = rocksdb_table_ops.tfra_rocksdb_table_insert(
            self.resource_handle, keys, values)

    return op

  def export(self, name=None):
    """
        Returns nothing in RocksDB Implement. It will dump some binary files to model_lib_abs_dir.

        Args:
            name: A name for the operation (optional).

        Returns:
            A pair of tensors with the first tensor containing all keys and the second tensors
            containing all values in the table.
        """
    print('EXPORT CALLED')
    with ops.name_scope(name, f"{self.name}_lookup_table_export_values",
                        (self.resource_handle,)):
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = rocksdb_table_ops.tfra_rocksdb_table_export(
            self.resource_handle, self._key_dtype, self._value_dtype)

    return exported_keys, exported_values

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    # full_name helps to figure out the name-based Saver's name for this saveable.
    if context.executing_eagerly():
      full_name = self._table_name
    else:
      full_name = self._resource_handle.op.name

    return {
        "table":
            functools.partial(
                self._Saveable,
                table=self,
                name=self._name,
                full_name=full_name,
            )
    }

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for RocksDBTable."""

    def __init__(self, table, name, full_name=""):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values"),
      ]
      super().__init__(table, specs, name)
      self.full_name = full_name

    def restore(self, restored_tensors, restored_shapes, name=None):
      print('RESTORE CALLED')
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope(name, f"{self.name}_table_restore"):
        with ops.colocate_with(self.op.resource_handle):
          return rocksdb_table_ops.tfra_rocksdb_table_import(
              self.op.resource_handle,
              restored_tensors[0],
              restored_tensors[1],
          )


ops.NotDifferentiable(prefix_op_name("RocksDBTableOfTensors"))
