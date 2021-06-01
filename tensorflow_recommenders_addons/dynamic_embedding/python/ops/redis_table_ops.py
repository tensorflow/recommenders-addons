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
"""Redis Lookup operations."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools
from hashlib import new

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import default
from tensorflow.python.ops.lookup_ops import LookupInterface
from tensorflow.python.training.saver import BaseSaverBuilder

from tensorflow_recommenders_addons.utils.resource_loader import LazySO

redis_table_ops = LazySO(
    "dynamic_embedding/core/_redis_table_ops.so").ops

class RedisTable(LookupInterface):
  """A generic mutable hash table implementation.

    Data can be inserted by calling the insert method and removed by calling the
    remove method. It does not support initialization via the init method.

    Example usage:

    ```python
    table = tfra.dynamic_embedding.RedisTable(key_dtype=tf.string,
                                                   value_dtype=tf.int64,
                                                   default_value=-1)
    sess.run(table.insert(keys, values))
    out = table.lookup(query_keys)
    print(out.eval())
    ```
    """

  default_redis_params={
    "connection_mode":1,
    "master_name":"master",
    # connection_options
    "host_name":"127.0.0.1",
    "host_port":26379,
    "password":"",
    "db":0,
		"connect_timeout":100,  # milliseconds
		"socket_timeout":100,  #  milliseconds
		#  connection_pool_options
		"size":10,
		"wait_timeout":100,		  #  milliseconds
		"connection_lifetime":10,  # minutes
		#  sentinel_connection_options
		"sentinel_connect_timeout":200,  # milliseconds
		"sentinel_socket_timeout":200,	#  milliseconds
    #  model_tag for version and any other information
    "model_tag":"test",
  }

  def __init__(
      self,
      key_dtype,
      value_dtype,
      default_value,
      name="RedisTable",
      checkpoint=False,
      init_size=0,
      params={},
  ):
    """Creates an empty `RedisTable` object.

        Creates a redis table through OS envionment variables, 
        the type of its keys and values are specified by key_dtype 
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
          A `RedisTable` object.

        Raises:
          ValueError: If checkpoint is True and no name was specified.
        """
    self._default_value = ops.convert_to_tensor(default_value,
                                                dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._init_size = init_size
    self._name = name

    self._default_redis_params = self.default_redis_params.copy()
    self._default_redis_params = {k:v for k, v in params.items() if k in self._default_redis_params}

    for k, v in self._default_redis_params.items():
      if not isinstance(v, str):
        v = str(v)
      os.environ[k]=v

    self._shared_name = None
    if context.executing_eagerly():
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      # TODO(rohanj): Use context.shared_name() instead.
      self._shared_name = "table_%d" % (ops.uid(),)
    super(RedisTable, self).__init__(key_dtype, value_dtype)

    self._resource_handle = self._create_resource()
    if checkpoint:
      _ = RedisTable._Saveable(self, name)
      if not context.executing_eagerly():
        self.saveable = RedisTable._Saveable(
            self,
            name=self._resource_handle.op.name,
            full_name=self._resource_handle.op.name,
        )
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self.saveable)
      else:
        self.saveable = RedisTable._Saveable(self,
                                                  name=name,
                                                  full_name=name)

  def _create_resource(self):
    # The table must be shared if checkpointing is requested for multi-worker
    # training to work correctly. Use the node name if no shared_name has been
    # explicitly specified.
    use_node_name_sharing = self._checkpoint and self._shared_name is None
    table_ref = redis_table_ops.tfra_redis_table_of_tensors(
        shared_name=self._shared_name,
        use_node_name_sharing=use_node_name_sharing,
        key_dtype=self._key_dtype,
        value_dtype=self._value_dtype,
        value_shape=self._default_value.get_shape(),
        init_size=self._init_size,
        embedding_name=self._name,
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
        return redis_table_ops.tfra_redis_table_size(
            self.resource_handle)

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
      op = redis_table_ops.tfra_redis_table_remove(
          self.resource_handle, keys)

    return op

  def lookup(self, keys, dynamic_default_values=None, name=None):
    """Looks up `keys` in a table, outputs the corresponding values.

        The `default_value` is used for keys not present in the table.

        Args:
          keys: Keys to look up. Can be a tensor of any shape. Must match the
            table's key_dtype.
          dynamic_default_values: The values to use if a key is missing in the
            table. If None (by default), the static default_value
            `self._default_value` will be used.
          name: A name for the operation (optional).

        Returns:
          A tensor containing the values in the same shape as `keys` using the
            table's value type.

        Raises:
          TypeError: when `keys` do not match the table data types.
        """
    with ops.name_scope(
        name,
        "%s_lookup_table_find" % self.name,
        (self.resource_handle, keys, self._default_value),
    ):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = redis_table_ops.tfra_redis_table_find(
            self.resource_handle,
            keys,
            dynamic_default_values
            if dynamic_default_values is not None else self._default_value,
        )
    return values

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
      with ops.colocate_with(self.resource_handle):
        # pylint: disable=protected-access
        op = redis_table_ops.tfra_redis_table_insert(
            self.resource_handle, keys, values)
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
        (
            exported_keys,
            exported_values,
        ) = redis_table_ops.tfra_redis_table_export(
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
                RedisTable._Saveable,
                table=self,
                name=self._name,
                full_name=full_name,
            )
    }

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for RedisTable."""

    def __init__(self, table, name, full_name=""):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values"),
      ]
      # pylint: disable=protected-access
      super(RedisTable._Saveable, self).__init__(table, specs, name)
      self.full_name = full_name

    def restore(self, restored_tensors, restored_shapes, name=None):
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope(name, "%s_table_restore" % self.name):
        with ops.colocate_with(self.op.resource_handle):
          return redis_table_ops.tfra_redis_table_import(
              self.op.resource_handle,
              restored_tensors[0],
              restored_tensors[1],
          )


ops.NotDifferentiable("TFRA>RedisTableOfTensors")