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
"""Ops to use embedding variables as resources."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import compat
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saver import BaseSaverBuilder

from tensorflow_recommenders_addons.embedding_variable.python import gen_ev_ops

__all__ = ["EmbeddingVariable"]


class EmbeddingVariable(resource_variable_ops.ResourceVariable,
                        saveable_object.SaveableObject):
  """Embedding Variable based on resource variable.

  See the ${variables} documentation for more details.

  A `EmbeddingVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `EmbeddingVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the parted variable. After construction, the type and embedding
  dim shape of the variable are fixed. The first demension of the embedding variable
  is mutable. The shape can be changed using read_sparse methods.

  Unlike tf.ResourceVariable, a tf.EmbeddingVariable is mutable. the shape of the
  EmbeddingVariable means the embedding dim, user can use the APIs(sparse_read()) to
  change the whole shape of the EmbeddingVariable. When read_sparse(index=i, ...) is
  called, if the i-th embedding value doesn't exist, it will be initialized and return,
   else it will return the i-th existing embedding value, when the embedding variable
  is updated by back propagation, the i-th embedding value will be updated or removed.

  For example:

   ```python
    a = tf.EmbeddingVariable([1.0, 3.0, 5.0])
    a.initializer.run()

    b = a.sparse_read([2])

    tf.Print(b, [b]).run()  # Will print 1.0, 3.0, 5.0
  ```

  """

  def __init__(self,
               embedding_dim,
               initializer,
               trainable=True,
               collections=None,
               caching_device=None,
               name=None,
               ktype=None,
               vtype=None,
               variable_def=None,
               import_scope=None,
               constraint=None,
               distribute_strategy=None,
               synchronization=None,
               aggregation=None,
               invalid_key=-1):
    """Creates a variable.

    Args:
      embedding_dim: EmbeddingVarible's dimension.
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      name: Optional name for the variable. Defaults to `'EmbeddingVariable'`
        and gets uniquified automatically.
      ktype: If set, EV's key will be converted to the given type.
        If None, int32 will be used.
      vtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
        a Tensor) or float32 will be used (if it is a Python object convertible
        to a Tensor).
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `EmbeddingVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        EmbeddingVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is None, which signifies that this EmbeddingVariable will not be added to any
    collections.
    @end_compatibility
    """
    if variable_def:
      if context.executing_eagerly():
        raise ValueError("Creating EmbeddingVariable from variable_def is "
                         "not supported when eager execution is enabled.")
      self._init_from_proto(variable_def, import_scope=import_scope)
    else:
      self._init_from_args(embedding_dim=embedding_dim,
                           initializer=initializer,
                           trainable=trainable,
                           collections=collections,
                           caching_device=caching_device,
                           name=name,
                           ktype=ktype,
                           vtype=vtype,
                           constraint=constraint,
                           synchronization=synchronization,
                           aggregation=aggregation,
                           distribute_strategy=distribute_strategy,
                           invalid_key=invalid_key)

  def __repr__(self):
    return "<tf.EmbeddingVariable '%s' embedding dim=%s ktype=%s vtype=%s>" % (
        self.name, self.shape, self._ktype.name, self.dtype.name)

  # LINT.IfChange
  # _VariableFromResource inherits from EmbeddingVariable but
  # doesn't call the constructor, so changes here might need to be reflected
  # there.
  # pylint: disable=unused-argument
  def _init_from_args(self,
                      embedding_dim,
                      initializer=None,
                      trainable=True,
                      collections=None,
                      caching_device=None,
                      name=None,
                      ktype=None,
                      vtype=None,
                      constraint=None,
                      synchronization=None,
                      aggregation=None,
                      distribute_strategy=None,
                      invalid_key=-1):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the EmbeddingVariable. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      name: Optional name for the variable. Defaults to `'EmbeddingVariable'` and gets
        uniquified automatically.
      ktype: If set, EV's key will be converted to the given type.
        If None, int32 will be used.
      vtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
        a Tensor) or float32 will be used (if it is a Python object convertible
        to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the GLOBAL_VARIABLES or TRAINABLE_VARIABLES
    collections, and the `collections` argument is ignored.
    @end_compatibility
    """
    if isinstance(embedding_dim, tensor_shape.TensorShape):
      embedding_shape = embedding_dim
    elif isinstance(embedding_dim, six.integer_types):
      embedding_shape = [embedding_dim]

    initial_value = initializer(shape=embedding_shape)
    init_from_fn = callable(initial_value)
    if ktype is None:
      ktype = dtypes.int32

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to EmbeddingVariable constructor must be a list, tuple, "
          "or set. Got %s of type %s" % (collections, type(collections)))
    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    self._initializer = initializer
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]

    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(name,
                          "EmbeddingVariable",
                          [] if init_from_fn else [initial_value],
                          skip_on_eager=False) as name:
        # pylint: disable=protected-access
        self._invalid_key = invalid_key
        self._ktype = ktype
        handle_name = ops.name_from_scope_name(name)
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          # When in eager mode use a uid for the shared_name, to prevent
          # accidental sharing.
          unique_id = "%s_%d" % (handle_name, ops.uid())
          shared_name = unique_id
        # Use attr_scope and device(None) to simulate the behavior of
        # colocate_with when the variable we want to colocate with doesn't
        # yet exist.
        device_context_manager = (ops.device if self._in_graph_mode else
                                  ops.NullContextmanager)
        attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
            s=[compat.as_bytes("loc:@%s" % handle_name)]))
        with ops.get_default_graph()._attr_scope({"_class": attr}):
          with ops.name_scope("Initializer"), device_context_manager(None):
            if init_from_fn:
              initial_value = initial_value()
            if isinstance(initial_value, trackable.CheckpointInitialValue):
              self._maybe_initialize_trackable()
              self._update_uid = initial_value.checkpoint_position.restore_uid
              initial_value = initial_value.wrapped_value
            initial_value = ops.convert_to_tensor(initial_value,
                                                  name="initial_value",
                                                  dtype=vtype)
          shape = initial_value.shape
          handle = self._embedding_variable_handle(
              shape=initial_value.get_shape(),
              dtype=initial_value.dtype.base_dtype,
              shared_name=shared_name,
              name=name,
              graph_mode=self._in_graph_mode)
        # pylint: disable=protected-access
        if (self._in_graph_mode and initial_value is not None
            and initial_value.op._get_control_flow_context() is not None):
          raise ValueError(
              "Initializer for variable %s is from inside a control-flow "
              "construct, such as a loop or conditional. When creating a "
              "variable inside a loop or conditional, use a lambda as the "
              "initializer." % name)
        # pylint: enable=protected-access
        vtype = initial_value.dtype.base_dtype

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            self._ev_is_initialized_op = (gen_ev_ops.ev_is_initialized_op(
                handle, Tkey=self._ktype, Tvalue=vtype))
          if initial_value is not None:
            # pylint: disable=g-backslash-continuation
            with ops.name_scope("Initialize") as n, \
                 ops.colocate_with(None, ignore_existing=True), \
                 ops.device(handle.device):
              # pylint: disable=protected-access
              initializer_op = (gen_ev_ops.initialize_ev_op(
                  handle,
                  variables._try_guard_against_uninitialized_dependencies(
                      name, initial_value),
                  ops.convert_to_tensor(invalid_key, dtype=self._ktype),
                  shape=initial_value.get_shape(),
                  name=n))
          cached_value = None
          graph_element = None
        else:
          gen_ev_ops.initialize_ev_op(handle,
                                      initial_value,
                                      ops.convert_to_tensor(invalid_key,
                                                            dtype=self._ktype),
                                      shape=initial_value.get_shape())
          self._ev_is_initialized_op = None
          initializer_op = None
          graph_element = None
          cached_value = None

        if not context.executing_eagerly():
          # Eager variables are only added to collections if they are part of an
          # eager variable store (otherwise in an interactive session they would
          # hog memory and cause OOM). This is done in ops/variable_scope.py.
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
      initial_value = initial_value if self._in_graph_mode else None
      new_dim = shape.as_list()
      new_dim.insert(0, 0)
      new_shape = tensor_shape.TensorShape(new_dim)
      super(resource_variable_ops.ResourceVariable,
            self).__init__(trainable=trainable,
                           shape=new_shape,
                           dtype=vtype,
                           handle=handle,
                           synchronization=synchronization,
                           constraint=constraint,
                           aggregation=aggregation,
                           distribute_strategy=distribute_strategy,
                           name=name,
                           unique_id=unique_id,
                           handle_name=handle_name,
                           graph_element=graph_element,
                           initial_value=initial_value,
                           initializer_op=initializer_op,
                           is_initialized_op=self._ev_is_initialized_op,
                           cached_value=cached_value,
                           caching_device=caching_device)

    tensors = gen_ev_ops.ev_export(self.handle, Tkey=self._ktype, Tvalue=vtype)
    self.specs = [
        BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
        BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values"),
    ]

  def restore(self, restored_tensors, restored_shapes):
    with ops.control_dependencies([self.initializer]):
      return gen_ev_ops.ev_import(self.handle, restored_tensors[0],
                                  restored_tensors[1])

  def _init_from_proto(self, variable_def, import_scope=None):
    """Initializes from `VariableDef` proto."""
    # Note that init_from_proto is currently not supported in Eager mode.
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError("Trying to restore Variable as EmbeddingVariable.")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
        ops.prepend_name_scope(variable_def.variable_name,
                               import_scope=import_scope))
    self._graph_shape = tensor_shape.TensorShape(
        self._handle.op.get_attr("shape"))
    self._handle_device = self._handle.device
    self._handle_name = self._handle.name
    self._initializer_op = g.as_graph_element(
        ops.prepend_name_scope(variable_def.initializer_name,
                               import_scope=import_scope))
    self._trainable = getattr(variable_def, "trainable", True)
    if variable_def.snapshot_name:
      self._cached_value = g.as_graph_element(
          ops.prepend_name_scope(variable_def.snapshot_name,
                                 import_scope=import_scope))
    else:
      self._cached_value = None
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
          save_slice_info_def=variable_def.save_slice_info_def)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("Tvalue"))
    self._invalid_key = -1
    self._initial_value = ops.convert_to_tensor([0],
                                                name="initial_value",
                                                dtype=self._dtype)
    self._ktype = dtypes.as_dtype(self._handle.op.get_attr("Tkey"))
    self._graph_element = None
    self._constraint = None

  def total_count(self):
    """The shape of this variable."""
    return gen_ev_ops.ev_shape(self._handle,
                               Tkey=self._ktype,
                               Tvalue=self.dtype)

  @property
  def invalid_key(self):
    return self._invalid_key

  def value(self):
    raise NotImplementedError("EmbeddingVariable does not implement value()")

  def eval(self, session=None):
    raise NotImplementedError("EmbeddingVariable does not implement eval()")

  def _set_save_slice_info(self, save_slice_info):
    """Sets the slice info for this `EmbeddingVariable`.

    Args:
      save_slice_info: A `Variable.SaveSliceInfo` object.
    """
    self._save_slice_info = save_slice_info

  def _get_save_slice_info(self):
    return self._save_slice_info

  def _read_variable_op(self):
    raise NotImplementedError(
        "EmbeddingVariable does not implement _read_variable_op()")

  def read_value(self):
    raise NotImplementedError(
        "EmbeddingVariable does not implement read_value()")

  def sparse_read(self, indices, name=None):
    """Reads the value of this variable sparsely, using `gather`."""
    if indices.dtype != self._ktype:
      raise errors_impl.InvalidArgumentError(
          None, None,
          "type of indices is not match with EmbeddingVariable key type.")
    with ops.name_scope("Gather" if name is None else name) as name:
      resource_variable_ops.variable_accessed(self)
      default_value = self._initializer(array_ops.concat(
          [array_ops.shape(indices),
           self.shape.as_list()[1:]], axis=0),
                                        dtype=self.dtype)
      value = gen_ev_ops.ev_gather(self._handle,
                                   indices,
                                   default_value,
                                   name=name)
    return array_ops.identity(value)

  def to_proto(self, export_scope=None):
    """Converts a `EmbeddingVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `EmbeddingVariable` is not
      in the specified name scope.
    """
    if context.executing_eagerly():
      raise RuntimeError("to_proto not supported in EAGER mode.")
    if export_scope is None or self.handle.name.startswith(export_scope):
      var_def = variable_pb2.VariableDef()
      var_def.variable_name = ops.strip_name_scope(self.handle.name,
                                                   export_scope)
      var_def.initializer_name = ops.strip_name_scope(self.initializer.name,
                                                      export_scope)
      if self._initial_value is not None:
        var_def.initial_value_name = ops.strip_name_scope(
            self._initial_value.name, export_scope)
      if self._cached_value is not None:
        var_def.snapshot_name = ops.strip_name_scope(self._cached_value.name,
                                                     export_scope)
      var_def.is_resource = True
      if self._save_slice_info:
        var_def.save_slice_info_def.MergeFrom(
            self._save_slice_info.to_proto(export_scope=export_scope))
      return var_def
    else:
      return None

  @staticmethod
  def from_proto(variable_def, import_scope=None):
    if context.executing_eagerly():
      raise RuntimeError("from_proto not supported in EAGER mode.")
    return EmbeddingVariable(variable_def=variable_def,
                             import_scope=import_scope)

  def is_initialized(self, name=None):
    return self._ev_is_initialized_op

  def assign_sub(self, delta, use_locking=None, name=None):
    raise NotImplementedError(
        "EmbeddingVariable does not implement assign_sub()")

  def assign_add(self, delta, use_locking=None, name=None):
    raise NotImplementedError(
        "EmbeddingVariable does not implement assign_add()")

  def assign(self, value, use_locking=None, name=None):
    raise NotImplementedError("EmbeddingVariable does not implement assign()")

  def _embedding_variable_handle(self, shape, dtype, shared_name, name,
                                 graph_mode):
    """Creates a variable handle with information to do shape inference."""
    container = ops.get_default_graph()._container  # pylint: disable=protected-access
    if container is None:
      container = ""
    return gen_ev_ops.ev_handle_op(shape=shape,
                                   shared_name=shared_name,
                                   name=name,
                                   Tkey=self._ktype,
                                   Tvalue=dtype,
                                   container=container)


# Register a conversion function which reads the value of the variable,
# allowing instances of the class to be used as tensors.

# Note: registering for Variable after EmbeddingVariable because inheritance will
# otherwise lead to the wrong behavior.
#ops.register_tensor_conversion_function(EmbeddingVariable, _dense_var_to_tensor)
ops.register_tensor_conversion_function(
    variables.Variable, variables.Variable._TensorConversionFunction)  # pylint: disable=protected-access


@ops.RegisterGradient("EVGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = gen_ev_ops.ev_shape(handle,
                                     Tkey=indices.dtype,
                                     Tvalue=grad.dtype)
  size = array_ops.expand_dims(array_ops.size(indices), 0)
  values_shape = array_ops.concat([size, params_shape[1:]], 0)
  values = array_ops.reshape(grad, values_shape)
  indices = array_ops.reshape(indices, size)
  return [ops.IndexedSlices(values, indices, params_shape), None, None]
