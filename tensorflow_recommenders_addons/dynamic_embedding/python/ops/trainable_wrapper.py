# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Trainable Wrapper implement."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_resource_variable_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import compat


class TrainableWrapper(resource_variable_ops.ResourceVariable):
  """
  This class is a trainable wrapper of Dynamic Embedding,
  and the key role is recording the map relation between params and ids.
  inheriting from the ResourceVariable make it trainable.
  """
  def __getattribute__(self, name):
    if name in ['sparse_read', 'gather_nd']:
        raise AttributeError('no such method: {}'.format(name))
    return super(resource_variable_ops.ResourceVariable, self).__getattribute__(name)

  def __init__(self, params, ids, max_norm, *args, **kwargs):
    """Creates an empty `TrainableWrapper` object.

    Creates a group of tables placed on devices,
    the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      params: A dynamic_embedding.Variable instance.
      ids: a tensor with any shape as same dtype of params.key_dtype.
      max_norm: If not `None`, each values is clipped if its l2-norm is larger
        than this value.
      other parameters is same with ResourceVariable.
    Returns:
      A `TrainableWrapper` object which is a subclass of ResourceVariable.
    """
    self.params = params
    self.ids = ids
    self.max_norm = max_norm
    self.prefetch_values_op = None
    super(TrainableWrapper, self).__init__(*args, **kwargs)

  def prefetch_values(self):
    if self.prefetch_values_op is None:
      self.prefetch_values_op = self.transform(self.params.lookup(self.ids))
    return self.prefetch_values_op

  def _init_from_args(self,
                      initial_value=None,
                      trainable=None,
                      collections=None,
                      caching_device=None,
                      name=None,
                      dtype=None,
                      constraint=None,
                      synchronization=None,
                      aggregation=None,
                      distribute_strategy=None,
                      shape=None):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
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
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: DistributionStrategy under which this variable
        was created.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the `GLOBAL_VARIABLES` or
    `TRAINABLE_VARIABLES` collections, and the `collections` argument is
    ignored.
    @end_compatibility
    """
    synchronization, aggregation, trainable = (
      variables.validate_synchronization_aggregation_trainable(
        synchronization, aggregation, trainable, name))
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if isinstance(initial_value, ops.Tensor) and hasattr(
        initial_value, "graph") and initial_value.graph.building_function:
      raise ValueError("Tensor-typed variable initializers must either be "
                       "wrapped in an init_scope or callable "
                       "(e.g., `tf.Variable(lambda : "
                       "tf.truncated_normal([10, 40]))`) when building "
                       "functions. Please file a feature request if this "
                       "restriction inconveniences you.")

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
        "collections argument to Variable constructor must be a list, tuple, "
        "or set. Got %s of type %s" % (collections, type(collections)))
    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if isinstance(initial_value, trackable.CheckpointInitialValue):
      self._maybe_initialize_trackable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(name, "TrainableWrapper", []
      if init_from_fn else [initial_value]) as name:
        # pylint: disable=protected-access
        handle_name = ops.name_from_scope_name(name)
        handle_name = (handle_name or "TrainableWrapperHandle")
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          # When in eager mode use a uid for the shared_name, to prevent
          # accidental sharing.
          unique_id = "%s_%d" % (handle_name, ops.uid())
          shared_name = context.shared_name()
        # Use attr_scope and device(None) to simulate the behavior of
        # colocate_with when the variable we want to colocate with doesn't
        # yet exist.
        device_context_manager = (
          ops.device if self._in_graph_mode else ops.NullContextmanager)
        attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
            s=[compat.as_bytes("loc:@%s" % handle_name)]))
        with ops.get_default_graph()._attr_scope({"_class": attr}):
          with ops.name_scope("Initializer"), device_context_manager(None):
            initial_value = ops.convert_to_tensor(
              initial_value() if init_from_fn else initial_value,
              name="initial_value", dtype=dtype)
          if shape is None:
            shape = initial_value.shape
          handle = resource_variable_ops.eager_safe_variable_handle(
            initial_value=initial_value,
            shape=None,  # shape,
            shared_name=shared_name,
            name=name,
            graph_mode=self._in_graph_mode)
        # pylint: disable=protected-access
        if (self._in_graph_mode and initial_value is not None and
            initial_value.op._get_control_flow_context() is not None):
          raise ValueError(
            "Initializer for variable %s is from inside a control-flow "
            "construct, such as a loop or conditional. When creating a "
            "variable inside a loop or conditional, use a lambda as the "
            "initializer." % name)
        # pylint: enable=protected-access
        dtype = initial_value.dtype.base_dtype

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            is_initialized_op = (
              gen_resource_variable_ops.var_is_initialized_op(handle))
          if initial_value is not None:
            # pylint: disable=g-backslash-continuation
            with ops.name_scope("Assign") as n, \
                ops.colocate_with(None, ignore_existing=True), \
                ops.device(handle.device):
              # pylint: disable=protected-access
              initializer_op = (
                gen_resource_variable_ops.assign_variable_op(
                  handle,
                  variables._try_guard_against_uninitialized_dependencies(
                    name,
                    initial_value),
                  name=n))
              # pylint: enable=protected-access
            # pylint: enable=g-backslash-continuation
          with ops.name_scope("Read"):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(handle.device):
              with ops.control_dependencies([
                gen_resource_variable_ops.assign_variable_op(
                  handle, self.prefetch_values(),
                  name="AssignBeforeInitRead")]):
                value = gen_resource_variable_ops.read_variable_op(handle,
                                                                   dtype)
            graph_element = value
            if caching_device is not None:
              # Variables may be created in a tf.device() or ops.colocate_with()
              # context. At the same time, users would expect caching device to
              # be independent of this context, and/or would not expect the
              # current device context to be merged with the caching device
              # spec.  Therefore we reset the colocation stack before creating
              # the cached value. Note that resetting the colocation stack will
              # also reset the device stack.
              with ops.colocate_with(None, ignore_existing=True):
                with ops.device(caching_device):
                  cached_value = array_ops.identity(value)
            else:
              cached_value = None
        else:
          gen_resource_variable_ops.assign_variable_op(handle, initial_value)
          is_initialized_op = None
          initializer_op = None
          graph_element = None
          if caching_device:
            with ops.device(caching_device):
              with ops.control_dependencies([
                gen_resource_variable_ops.assign_variable_op(
                  handle, self.prefetch_values(),
                  name="AssignBeforeInitRead")]):
                cached_value = gen_resource_variable_ops.read_variable_op(
                  handle, dtype)
          else:
            cached_value = None
        if not context.executing_eagerly():
          # Eager variables are only added to collections if they are part of an
          # eager variable store (otherwise in an interactive session they would
          # hog memory and cause OOM). This is done in ops/variable_scope.py.
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
      initial_value = initial_value if self._in_graph_mode else None
      super(resource_variable_ops.ResourceVariable, self).__init__(
        trainable=trainable, shape=shape, dtype=dtype, handle=handle,
        synchronization=synchronization, constraint=constraint,
        aggregation=aggregation, distribute_strategy=distribute_strategy,
        name=name, unique_id=unique_id, handle_name=handle_name,
        graph_element=graph_element, initial_value=initial_value,
        initializer_op=initializer_op, is_initialized_op=is_initialized_op,
        cached_value=cached_value)

  def update_op(self):
    return self.params.upsert(self.ids, self.read_value(False))

  def size(self):
    return self.params.size()

  def _read_variable_op(self, do_prefetch=True):
    resource_variable_ops.variable_accessed(self)
    if do_prefetch:
      with ops.control_dependencies([
          gen_resource_variable_ops.assign_variable_op(
            self._handle, self.prefetch_values(), name="AssignBeforeReadVariable")]):
        _result = gen_resource_variable_ops.read_variable_op(self._handle,
                                                             self._dtype)
    else:
      _result = gen_resource_variable_ops.read_variable_op(self._handle,
                                                           self._dtype)

    if not context.executing_eagerly():
      # Note that if a control flow context is active the input of the read op
      # might not actually be the handle. This line bypasses it.
      tape.record_operation(
        "ReadVariableOp", [_result], [self._handle], lambda x: [x])
    result = self.transform(_result)
    return result

  def read_value(self, do_prefetch=True):
    """Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.
    Args:
      do_prefetch: get value from `params` before reading, if True

    Returns:
     the read operation.
    """
    with ops.name_scope("Read"):
      # Ensure we read the variable in the same device as the handle.
      with ops.device(self._handle.device):
        value = self._read_variable_op(do_prefetch)
    # Return an identity so it can get placed on whatever device the context
    # specifies instead of the device where the variable is.
    return array_ops.identity(value)

  @staticmethod
  def _clip(params, ids, max_norm):
    def _rank(x):
      rank = ops.convert_to_tensor(x).get_shape().ndims
      if rank:
        return rank, True
      else:
        return array_ops.rank(x), False

    if max_norm is None:
      return params
    ids_rank, ids_static = _rank(ids)
    params_rank, params_static = _rank(params)
    return clip_ops.clip_by_norm(
      params,
      max_norm,
      axes=(list(range(ids_rank, params_rank)) if ids_static and params_static
            else math_ops.range(ids_rank, params_rank)))

  def transform(self, result):
    _result = array_ops.reshape(result, shape=array_ops.shape(result))
    if self.max_norm is not None:
      _result = self._clip(_result, self.ids, self.max_norm)
    return _result
