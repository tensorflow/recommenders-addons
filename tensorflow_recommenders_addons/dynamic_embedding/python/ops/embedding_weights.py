import abc
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute.coordinator.values import PerWorkerValues
from tensorflow.python.ops.resource_variable_ops import VariableSpec, ResourceVariableGradient, \
  eager_safe_variable_handle, _maybe_set_handle_data, get_eager_safe_handle_data
import contextlib
import weakref
from tensorflow.core.framework import variable_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

from tensorflow_recommenders_addons.utils.resource_loader import get_tf_version_triple
try:
  from tensorflow.python.compat import compat as forward_compat
except:
  forward_compat = None
from tensorflow.python.eager import context
from tensorflow.python.eager import tape as tape_record
if not hasattr(tape_record, 'record_operation'):
  # tf version >= 2.13.0
  from tensorflow.python.eager import record as tape_record
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import clip_ops, handle_data_util
from tensorflow.python.framework import ops, composite_tensor

try:  # tf version >= 2.14.0
  from tensorflow.python.framework.tensor import Tensor
except:
  from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.framework import tensor as tensor_module

try:  # tf version >= 2.14.0
  from tensorflow.python.ops.array_ops_stack import stack
except:
  from tensorflow.python.ops.array_ops import stack
try:  # tf version >= 2.10.0
  from tensorflow.python.trackable import base as trackable
except:
  from tensorflow.python.training.tracking import base as trackable
try:  # The data_structures has been moved to the new package in tf 2.11
  from tensorflow.python.trackable import data_structures
except:
  from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import compat, dispatch

try:  # tf version >= 2.14.0
  from tensorflow.python.distribute import distribute_lib as distribute_ctx, ps_values

  assert hasattr(distribute_ctx, 'has_strategy')
except:
  from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx


class EmbeddingWeights():

  @abc.abstractmethod
  def verify_embedding_weights(self, sparse_ids, sparse_weights=None):
    raise NotImplementedError

  @abc.abstractmethod
  def embedding_lookup(self,
                       ids,
                       name=None,
                       max_norm=None,
                       return_trainable=False
                      ) -> (tf.Tensor, "EmbeddingWeights"):
    """
    embedding lookup, and store the result. No by-product will
    be introduced in this call. So it can be decorated by `tf.function`.

    Args:
      shadow: A ShadowVariable object.
      ids: A tensor with any shape as same dtype of params.key_dtype.
      name: A name for the operation.

    Returns:
      A tensor with shape [shape of ids] + [dim],
        dim is equal to the value dim of params.
        containing the values from the params tensor(s) for keys in ids.
    """
    raise NotImplementedError

  @staticmethod
  def verify_embedding_param_weights(embedding_weights,
                                     sparse_ids,
                                     sparse_weights=None):
    if embedding_weights is None:
      raise ValueError("Missing embedding_weights %s." % embedding_weights)

    if embedding_weights.key_dtype != sparse_ids.dtype:
      raise TypeError(
          "embedding_weights.key_dtype should be same with sparse_ids.dtype: "
          "{} vs. {}".format(embedding_weights.key_dtype, sparse_ids.dtype))

    weights_dtype = sparse_weights.dtype if sparse_weights is not None else None
    if weights_dtype and embedding_weights.value_dtype != weights_dtype:
      raise TypeError(
          "embedding_weights.value_dtype should be same with sparse_weights.dtype"
          ": {} vs. {}".format(embedding_weights.value_dtype, weights_dtype))


class ModelMode(object):
  """The global config of model modes.

    `TrainableWrapper.read_value` is not thread-safe that causes threads
    competition and Out-Of-Bound exception in concurrent serving scenario.

    To resolve this, we define the `ModelMode` APIs to instruct
    the `TrainableWrapper` to build a different thread-safe sub-graph
    for 'TrainableWrapper.read_value' on inference mode.

    **NOTE** These APIs should be called before any graph are built.

  The following standard modes are defined:

  * `TRAIN`: training/fitting mode.
  * `INFERENCE`: prediction/inference mode.
  """

  TRAIN = 'train'
  INFERENCE = 'inference'

  # The default setting is training mode.
  CURRENT_SETTING = TRAIN

class TrainablePerWorker(ps_values.PerWorkerVariable):
  def __init__(self, strategy, next_creator, **kwargs):
    super(TrainablePerWorker,
          self).__init__(strategy, next_creator, **kwargs)
    self._trainable = kwargs.get("trainable", True)
  def read_value(self):
    tfprint = tf.print("TrainablePerWorker read_value:", self._coordinator_instance.device)
    with tf.control_dependencies([tfprint]):
      pass
    with tf.GradientTape() as tape:
      self._maybe_create_per_worker_vars()
      value = super(ps_values.PerWorkerVariable).read_value()
      tape.watch(value)
    return value
  def value(self):
    tfprint = tf.print("TrainablePerWorker value:", self._coordinator_instance.device)
    with tf.control_dependencies([tfprint]):
      pass
    with tf.GradientTape() as tape:
      self._maybe_create_per_worker_vars()
      value = super(ps_values.PerWorkerVariable).value()
      tape.watch(value)
    return value
  def assign(self, value, use_locking=False, name=None, read_value=True):
    self._maybe_create_per_worker_vars()
    tfprint = tf.print("PerWorkerVariable:", value, use_locking, name, read_value)
    with tf.control_dependencies([tfprint]):
      pass
    with ops.device(self._coordinator_instance.device):
      return self._coordinator_instance.assign(
        value, use_locking=use_locking, name=name, read_value=read_value)

  def assign_add(self, delta, use_locking=False, name=None, read_value=True):
    self._maybe_create_per_worker_vars()
    with ops.device(self._coordinator_instance.device):
      return self._coordinator_instance.assign_add(
        delta, use_locking=use_locking, name=name, read_value=read_value)

class PerWorkerResourceVariable(ps_values.PerWorkerVariable, PerWorkerValues):
  def __init__(
      self,  # pylint: disable=super-init-not-called
      initial_value=None,
      trainable=None,
      collections=None,
      validate_shape=True,  # pylint: disable=unused-argument
      caching_device=None,
      name=None,
      dtype=None,
      variable_def=None,
      import_scope=None,
      constraint=None,
      distribute_strategy=None,
      synchronization=None,
      aggregation=None,
      shape=None,
      handle=None,
      experimental_enable_variable_lifting=None,
      **kwargs
  ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. Can also be a callable with
        no argument that returns the initial value when called. (Note that
        initializer functions from init_ops.py must first be bound to a shape
        before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `ResourceVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        ResourceVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      handle: (optional) The handle of a `tf.Variable`. If provided, only
        `trainable`, `shape`, `dtype`, and `handle` will be used to construct
        this `tf.Variable`.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is `None`, which signifies that this `Variable` will not be added to any
    collections.
    @end_compatibility
    """
    if variable_def:
      if initial_value is not None:
        raise ValueError(f"The variable_def and initial_value args to "
                         f"`tf.Variable` are mutually exclusive, but got both: "
                         f"variable_def={variable_def},\n"
                         f"initial_value={initial_value}")
      if context.executing_eagerly():
        raise ValueError(f"Creating a `tf.Variable` with a `variable_def` arg "
                         f"is not supported when eager execution is enabled. "
                         f"Got: variable_def={variable_def}")
      self._init_from_proto(
        variable_def,
        import_scope=import_scope,
        validate_shape=validate_shape)
    elif handle is not None:
      self._init_from_handle(trainable=trainable,
                             shape=shape,
                             dtype=dtype,
                             handle=handle)
    else:
      self._init_from_args(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        name=name,
        dtype=dtype,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation,
        shape=shape,
        distribute_strategy=distribute_strategy,
        validate_shape=validate_shape,
        experimental_enable_variable_lifting=experimental_enable_variable_lifting,
      )
    self._trainable = kwargs.get("trainable", True)


# CompositeTensor method
  @property
  def _type_spec(self):
    return VariableSpec.from_value(self)

  # CompositeTensor method
  def _shape_invariant_to_type_spec(self, shape):
    return VariableSpec(shape, self.dtype, self.trainable)

  # CompositeTensorGradient protocol
  __composite_gradient__ = ResourceVariableGradient()

  def _init_from_args(
      self,
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
      shape=None,
      validate_shape=True,
      experimental_enable_variable_lifting=None,
  ):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound to
        a shape before being used here.)
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
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: DistributionStrategy under which this variable was
        created.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

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
    if experimental_enable_variable_lifting is None:
      experimental_enable_variable_lifting = True
    if initial_value is None:
      raise ValueError("The `initial_value` arg to `tf.Variable` must "
                       "be specified except when you are not providing a "
                       "`variable_def`. You provided neither.")
    init_from_fn = callable(initial_value)

    if isinstance(initial_value, tensor_module.Tensor) and hasattr(
        initial_value, "graph") and initial_value.graph.building_function:
      raise ValueError(f"Argument `initial_value` ({initial_value}) could not "
                       "be lifted out of a `tf.function`. "
                       f"(Tried to create variable with name='{name}'). "
                       "To avoid this error, when constructing `tf.Variable`s "
                       "inside of `tf.function` you can create the "
                       "`initial_value` tensor in a "
                       "`tf.init_scope` or pass a callable `initial_value` "
                       "(e.g., `tf.Variable(lambda : "
                       "tf.truncated_normal([10, 40]))`). "
                       "Please file a feature request if this "
                       "restriction inconveniences you.")

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
        f"collections argument to Variable constructor must be a list, "
        f"tuple, or set. Got {collections} of type {type(collections)}")
    if constraint is not None and not callable(constraint):
      raise ValueError(f"Argument `constraint` must be None or a callable. "
                       f"a callable. Got a {type(constraint)}:  {constraint}")

    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
    if experimental_enable_variable_lifting:
      maybe_init_scope = ops.init_scope
    else:
      maybe_init_scope = contextlib.nullcontext
    with maybe_init_scope():
      with ops.name_scope(
          name,
          "Variable", [] if init_from_fn else [initial_value],
          skip_on_eager=False) as name:
        # pylint: disable=protected-access
        handle_name = ops.name_from_scope_name(name)
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          # When in eager mode, use a uid for the shared_name, to prevent
          # accidental sharing.
          unique_id = "%s_%d" % (handle_name, ops.uid())
          shared_name = None  # Never shared
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
            if init_from_fn:
              initial_value = initial_value()
            if isinstance(initial_value, trackable.CheckpointInitialValue):
              self._maybe_initialize_trackable()
              self._update_uid = initial_value.checkpoint_position.restore_uid
              initial_value = initial_value.wrapped_value
            initial_value = ops.convert_to_tensor(
              initial_value, name="initial_value", dtype=dtype)
          if shape is not None:
            if not initial_value.shape.is_compatible_with(shape):
              raise ValueError(
                f"In this `tf.Variable` creation, the initial value's shape "
                f"({initial_value.shape}) is not compatible with "
                f"the explicitly supplied `shape` argument ({shape}).")
          else:
            shape = initial_value.shape
          handle = eager_safe_variable_handle(
            initial_value=initial_value,
            shape=shape,
            shared_name=shared_name,
            name=name,
            graph_mode=self._in_graph_mode)
          handle._parent_trackable = weakref.ref(self)
          handle._name = handle_name + ":0"
          handle._unique_id = unique_id
        # pylint: disable=protected-access
        if (self._in_graph_mode and initial_value is not None and
            initial_value.op._get_control_flow_context() is not None):
          raise ValueError(
            f"The `initial_value` passed to `tf.Variable` {name} is from "
            f"inside a control-flow  construct, such as a loop or "
            f"conditional. When creating a "
            f"`tf.Variable` inside a loop or conditional, use a lambda as "
            f"the `initial_value`. Got: initial_value=({initial_value})")
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
                    name, initial_value),
                  name=n))
              # pylint: enable=protected-access
            # pylint: enable=g-backslash-continuation
          with ops.name_scope("Read"):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(handle.device):
              value = gen_resource_variable_ops.read_variable_op(handle, dtype)
              _maybe_set_handle_data(dtype, handle, value)
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
              cached_value = gen_resource_variable_ops.read_variable_op(
                handle, dtype)
              _maybe_set_handle_data(dtype, handle, cached_value)
          else:
            cached_value = None

        if cached_value is not None:
          # Store the variable object so that the original variable can be
          # accessed to generate functions that are compatible with SavedModel.
          cached_value._cached_variable = weakref.ref(self)  # pylint: disable=protected-access

        if self._in_graph_mode:
          # Eager variables are only added to collections if they are part of an
          # eager variable store (otherwise in an interactive session they would
          # hog memory and cause OOM). This is done in ops/variable_scope.py.
          ops.add_to_collections(collections, self)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
      initial_value = initial_value if self._in_graph_mode else None
      super(PerWorkerResourceVariable, self).__init__(
        trainable=trainable,
        shape=shape,
        dtype=dtype,
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
        is_initialized_op=is_initialized_op,
        cached_value=cached_value,
        caching_device=caching_device,
        validate_shape=validate_shape,
      )

  def _init_from_proto(self,
                       variable_def,
                       import_scope=None,
                       validate_shape=True):
    """Initializes from `VariableDef` proto."""
    # Note that init_from_proto is currently not supported in Eager mode.
    assert not context.executing_eagerly()
    self._in_graph_mode = True
    assert isinstance(variable_def, variable_pb2.VariableDef)
    if not variable_def.is_resource:
      raise ValueError(f"The `variable_def` you passed to `tf.Variable` is "
                       f"Trying to restore a TF 1.x Reference Variable "
                       f"as a TF 2.x ResourceVariable. This is unsupported. "
                       f"Got variable_def={variable_def}")

    # Create from variable_def.
    g = ops.get_default_graph()
    self._handle = g.as_graph_element(
      ops.prepend_name_scope(
        variable_def.variable_name, import_scope=import_scope),
      allow_operation=False)
    self._shape = tensor_shape.TensorShape(self._handle.op.get_attr("shape"))
    self._handle_name = self._handle.name
    self._unique_id = self._handle_name
    self._initializer_op = g.as_graph_element(
      ops.prepend_name_scope(
        variable_def.initializer_name, import_scope=import_scope))
    # Check whether initial_value_name exists for backwards compatibility.
    if (hasattr(variable_def, "initial_value_name") and
        variable_def.initial_value_name):
      self._initial_value = g.as_graph_element(
        ops.prepend_name_scope(
          variable_def.initial_value_name, import_scope=import_scope))
    else:
      self._initial_value = None
    synchronization, aggregation, trainable = (
      variables.validate_synchronization_aggregation_trainable(
        variable_def.synchronization, variable_def.aggregation,
        variable_def.trainable, variable_def.variable_name))
    self._synchronization = synchronization
    self._aggregation = aggregation
    self._trainable = trainable
    if variable_def.snapshot_name:
      snapshot = g.as_graph_element(
        ops.prepend_name_scope(
          variable_def.snapshot_name, import_scope=import_scope))
      if snapshot.op.type != "ReadVariableOp":
        self._cached_value = snapshot
      else:
        self._cached_value = None
      while snapshot.op.type != "ReadVariableOp":
        snapshot = snapshot.op.inputs[0]
      self._graph_element = snapshot
    else:
      self._cached_value = None
      # Legacy case for protos without the snapshot name; assume it's the
      # following.
      self._graph_element = g.get_tensor_by_name(self._handle.op.name +
                                                 "/Read/ReadVariableOp:0")
    if variable_def.HasField("save_slice_info_def"):
      self._save_slice_info = variables.Variable.SaveSliceInfo(
        save_slice_info_def=variable_def.save_slice_info_def,
        import_scope=import_scope)
    else:
      self._save_slice_info = None
    self._caching_device = None
    self._dtype = dtypes.as_dtype(self._handle.op.get_attr("dtype"))
    self._constraint = None
    self._validate_shape = validate_shape

  def _init_from_handle(self,
                        trainable=None,
                        shape=None,
                        dtype=None,
                        handle=None):
    handle_data = get_eager_safe_handle_data(handle)
    if not handle_data.is_set:
      # The handle may not have the handle shape and dtype if it was created
      # using tf.placeholder.
      handle_data = handle_data_util.create_handle_data(shape, dtype)
      handle_data_util.set_handle_data(handle, handle_data)
    # pylint: disable=protected-access
    if hasattr(handle, "_name") and isinstance(handle._name, str):
      handle_name = handle._name.rstrip(":0")
    else:
      handle_name = None
    # pylint: enable=protected-access
    unique_id = getattr(handle, "_unique_id", None)
    super().__init__(
      trainable=trainable, shape=shape, dtype=dtype, handle=handle,
      unique_id=unique_id, handle_name=handle_name)




class TrainableWrapper(PerWorkerResourceVariable):
  """
    This class is a trainable wrapper of Dynamic Embedding,
    and the key role is recording the map relation between params and ids.
    inheriting from the ResourceVariable make it trainable.
    """

  def __getattribute__(self, name):
    if name in ["sparse_read", "gather_nd"]:
      raise AttributeError("no such method: {}".format(name))

    return super(PerWorkerResourceVariable,
                 self).__getattribute__(name)

  def __init__(self, *args, **kwargs):
    """Creates an empty `TrainableWrapper` object.©

        Creates a group of tables placed on devices,
        the type of its keys and values are specified by key_dtype
        and value_dtype, respectively.

        Args:
          params: A dynamic_embedding.Variable instance.
          ids: A tensor with any shape as same dtype of params.key_dtype.
          max_norm: If not `None`, each values is clipped if its l2-norm is larger
            than this value.
          other parameters is same with ResourceVariable.
        Returns:
          A `TrainableWrapper` object which is a subclass of ResourceVariable.
        """
    self.params = kwargs.pop("params")
    self.ids = kwargs.get("ids")
    self.exists = None
    self.max_norm = kwargs.get("max_norm")
    self.prefetch_values_op = None
    self.model_mode = kwargs.get("model_mode")
    kwargs.pop("model_mode")
    self._tracked_slots = []
    self._optimizer_vars = data_structures.NoDependency([])
    printop = tf.print("st_b:", kwargs, args,
      output_stream=tf.compat.v1.logging.error)
    with tf.control_dependencies([printop]):
      pass
    # strategy = kwargs.pop("distribute_strategy")
    # next_creator = kwargs.pop("next_creator", None)
    super(TrainableWrapper, self).__init__(**kwargs)
    self._trainable = kwargs.get("trainable", True)
    # self._handle = self.handle

  def prefetch_values(self, update=False):
    if update or (self.prefetch_values_op is None):
      if self.params.bp_v2:
        r, self.exists = self.params.lookup(self.ids, return_exists=True)
        self.prefetch_values_op = self.transform(r)
      else:
        self.prefetch_values_op = self.transform(self.params.lookup(self.ids))
    return self.prefetch_values_op

  def __repr__(self):
    if context.executing_eagerly() and not self._in_graph_mode:
      return "<tf.Variable '%s' shape=%s dtype=%s, numpy=%s>" % (
        self.name, self.get_shape(), self.dtype.name,
        ops.numpy_text(self.read_value(), is_repr=True))
    else:
      return "<tf.Variable '%s' shape=%s dtype=%s>" % (
        self.name, self.get_shape(), self.dtype.name)

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
                      shape=None,
                      *args,
                      **kwargs):
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
    (
      synchronization,
      aggregation,
      trainable,
    ) = variables.validate_synchronization_aggregation_trainable(
      synchronization, aggregation, trainable, name)
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if (isinstance(initial_value, Tensor) and hasattr(initial_value, "graph")
        and initial_value.graph.building_function):
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
      with ops.name_scope(name,
                          "TrainableWrapper",
                          [] if init_from_fn else [initial_value],
                          skip_on_eager=False) as name:
        # pylint: disable=protected-access
        handle_name = ops.name_from_scope_name(name)
        handle_name = handle_name or "TrainableWrapperHandle"
        if self._in_graph_mode:
          shared_name = handle_name
          unique_id = shared_name
        else:
          # When in eager mode use a uid for the shared_name, to prevent
          # accidental sharing.
          unique_id = "%s_%d" % (handle_name, ops.uid())
          tf_major_version, _, _ = get_tf_version_triple()
          if int(tf_major_version) >= 2:
            shared_name = None  # Never shared
          else:
            shared_name = context.shared_name()
        # Use attr_scope and device(None) to simulate the behavior of
        # colocate_with when the variable we want to colocate with doesn't
        # yet exist.
        device_context_manager = (ops.device if self._in_graph_mode else
                                  ops.NullContextmanager)
        attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
          s=[compat.as_bytes("loc:@%s" % handle_name)]))
        with ops.get_default_graph()._attr_scope({"_class": attr}):
          with ops.name_scope("Initializer"), device_context_manager(None):
            initial_value = ops.convert_to_tensor(
              initial_value() if init_from_fn else initial_value,
              name="initial_value",
              dtype=dtype,
            )
          if shape is None:
            shape = initial_value.shape
          handle = resource_variable_ops.eager_safe_variable_handle(
            initial_value=initial_value,
            shape=None,  # shape,
            shared_name=shared_name,
            name=name,
            graph_mode=self._in_graph_mode,
          )
        # pylint: disable=protected-access
        if (self._in_graph_mode and initial_value is not None
            and initial_value.op._get_control_flow_context() is not None):
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
            with ops.name_scope("Assign") as n, ops.colocate_with(
                None, ignore_existing=True), ops.device(handle.device):
              # pylint: disable=protected-access
              initializer_op = gen_resource_variable_ops.assign_variable_op(
                handle,
                variables._try_guard_against_uninitialized_dependencies(
                  name, initial_value),
                name=n,
              )
              # pylint: enable=protected-access
            # pylint: enable=g-backslash-continuation
          with ops.name_scope("Read"):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(handle.device):
              with ops.control_dependencies([
                gen_resource_variable_ops.assign_variable_op(
                  handle,
                  self.prefetch_values(),
                  name="AssignBeforeInitRead",
                )
              ]):
                value = gen_resource_variable_ops.read_variable_op(
                  handle, dtype)
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
                  handle,
                  self.prefetch_values(),
                  name="AssignBeforeInitRead",
                )
              ]):
                cached_value = (gen_resource_variable_ops.read_variable_op(
                  handle, dtype))
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
      super(ps_values.PerWorkerVariable, self).__init__(
        trainable=trainable,
        shape=shape,
        dtype=dtype,
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
        is_initialized_op=is_initialized_op,
        cached_value=cached_value,
      )

  def update_op(self, v0=None):
    v1 = self.read_value(False)
    if self.params.bp_v2:
      assert v0 is not None
      update_param_op = self.params.accum(self.ids, v0, v1, self.exists)
    else:
      update_param_op = self.params.upsert(self.ids, v1)
    if self.params.restrict_policy is not None:
      update_status_op = self.params.restrict_policy.apply_update(self.ids)
      return control_flow_ops.group([update_param_op, update_status_op])
    return update_param_op

  def size(self):
    return self.params.size()

  def _read_variable_op(self, do_prefetch=True, no_copy=False):
    resource_variable_ops.variable_accessed(self)
    if no_copy and forward_compat:
      if forward_compat.forward_compatible(2022, 5, 3):
        gen_resource_variable_ops.disable_copy_on_read(self.handle)
    if self.model_mode == "train":
      if do_prefetch:
        with ops.control_dependencies([
          gen_resource_variable_ops.assign_variable_op(
            self._handle,
            self.prefetch_values(),
            name="AssignBeforeReadVariable")
        ]):
          _result = gen_resource_variable_ops.read_variable_op(
            self._handle, self._dtype)
      else:
        _result = gen_resource_variable_ops.read_variable_op(
          self._handle, self._dtype)
    else:
      _result = self.prefetch_values()

    if not context.executing_eagerly():
      # Note that if a control flow context is active the input of the read op
      # might not actually be the handle. This line bypasses it.
      tape_record.record_operation("ReadVariableOp", [_result], [self._handle],
                                   lambda x: [x])
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
            else math_ops.range(ids_rank, params_rank)),
    )

  def transform(self, result):
    if self.max_norm is not None:
      result = self._clip(result, self.ids, self.max_norm)
    return result

  def _track_optimizer_slots(self, slots):
    if not all(isinstance(s, TrainableWrapper) for s in slots):
      raise TypeError(
        'Can only track TrainableWrapper slots, but get {}'.format(
          [type(s) for s in slots]))
    identifiers = [optimizer_v2._var_key(s) for s in self._tracked_slots]
    for s in slots:
      if optimizer_v2._var_key(s) not in identifiers:
        self._tracked_slots.append(s)

    if self.params.restrict_policy is not None:
      self.params.restrict_policy._track_params_from_optimizer_slots(slots)

  def _reset_ids(self, ids):
    self.ids = ids
    self.prefetch_values(update=True)
    for s in self._tracked_slots:
      s._reset_ids(ids)
