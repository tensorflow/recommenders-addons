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
from tensorflow_recommenders_addons.utils.resource_loader import get_tf_version_triple

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

_ANONYMOUS_TRAINABLE_STORE_KEY = '_anonymous_trainable_store_key'


class TrainableWrapper(resource_variable_ops.ResourceVariable):
  """
    This class is a trainable wrapper of Dynamic Embedding,
    and the key role is recording the map relation between params and ids.
    inheriting from the ResourceVariable make it trainable.
    """

  def __getattribute__(self, name):
    if name in ["sparse_read", "gather_nd"]:
      raise AttributeError("no such method: {}".format(name))
    return super(resource_variable_ops.ResourceVariable,
                 self).__getattribute__(name)

  def __init__(self, params, ids, max_norm, *args, **kwargs):
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
    self.params = params
    self.ids = ids
    self.exists = None
    self.max_norm = max_norm
    self.prefetch_values_op = None
    self.model_mode = kwargs.get("model_mode")
    kwargs.pop("model_mode")
    self._tracked_slots = []
    self._optimizer_vars = data_structures.ListWrapper([])
    super(TrainableWrapper, self).__init__(*args, **kwargs)

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
  ):
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

    if (isinstance(initial_value, ops.Tensor)
        and hasattr(initial_value, "graph")
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
      super(resource_variable_ops.ResourceVariable, self).__init__(
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

  def _read_variable_op(self, do_prefetch=True):
    resource_variable_ops.variable_accessed(self)
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
      tape.record_operation("ReadVariableOp", [_result], [self._handle],
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
    if (isinstance(initial_value, ops.Tensor)
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
          trainable_name = ops.get_default_graph().unique_name(
              _ANONYMOUS_TRAINABLE_STORE_KEY)
        if not context.executing_eagerly() and not ops.inside_function():
          wrapper = de.TrainableWrapper(params,
                                        ids,
                                        max_norm=max_norm,
                                        initial_value=initial_value,
                                        dtype=params.value_dtype,
                                        trainable=params.trainable,
                                        collections=collections,
                                        model_mode=ModelMode.CURRENT_SETTING,
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
                  model_mode=ModelMode.CURRENT_SETTING)
              params._trainable_store[trainable_name] = shadow
          return shadow

      with ops.colocate_with(ids, ignore_existing=True):
        trainable_ = _create_or_get_trainable(name)

      if isinstance(trainable_, de.shadow_ops.ShadowVariable):
        embeddings = de.shadow_ops.embedding_lookup(
            trainable_,
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

    embeddings = array_ops.identity(trainable_)
    embeddings = array_ops.reshape(embeddings, shape=embeddings_shape)

  return (embeddings, trainable_) if return_trainable else embeddings


def embedding_lookup_unique(params,
                            ids,
                            partition_strategy=None,
                            name=None,
                            validate_indices=None,
                            max_norm=None,
                            return_trainable=False):
  """Version of embedding_lookup that avoids duplicate lookups.
  This can save communication in the case of repeated ids.
  Same interface as embedding_lookup.

  Args:
    params: A dynamic_embedding.Variable instance.
    ids: a tensor with any shape as same dtype of params.key_dtype.
    partition_strategy: No used, for API compatiblity with `nn.emedding_lookup`.
    name: A name for the operation. Name is optional in graph mode and required
      in eager mode.
    validate_indices: No used, just for compatible with nn.embedding_lookup .
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    return_trainable: optional, If True, also return TrainableWrapper

  Returns:
    A tensor with shape [shape of ids] + [dim],
      dim is equal to the value dim of params.
      containing the values from the params tensor(s) for keys in ids.
    trainable_wrap:
      A TrainableWrapper object used to fill the Optimizers `var_list`
        Only provided if `return_trainable` is True.
  """
  with ops.name_scope(name, "EmbeddingLookupUnique", [params, ids]):
    ids = ops.convert_to_tensor(ids)
    shape = array_ops.shape(ids)
    ids_flat = array_ops.reshape(ids, math_ops.reduce_prod(shape,
                                                           keepdims=True))
    unique_ids, idx = array_ops.unique(ids_flat)
    unique_embeddings, trainable_ = embedding_lookup(
        params,
        unique_ids,
        partition_strategy=partition_strategy,
        name=name,
        validate_indices=None,
        max_norm=validate_indices,
        return_trainable=True)
    embeddings_flat = array_ops.gather(unique_embeddings, idx)
    embeddings_shape = array_ops.concat(
        [shape, array_ops.shape(unique_embeddings)[1:]], 0)
    embeddings = array_ops.reshape(embeddings_flat, embeddings_shape)
    embeddings.set_shape(ids.get_shape().concatenate(
        unique_embeddings.get_shape()[1:]))
    return (embeddings, trainable_) if return_trainable else embeddings


def embedding_lookup_sparse(
    params,
    sp_ids,
    sp_weights,
    partition_strategy=None,  # no used
    name="embedding_lookup_sparse",
    combiner="mean",
    max_norm=None,
    return_trainable=False,
):
  """Provides a dynamic version of embedding_lookup_sparse
      similar with tf.nn.embedding_lookup_sparse.

    This op assumes that there is at least one id for each row in the dense tensor
    represented by sp_ids (i.e. there are no rows with empty features), and that
    all the indices of sp_ids are in canonical row-major order.

    It also assumes that all id values lie in the range [0, p0), where p0
    is the sum of the size of params along dimension 0.

    Args:
      params: A single `dynamic_embedding.Variable` instance representing
        the complete embedding tensor.
      sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
        and M is arbitrary.
      sp_weights: either a `SparseTensor` of float / double weights, or `None` to
        indicate all weights should be taken to be 1. If specified, `sp_weights`
        must have exactly the same shape and indices as `sp_ids`.
      partition_strategy: No used.
      name: a name for the operation. Name is optional in graph mode and required
        in eager mode.
      combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
        and "sum" are supported. "sum" computes the weighted sum of the embedding
        results for each row. "mean" is the weighted sum divided by the total
        weight. "sqrtn" is the weighted sum divided by the square root of the sum
        of the squares of the weights.
      max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
        than this value, before combining.
      return_trainable: optional, If True, also return TrainableWrapper create by
        `dynamic_embedding.embedding_lookup`

    Returns:
      combined_embeddings: A dense tensor representing the combined embeddings
        for the sparse ids. For each row in the dense tensor represented by
        `sp_ids`, the op looks up the embeddings for all ids in that row,
        multiplies them by the corresponding weight, and combines these embeddings
        as specified.

        In other words, if

          `shape(combined params) = [+infinity, dim]`

        and

          `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`

        then

          `shape(output) = [d0, dim]`.

        For instance, if params dim=20, and sp_ids / sp_weights are

          ```python
          [0, 0]: id 1, weight 2.0
          [0, 1]: id 3, weight 0.5
          [1, 0]: id 0, weight 1.0
          [2, 3]: id 1, weight 3.0
          ```

        with `combiner`="mean", then the output will be a 3x20 matrix where

          ```python
          output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
          output[1, :] = (params[0, :] * 1.0) / 1.0
          output[2, :] = (params[1, :] * 3.0) / 3.0
          ```
      trainable_wrap:
        A TrainableWrapper object used to fill the Optimizers `var_list`
          Only provided if `return_trainable` is True.
    Raises:
      TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
        neither `None` nor `SparseTensor`.
      ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
    """
  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")

  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")

  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" + name if scope.name else name
  with ops.name_scope(full_name + "/"):
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    ids, idx = array_ops.unique(ids)

    embeddings, trainable_ = embedding_lookup(
        params,
        ids,
        name=name + '/embedding_lookup',
        partition_strategy=partition_strategy,
        max_norm=max_norm,
        return_trainable=True,
    )
    if embeddings.dtype in (dtypes.float16, dtypes.bfloat16):
      embeddings = math_ops.cast(embeddings, dtypes.float32)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      embeddings = array_ops.gather(embeddings, idx)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                             0)

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(
            orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids, name=name)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum, name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div(embeddings, weight_sum_sqrt, name=name)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = de.math.sparse_segment_sum(embeddings,
                                                idx,
                                                segment_ids,
                                                name=name)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(embeddings,
                                                  idx,
                                                  segment_ids,
                                                  name=name)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(embeddings,
                                                    idx,
                                                    segment_ids,
                                                    name=name)
      else:
        assert False, "Unrecognized combiner"

    return (embeddings, trainable_) if return_trainable else embeddings


def safe_embedding_lookup_sparse(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    name="safe_embedding_lookup_sparse",
    partition_strategy=None,  # no used
    max_norm=None,
    return_trainable=False,
):
  """Provides a dynamic version of `tf.nn.safe_embedding_lookup_sparse`.

    Lookup embedding results, accounting for empty features and invalid weights.

    Any IDs will be treated as valid include non-positive IDs.
    Invalid weights (<= 0) are pruned from input weights, as well as any IDs
    with non-positive weight. For an entry with no features, the embedding vector
    for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

    The ids and weights may be multi-dimensional. Embeddings are always aggregated
    along the last dimension.

    Args:
      embedding_weights: A single `dynamic_embedding.Variable` instance
        representing the complete embedding tensor.
      sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
        ids. `d_0` is typically batch size.
      sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
        float weights corresponding to `sparse_ids`, or `None` if all weights are
        be assumed to be 1.0.
      combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
        default.
      default_id: The id to use for an entry with no features.
      name: A name for this operation. Name is optional in graph mode and required
        in eager mode.
      partition_strategy: A string specifying the partitioning strategy. Currently
        `"div"` and `"mod"` are supported. Default is `"div"`.
      max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
        combining.

    Returns:
      combined_embeddings:
        A dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.
      trainable_wrap:
        A TrainableWrapper object used to fill the Optimizers `var_list`
          Only provided if `return_trainable` is True.

    Raises:
      ValueError: if `embedding_weights` is empty.
  """
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

  scope = variable_scope.get_variable_scope()
  full_name = scope.name + "/" + name if scope.name else name
  with ops.name_scope(full_name + "/"):
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = tensor_shape.dimension_value(
        sparse_ids.dense_shape.get_shape()[0])
    original_rank = (array_ops.size(original_shape)
                     if original_rank_dim is None else original_rank_dim)
    sparse_ids = de.math.sparse_reshape(
        sparse_ids,
        [
            math_ops.reduce_prod(
                array_ops.slice(original_shape, [0], [original_rank - 1])),
            array_ops.gather(original_shape, original_rank - 1),
        ],
    )
    if sparse_weights is not None:
      sparse_weights = sparse_tensor.SparseTensor(sparse_ids.indices,
                                                  sparse_weights.values,
                                                  sparse_ids.dense_shape)

    # Prune invalid weights.
    if combiner != "sum":
      sparse_ids, sparse_weights = _prune_invalid_weights(
          sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = de.math.sparse_fill_empty_rows(
        sparse_ids, default_id or 0)
    if sparse_weights is not None:
      sparse_weights, _ = de.math.sparse_fill_empty_rows(sparse_weights, 1.0)

    result, trainable_ = embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=name + "/embedding_lookup_sparse",
        max_norm=max_norm,
        return_trainable=True,
    )

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.stack([1, array_ops.shape(result)[1]]),
      )

      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name="where")

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat(
            [
                array_ops.slice(
                    math_ops.cast(original_shape, dtypes.int32),
                    [0],
                    [original_rank - 1],
                ),
                array_ops.slice(array_ops.shape(result), [1], [-1]),
            ],
            0,
        ),
    )
    final_result.set_shape(
        tensor_shape.unknown_shape(
            (tensor_shape.Dimension(original_rank_dim) - 1).value).concatenate(
                result.get_shape()[1:]))
    return (final_result, trainable_) if return_trainable else final_result


def _prune_invalid_weights(sparse_ids, sparse_weights):
  """Prune invalid weights (< 0) from the input ids and weights."""
  if sparse_weights is not None:
    is_weights_valid = math_ops.greater(sparse_weights.values, 0)
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_weights_valid)
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_weights_valid)
  return sparse_ids, sparse_weights


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


def get_model_mode():
  """ get model mode.

  Returns:
    A string: `train` or 'inference'
  """
  return ModelMode.CURRENT_SETTING


def enable_train_mode():
  """ enable train mode.
  """
  ModelMode.CURRENT_SETTING = ModelMode.TRAIN


def enable_inference_mode():
  """ set inference mode.
  """
  ModelMode.CURRENT_SETTING = ModelMode.INFERENCE
