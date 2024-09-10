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
"""patch on optimizers"""

import functools
import six

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.python.distribute import central_storage_strategy
try:  # tf version >= 2.14.0
  from tensorflow.python.distribute import distribute_lib as distribute_ctx
  assert hasattr(distribute_ctx, 'has_strategy')
except:
  from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
try:  # tf version >= 2.14.0
  from tensorflow.python.framework.tensor import Tensor
except:
  from tensorflow.python.framework.ops import Tensor
try:  # tf version >= 2.13.0
  from tensorflow.python.framework.indexed_slices import IndexedSlices
except:
  from tensorflow.python.framework.ops import IndexedSlices
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
try:  # tf version >= 2.14.0
  from tensorflow.python.ops.cond import cond
except:
  from tensorflow.python.ops.control_flow_ops import cond
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import slot_creator
try:
  from tensorflow.python.distribute.sharded_variable import ShardedVariable
except:
  ShardedVariable = type('Dummy', (object,), {})
try:  # tf version >= 2.10.0
  from tensorflow.python.trackable import base as trackable
except:
  from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.keras.optimizer_v2 import optimizer_v2 as optimizer_v2_legacy
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_v2_legacy_utils

try:  # tf version >= 2.16
  from tf_keras.optimizers.legacy import Optimizer as keras_OptimizerV2_legacy
  from tf_keras.optimizers import Optimizer as keras_OptimizerV2
except:
  try:  # Keras version >= 2.12.0
    from tensorflow.keras.optimizers.legacy import Optimizer as keras_OptimizerV2_legacy
    from tensorflow.keras.optimizers import Optimizer as keras_OptimizerV2
  except:
    from tensorflow.keras.optimizers import Optimizer as keras_OptimizerV2_legacy
    keras_OptimizerV2 = keras_OptimizerV2_legacy

from tensorflow.python.eager import tape
from tensorflow.python.distribute import values_util as distribute_values_util
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.ops.variables import VariableAggregation
try:  # The data_structures has been moved to the new package in tf 2.11
  from tensorflow.python.trackable import data_structures
except:
  from tensorflow.python.training.tracking import data_structures
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import \
  TrainableWrapperDistributedPolicy
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import trainable_wrapper_filter

try:
  import horovod.tensorflow as hvd
except Exception as e:
  logging.warning("An exception occurred when import horovod.tensorflow: " +
                  str(e))
  hvd = None


def DynamicEmbeddingOptimizer(self, bp_v2=False, synchronous=False, **kwargs):
  """ An optimizer wrapper to make any TensorFlow optimizer capable of training
  Dynamic Embeddding Variables.

  Args:
    self: a TensorFlow optimizer.
    bp_v2: If True, updating parameters will use updating instead of setting, which solves
      the race condition problem among workers during back-propagation in large-scale
      distributed asynchronous training. Reference: https://www.usenix.org/system/files/osdi20-jiang.pdf
    synchronous: If True, we will use DE custom all-reduce method(now implemented by horovod) to merge the dense grad
      of model parameter, the default reduce method is SUM. If False, we should use
      For TrainableWrapper's grad, keep same with before.

  Example usage:

    ```python
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(
        tf.train.AdamOptimizer(0.001))
    ```

  Returns:
    The optimizer itself but has ability to train Dynamic Embedding Variables.
  """
  self._bp_v2 = bp_v2
  self._custom_sync = synchronous

  original_apply_gradients = self.apply_gradients
  if hasattr(self, 'add_variable_from_reference'):
    original_add_variable_from_reference = self.add_variable_from_reference

  # pylint: disable=protected-access
  def _distributed_apply(distribution, grads_and_vars, name, apply_state):
    """`apply_gradients` using a `DistributionStrategy`."""

    def apply_grad_to_update_var(var, grad):
      """Apply gradient to variable."""
      if isinstance(var, Tensor):
        raise NotImplementedError("Trying to update a Tensor ", var)

      apply_kwargs = {}
      if not isinstance(var, de.TrainableWrapper):
        if isinstance(grad, IndexedSlices):
          if var.constraint is not None:
            raise RuntimeError(
                "Cannot use a constraint function on a sparse variable.")
          if "apply_state" in self._sparse_apply_args:
            apply_kwargs["apply_state"] = apply_state
          return self._resource_apply_sparse_duplicate_indices(
              grad.values, var, grad.indices, **apply_kwargs)

        if "apply_state" in self._dense_apply_args:
          apply_kwargs["apply_state"] = apply_state
        update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
        if var.constraint is not None:
          with ops.control_dependencies([update_op]):
            return var.assign(var.constraint(var))
        else:
          return update_op
      else:
        if not var.params.trainable:
          return control_flow_ops.no_op()

        with ops.colocate_with(None, ignore_existing=True):
          _slots = [self.get_slot(var, _s) for _s in self.get_slot_names()]
          var._track_optimizer_slots(_slots)

          with ops.control_dependencies([grad]):
            if isinstance(var, de.shadow_ops.ShadowVariable):
              v0 = var.read_value(do_prefetch=False)
            else:
              v0 = var.read_value(do_prefetch=var.params.bp_v2)
            s0 = [_s.read_value() for _s in _slots]
            _before = [v0] + s0

          if isinstance(grad, IndexedSlices):
            if var.constraint is not None:
              raise RuntimeError(
                  "Cannot use a constraint function on a sparse variable.")
            if "apply_state" in self._sparse_apply_args:
              apply_kwargs["apply_state"] = apply_state
            with ops.control_dependencies(_before):
              _apply_op = self._resource_apply_sparse_duplicate_indices(
                  grad.values, var, grad.indices, **apply_kwargs)
            with ops.control_dependencies([_apply_op]):
              _after = control_flow_ops.group(
                  [var.update_op(v0=v0)] +
                  [_s.update_op(v0=s0[si]) for si, _s in enumerate(_slots)])
              return _after

          if "apply_state" in self._dense_apply_args:
            apply_kwargs["apply_state"] = apply_state
          with ops.control_dependencies(_before):
            update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
          if var.constraint is not None:
            with ops.control_dependencies([update_op]):
              return var.assign(var.constraint(var))
          else:
            with ops.control_dependencies([update_op]):
              _after = control_flow_ops.group(
                  [var.update_op(v0=v0)] +
                  [_s.update_op(v0=s0[si]) for si, _s in enumerate(_slots)])
            return _after

    update_ops = []
    with optimizer_v2_legacy.name_scope_only_in_function_or_graph(
        name or self._name):
      for grad, var in grads_and_vars:
        # Colocate the update with variables to avoid unnecessary communication
        # delays. See b/136304694.
        with distribution.extended.colocate_vars_with(var):
          with optimizer_v2_legacy.name_scope_only_in_function_or_graph(
              "update" if ops.executing_eagerly_outside_functions(
              ) else "update_" + var.op.name):
            update_op = distribution.extended.update(var,
                                                     apply_grad_to_update_var,
                                                     args=(grad,),
                                                     group=False)
            replica_context = distribute_ctx.get_replica_context()
            if (replica_context is None or replica_context
                is distribute_ctx._get_default_replica_context()):
              # In cross-replica context, extended.update returns a list of
              # update ops from all replicas (group=False).
              update_ops.extend(update_op)
            else:
              # In replica context, extended.update return the single update op
              # of current replica.
              update_ops.append(update_op)

      any_symbolic = any(
          isinstance(i, ops.Operation) or tf_utils.is_symbolic_tensor(i)
          for i in update_ops)
      if not context.executing_eagerly() or any_symbolic:
        # If the current context is graph mode or any of the update ops are
        # symbolic then the step update should be carried out under a graph
        # context. (eager updates execute immediately)
        with backend._current_graph(update_ops).as_default():  # pylint: disable=protected-access
          with ops.control_dependencies(update_ops):
            return self._iterations.assign_add(1, read_value=False)

      return self._iterations.assign_add(1)

  def add_slot_v2_lagacy(var, slot_name, initializer="zeros", shape=None):
    """Add a new slot variable for `var`."""
    if slot_name not in self._slot_names:
      self._slot_names.append(slot_name)
    var_key = optimizer_v2_legacy._var_key(var)
    slot_dict = self._slots.setdefault(var_key, {})
    weight = slot_dict.get(slot_name, None)
    if weight is None:
      if isinstance(initializer, six.string_types) or callable(initializer):
        initializer = initializers.get(initializer)
        if isinstance(
            initializer,
            trackable.CheckpointInitialValueCallable) or (shape is not None):
          slot_shape = shape
        else:
          slot_shape = var.shape
        initial_value = functools.partial(initializer,
                                          shape=slot_shape,
                                          dtype=var.dtype)
      else:
        initial_value = initializer
      with self._distribution_strategy_scope():
        strategy = distribute_ctx.get_strategy()
        if not strategy.extended.variable_created_in_scope(var):
          raise ValueError(
              "Trying to create optimizer slot variable under the scope for "
              "tf.distribute.Strategy ({}), which is different from the scope "
              "used for the original variable ({}). Make sure the slot "
              "variables are created under the same strategy scope. This may "
              "happen if you're restoring from a checkpoint outside the scope".
              format(strategy, var))

        with strategy.extended.colocate_vars_with(var):
          if distribute_utils.is_distributed_variable(var):
            var_check = var.values[0]
          else:
            var_check = var

          if isinstance(var_check, de.TrainableWrapper):
            weight = de.create_slots(var, initial_value, slot_name,
                                     var._shared_name, self._bp_v2)
            # Record the optimizer Variable into trace.
            for _de_opt_var in var_check._optimizer_vars.value:
              self._track_trackable(_de_opt_var, _de_opt_var.name)
          else:
            weight = variables.Variable(
                name="%s/%s" % (
                    var._shared_name,
                    slot_name,
                ),  # pylint: disable=protected-access
                dtype=var.dtype,
                trainable=False,
                initial_value=initial_value,
            )
        backend.track_variable(weight)
        slot_dict[slot_name] = weight
        self._restore_slot_variable(slot_name=slot_name,
                                    variable=var,
                                    slot_variable=weight)
        self._weights.append(weight)
    return weight

  def _distributed_apply_gradients_fn(distribution, grads_and_vars, **kwargs):
    """`apply_gradients` using a `DistributionStrategy`."""

    def apply_grad_to_update_var(var, grad):

      def _update_step_fn(var, grad):
        if self.jit_compile:
          return self._update_step_xla(grad, var, id(self._var_key(var)))
        else:
          return self._update_step(grad, var)

      if not isinstance(var, de.TrainableWrapper):
        return _update_step_fn(var, grad)
      else:
        if not var.params.trainable:
          return control_flow_ops.no_op()

        with ops.colocate_with(None, ignore_existing=True):
          _slots = [
              _s for _s in self._variables
              if isinstance(_s, de.TrainableWrapper)
          ]
          var._track_optimizer_slots(_slots)

          with ops.control_dependencies([grad]):
            if isinstance(var, de.shadow_ops.ShadowVariable):
              v0 = var.read_value(do_prefetch=False)
            else:
              v0 = var.read_value(do_prefetch=var.params.bp_v2)
            s0 = [_s.read_value() for _s in _slots]
            _before = [v0] + s0

          with ops.control_dependencies(_before):
            _update_step_fn(var, grad)

          with ops.control_dependencies([var]):
            _after = control_flow_ops.group(
                [var.update_op(v0=v0)] +
                [_s.update_op(v0=s0[si]) for si, _s in enumerate(_slots)])
            return _after

    for grad, var in grads_and_vars:
      distribution.extended.update(var,
                                   apply_grad_to_update_var,
                                   args=(grad,),
                                   group=False)

    if self.use_ema:
      _, var_list = zip(*grads_and_vars)
      self._update_model_variables_moving_average(var_list)
      if self.ema_overwrite_frequency:
        # Only when self.ema_overwrite_frequency is not None, we
        # overwrite the model variables.
        should_overwrite_model_vars = (self.iterations +
                                       1) % self.ema_overwrite_frequency == 0
        cond(
            math_ops.cast(should_overwrite_model_vars, dtypes.bool),
            true_fn=lambda: self.
            _overwrite_model_variables_with_average_value(  # noqa: E501
                var_list),
            false_fn=lambda: None,
        )
    return self.iterations.assign_add(1)

  def add_variable_from_reference(model_variable,
                                  variable_name,
                                  shape=None,
                                  initial_value=None):
    """Create an optimizer variable from model variable.

        Create an optimizer variable based on the information of model variable.
        For example, in SGD optimizer momemtum, for each model variable, a
        corresponding momemtum variable is created of the same shape and dtype.

        Args:
          model_variable: tf.Variable. The corresponding model variable to the
            optimizer variable to be created.
          variable_name: String. The name prefix of the optimizer variable to be
            created. The create variables name will follow the pattern
            `{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
          shape: List or Tuple, defaults to None. The shape of the optimizer
            variable to be created. If None, the created variable will have the
            same shape as `model_variable`.
          initial_value: A Tensor, or Python object convertible to a Tensor,
            defaults to None. The initial value of the optimizer variable, if
            None, the initial value will be default to 0.

        Returns:
          An optimizer variable.
        """
    if distribute_utils.is_distributed_variable(model_variable):
      var_check = model_variable.values[0]
    else:
      var_check = model_variable
    if isinstance(var_check, de.TrainableWrapper):
      variable = de.create_slots(model_variable, initial_value, variable_name,
                                 model_variable._shared_name, self._bp_v2)
      self._variables.append(variable)
      # Record the optimizer Variable into trace.
      for _de_opt_var in var_check._optimizer_vars.value:
        self._track_trackable(_de_opt_var, _de_opt_var.name)
    else:
      variable = original_add_variable_from_reference(model_variable,
                                                      variable_name, shape,
                                                      initial_value)
    return variable

  def get_slot_v1(var, name):
    raise NotImplementedError(
        "TFRA doesn't overwrite get_slot function for tf.compat.v1.train.Optimizer now"
    )

  def get_slot_v2_lagacy(var, slot_name):
    var_key = optimizer_v2_legacy._var_key(var)
    slot_dict = self._slots[var_key]
    slot_variable = slot_dict[slot_name]
    if isinstance(slot_variable, ShardedVariable):
      # Construct a ShardedVariable that points to the input
      # ShardedVariable's component shard's slot variables.
      shard_vars = []
      for shard in slot_variable.variables:
        slot_shard = self.get_slot(shard, slot_name)
        shard_vars.append(slot_shard)
      slot_variable = ShardedVariable(shard_vars, name=slot_variable.name)
    elif isinstance(var, de.TrainableWrapper) and hasattr(
        var, "_distributed_container"):
      replica_id = distribute_values_util.get_current_replica_id_as_int()
      if replica_id is None:
        for index, replica_var in enumerate(
            var._distributed_container().values):
          if replica_var is var:
            replica_id = index
            break
      slot_variable = slot_dict[slot_name]._get_replica(replica_id)
    return slot_variable

  def _get_or_make_slot(var, val, slot_name, op_name):
    """Find or create a slot for a variable.

        Args:
          var: A `Variable` object.
          val: A `Tensor`.  The initial value of the slot.
          slot_name: Name for the slot.
          op_name: Name to use when scoping the Variable that
            needs to be created for the slot.

        Returns:
          A `Variable` object.
        """
    named_slots = self._slot_dict(slot_name)
    if optimizer._var_key(var) not in named_slots:
      if isinstance(var, de.TrainableWrapper):
        new_slot_variable = de.create_slots(var, val, slot_name, op_name,
                                            self._bp_v2)
      else:
        new_slot_variable = slot_creator.create_slot(var, val, op_name)
      self._restore_slot_variable(slot_name=slot_name,
                                  variable=var,
                                  slot_variable=new_slot_variable)
      named_slots[optimizer._var_key(var)] = new_slot_variable
    return named_slots[optimizer._var_key(var)]

  def _get_or_make_slot_with_initializer(var, initializer, shape, dtype,
                                         slot_name, op_name):
    """Find or create a slot for a variable, using an Initializer.

        Args:
          var: A `Variable` object.
          initializer: An `Initializer`.  The initial value of the slot.
          shape: Shape of the initial value of the slot.
          dtype: Type of the value of the slot.
          slot_name: Name for the slot.
          op_name: Name to use when scoping the Variable that
            needs to be created for the slot.

        Returns:
          A `Variable` object.
        """
    named_slots = self._slot_dict(slot_name)
    if optimizer._var_key(var) not in named_slots:
      if isinstance(var, de.TrainableWrapper):
        new_slot_variable = de.create_slots(var, initializer, slot_name,
                                            op_name, self._bp_v2)
      else:
        new_slot_variable = slot_creator.create_slot_with_initializer(
            var, initializer, shape, dtype, op_name)
      self._restore_slot_variable(slot_name=slot_name,
                                  variable=var,
                                  slot_variable=new_slot_variable)
      named_slots[optimizer._var_key(var)] = new_slot_variable
    return named_slots[optimizer._var_key(var)]

  def _zeros_slot(var, slot_name, op_name):
    """Find or create a slot initialized with 0.0.

        Args:
          var: A `Variable` object.
          slot_name: Name for the slot.
          op_name: Name to use when scoping the Variable that
            needs to be created for the slot.

        Returns:
          A `Variable` object.
        """
    named_slots = self._slot_dict(slot_name)
    if optimizer._var_key(var) not in named_slots:
      if isinstance(var, de.TrainableWrapper):
        new_slot_variable = de.create_slots(var, 0.0, slot_name, op_name,
                                            self._bp_v2)
      else:
        new_slot_variable = slot_creator.create_zeros_slot(var, op_name)
      self._restore_slot_variable(slot_name=slot_name,
                                  variable=var,
                                  slot_variable=new_slot_variable)
      named_slots[optimizer._var_key(var)] = new_slot_variable
    return named_slots[optimizer._var_key(var)]

  def _hvd_aggregate_gradients(hvd_handle,
                               grads_and_vars_in,
                               sparse_as_dense=True):
    if hvd_handle.size() <= 1:
      return grads_and_vars_in
    var_list = []
    aggregated_grad = []
    for grad, var in grads_and_vars_in:
      var_list.append(var)
      with ops.device(var.device):
        # Dense gradients.
        if grad is None:
          aggregated_grad.append(None)  # pass-through.
          continue
        else:
          '''
          Treat all sparse gradients as dense tensors.  This can help improve
          performance and memory utilization if the original sparse gradient
          has high density.
          '''
          if sparse_as_dense:
            grad = ops.convert_to_tensor(grad) if isinstance(
                grad, IndexedSlices) else grad
          aggregated_grad.append(hvd_handle.allreduce(grad, op=hvd_handle.Sum))
    return zip(aggregated_grad, var_list)

  def apply_gradients_sync_v1(grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.
    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: apply gradients op to be executed by each replica.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if hvd is None:
      raise ValueError(
          "Please install Horovod properly first if you want to use distributed synchronous training based on Horovod"
      )
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    trainable_wrapper_grad_and_vars = []

    with backend.name_scope(name or self._name):
      test_unaggregated_lambda = lambda x: isinstance(x[1], de.TrainableWrapper)
      dense_grads_and_vars_aggregated, trainable_wrapper_grad_and_vars = trainable_wrapper_filter(
          grads_and_vars, test_unaggregated_lambda)
      aggregated_grads_and_vars = _hvd_aggregate_gradients(
          hvd, dense_grads_and_vars_aggregated)
      if dense_grads_and_vars_aggregated:
        update_op = original_apply_gradients(aggregated_grads_and_vars,
                                             global_step)
      else:
        update_op = control_flow_ops.no_op()
      if trainable_wrapper_grad_and_vars:
        trainable_update_op = original_apply_gradients(
            trainable_wrapper_grad_and_vars, global_step)
        train_op = control_flow_ops.group([update_op, trainable_update_op])
      else:
        train_op = update_op
      return train_op

  def apply_gradients_sync_v2_lagacy(grads_and_vars,
                                     name=None,
                                     experimental_aggregate_gradients=True):
    if hvd is None:
      raise ValueError(
          "Please install Horovod properly first if you want to use distributed synchronous training based on Horovod"
      )
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    grads_and_vars = optimizer_v2_legacy_utils.filter_empty_gradients(
        grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]

    with ops.name_scope_v2(self._name):
      # Create iteration if necessary.
      with ops.init_scope():
        self._create_all_weights(var_list)

      if not grads_and_vars:
        # Distribution strategy does not support reducing an empty list of
        # gradients
        return control_flow_ops.no_op()

      if distribute_ctx.in_cross_replica_context():
        raise RuntimeError(
            "`apply_gradients() cannot be called in cross-replica context. "
            "Use `tf.distribute.Strategy.run` to enter replica "
            "context.")

      strategy = distribute_ctx.get_strategy()

      apply_state = self._prepare(var_list)
      if experimental_aggregate_gradients:
        grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
        test_unaggregated_lambda = lambda x: isinstance(x[1], de.
                                                        TrainableWrapper)
        dense_grads_and_vars_aggregated, sparse_grads_and_vars_unaggregated = trainable_wrapper_filter(
            grads_and_vars, test_unaggregated_lambda)
        dense_grads_and_vars_aggregated = _hvd_aggregate_gradients(
            hvd, dense_grads_and_vars_aggregated)
        grads_and_vars = []
        for g_and_v in dense_grads_and_vars_aggregated:
          grads_and_vars.append(g_and_v)
        for g_and_v in sparse_grads_and_vars_unaggregated:
          grads_and_vars.append(g_and_v)
      grads_and_vars = self._transform_gradients(grads_and_vars)

      if optimizer_v2_legacy_utils.strategy_supports_no_merge_call():
        return self._distributed_apply(strategy, grads_and_vars, name,
                                       apply_state)
      else:
        return distribute_ctx.get_replica_context().merge_call(
            functools.partial(self._distributed_apply, apply_state=apply_state),
            args=(grads_and_vars,),
            kwargs={
                "name": name,
            })

  def apply_gradients_sync_v2(
      grads_and_vars,
      name=None,
      skip_gradients_aggregation=False,
      **kwargs,
  ):
    if hvd is None:
      raise ValueError(
          "Please install Horovod properly first if you want to use distributed synchronous training based on Horovod"
      )
    if hasattr(self, '_mesh') and hasattr(self, '_run_with_dtensor'):
      if self._mesh or self._run_with_dtensor:
        # Skip any usage of strategy logic for DTensor
        return super(keras_OptimizerV2, self).apply_gradients(grads_and_vars,
                                                              name=name)

    # `experimental_aggregate_gradients` is an arg in `apply_gradients` of
    # v2 optimizer -- the reverse of `skip_gradients_aggregation`.
    # We read it from kwargs for backward compatibility.
    experimental_aggregate_gradients = kwargs.pop(
        "experimental_aggregate_gradients", True)
    if not skip_gradients_aggregation and experimental_aggregate_gradients:
      test_unaggregated_lambda = lambda x: isinstance(x[1], de.TrainableWrapper)
      dense_grads_and_vars_aggregated, sparse_grads_and_vars_unaggregated = trainable_wrapper_filter(
          grads_and_vars, test_unaggregated_lambda)
      dense_grads_and_vars_aggregated = _hvd_aggregate_gradients(
          hvd, dense_grads_and_vars_aggregated)
      grads_and_vars = []
      for g_and_v in dense_grads_and_vars_aggregated:
        grads_and_vars.append(g_and_v)
      for g_and_v in sparse_grads_and_vars_unaggregated:
        grads_and_vars.append(g_and_v)
    return super(keras_OptimizerV2, self).apply_gradients(grads_and_vars,
                                                          name=name)

  def _dist_tw_grads_and_vars_filter(grads_and_vars_in):
    dense_grads_and_vars_aggregated_out = []
    sparse_grads_and_vars_unaggregated_out = []
    test_unaggregated_lambda = lambda x: isinstance(
        x[1], de.DistributedVariableWrapper)
    for g_and_v in grads_and_vars_in:
      if test_unaggregated_lambda(g_and_v):
        sparse_grads_and_vars_unaggregated_out.append(g_and_v)
      else:
        dense_grads_and_vars_aggregated_out.append(g_and_v)
    return tuple(dense_grads_and_vars_aggregated_out), tuple(
        sparse_grads_and_vars_unaggregated_out)

  def apply_gradients_strategy_v2_lagacy(grads_and_vars,
                                         name=None,
                                         experimental_aggregate_gradients=True):
    grads_and_vars = optimizer_v2_legacy_utils.filter_empty_gradients(
        grads_and_vars)
    var_list = [v for (_, v) in grads_and_vars]

    with ops.name_scope_v2(self._name):
      # Create iteration if necessary.
      with ops.init_scope():
        self._create_all_weights(var_list)

      if not grads_and_vars:
        # Distribution strategy does not support reducing an empty list of
        # gradients
        return control_flow_ops.no_op()

      if distribute_ctx.in_cross_replica_context():
        raise RuntimeError(
            "`apply_gradients() cannot be called in cross-replica context. "
            "Use `tf.distribute.Strategy.run` to enter replica "
            "context.")

      strategy = distribute_ctx.get_strategy()
      if (not experimental_aggregate_gradients and strategy and isinstance(
          strategy, (parameter_server_strategy.ParameterServerStrategyV1,
                     parameter_server_strategy_v2.ParameterServerStrategyV2,
                     central_storage_strategy.CentralStorageStrategy,
                     central_storage_strategy.CentralStorageStrategyV1))):
        raise NotImplementedError(
            "`experimental_aggregate_gradients=False is not supported for "
            "ParameterServerStrategy and CentralStorageStrategy")

      apply_state = self._prepare(var_list)
      if experimental_aggregate_gradients:
        grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
        dense_grads_and_vars_aggregated, sparse_grads_and_vars_unaggregated = \
          _dist_tw_grads_and_vars_filter(grads_and_vars)
        dense_grads_and_vars_aggregated = self._aggregate_gradients(
            dense_grads_and_vars_aggregated)
        grads_and_vars = dense_grads_and_vars_aggregated
        for g_and_v in sparse_grads_and_vars_unaggregated:
          grads_and_vars.append(g_and_v)

      grads_and_vars = self._transform_gradients(grads_and_vars)

      if optimizer_v2_legacy_utils.strategy_supports_no_merge_call():
        return self._distributed_apply(strategy, grads_and_vars, name,
                                       apply_state)
      else:
        return distribute_ctx.get_replica_context().merge_call(
            functools.partial(self._distributed_apply, apply_state=apply_state),
            args=(grads_and_vars,),
            kwargs={
                "name": name,
            })

  def apply_gradients_strategy_v2(
      grads_and_vars,
      name=None,
      skip_gradients_aggregation=False,
      **kwargs,
  ):
    if hasattr(self, '_mesh') and hasattr(self, '_run_with_dtensor'):
      if self._mesh or self._run_with_dtensor:
        # Skip any usage of strategy logic for DTensor
        return super(keras_OptimizerV2, self).apply_gradients(grads_and_vars,
                                                              name=name)

    # `experimental_aggregate_gradients` is an arg in `apply_gradients` of
    # v2 optimizer -- the reverse of `skip_gradients_aggregation`.
    # We read it from kwargs for backward compatibility.
    experimental_aggregate_gradients = kwargs.pop(
        "experimental_aggregate_gradients", True)
    if not skip_gradients_aggregation and experimental_aggregate_gradients:
      dense_grads_and_vars_aggregated, sparse_grads_and_vars_unaggregated = \
        _dist_tw_grads_and_vars_filter(grads_and_vars)
      dense_grads_and_vars_aggregated = self.aggregate_gradients(
          dense_grads_and_vars_aggregated)
      grads_and_vars = dense_grads_and_vars_aggregated
      for g_and_v in sparse_grads_and_vars_unaggregated:
        grads_and_vars.append(g_and_v)

    return super(keras_OptimizerV2, self).apply_gradients(grads_and_vars,
                                                          name=name)

  def compute_gradients_horovod_wrapper(compute_gradients_horovod_func):

    def compute_gradients_horovod_wrapper_impl(*args, **kwargs):
      if self._finish_register_tw is False:
        var_list = kwargs.get('var_list', None)
        if var_list is None:
          raise Exception(
              "var_list parameter must not be None when using compute_gradients function!"
          )
        _, trainable_wrapper_vars = trainable_wrapper_filter(var_list)
        for tw in trainable_wrapper_vars:
          self.register_local_var(tw)
        self._finish_register_tw = True
      return compute_gradients_horovod_func(*args, **kwargs)

    return compute_gradients_horovod_wrapper_impl

  if isinstance(self, optimizer.Optimizer):
    # TF1 Optimizer
    self._get_or_make_slot = _get_or_make_slot
    self._get_or_make_slot_with_initializer = _get_or_make_slot_with_initializer
    self._zeros_slot = _zeros_slot
    # TODO(MoFHeka): Support distribute strategy with tf.compat.v1.train.Optimizer
    # self.get_slot = get_slot_v1
    if self._custom_sync:
      self.apply_gradients = apply_gradients_sync_v1
    elif 'horovod' in str(self):
      # If use Horovod keras optimizer wrapper
      if not hasattr(self, 'register_local_var'):
        import horovod
        raise Exception(
            f"Input optimizer must has register_local_var function! Your Horovod version is {horovod.__version__}, please upgrade to greater than 0.26.0"
        )
      self._finish_register_tw = False
      compute_gradients_horovod_v1 = self.compute_gradients
      self.compute_gradients = compute_gradients_horovod_wrapper(
          compute_gradients_horovod_v1)
    if distribute_ctx.has_strategy():
      raise Exception(
          "Distribute strategy isn't supported when using tf.compat.v1.train.Optimizer in TFRA for now."
      )
  elif (isinstance(self, optimizer_v2_legacy.OptimizerV2)
        or isinstance(self, keras_OptimizerV2_legacy)
        or isinstance(self, keras_OptimizerV2)):
    # TF2 Keras Optimizer
    if hasattr(self, '_distributed_apply'):
      # Compatible with Keras < 2.12.0 legacy optimizer
      self.add_slot = add_slot_v2_lagacy
      self.get_slot = get_slot_v2_lagacy
      self._distributed_apply = _distributed_apply
      if self._custom_sync:
        self.apply_gradients = apply_gradients_sync_v2_lagacy
      elif 'horovod._keras' in str(self):
        # If use Horovod keras optimizer wrapper
        if not hasattr(self, 'register_local_var'):
          import horovod
          raise Exception(
              f"Input optimizer must has register_local_var function! Your Horovod version is {horovod.__version__}, please upgrade to greater than 0.26.0"
          )
        self._finish_register_tw = False
        compute_gradients_horovod_v2 = self._compute_gradients
        self._compute_gradients = compute_gradients_horovod_wrapper(
            compute_gradients_horovod_v2)
      else:
        self.apply_gradients = apply_gradients_strategy_v2_lagacy
    elif hasattr(self, '_distributed_apply_gradients_fn'):
      # Latest Keras optimizer
      self.add_variable_from_reference = add_variable_from_reference
      self._distributed_apply_gradients_fn = _distributed_apply_gradients_fn
      if self._custom_sync:
        self.apply_gradients = apply_gradients_sync_v2
      else:
        self.apply_gradients = apply_gradients_strategy_v2
    else:
      raise Exception(f"Optimizer type is not supported! got {str(type(self))}")
  else:
    raise Exception(f"Optimizer type is not supported! got {str(type(self))}")
  return self


def create_slots(variable, init, slot_name, op_name, bp_v2):
  """Helper function for creating a slot variable for statefull optimizers."""
  if distribute_utils.is_distributed_variable(variable):
    strategy_devices = variable.distribute_strategy.extended.worker_devices
    primary = variable._get_on_device_or_primary()
    params_var_ = primary.params
  else:
    strategy_devices = None
    primary = variable
    params_var_ = primary.params

  scope_store = variable_scope._get_default_variable_store()
  if params_var_.short_file_name:
    full_name = params_var_.name + "/" + slot_name
  else:
    full_name = params_var_.name + "/" + op_name + "/" + slot_name

  if full_name not in scope_store._vars:
    with ops.colocate_with(primary, ignore_existing=True):
      slot_variable_ = de.Variable(
          name=full_name,
          key_dtype=params_var_.key_dtype,
          value_dtype=params_var_.value_dtype,
          dim=params_var_.dim,
          devices=params_var_.devices,
          partitioner=params_var_.partition_fn,
          initializer=init,
          init_size=params_var_.init_size,
          kv_creator=params_var_.kv_creator,
          trainable=False,
          checkpoint=params_var_.checkpoint,
          bp_v2=bp_v2 if bp_v2 is not None else params_var_.bp_v2)

    scope_store._vars[full_name] = slot_variable_
    primary._optimizer_vars.value.append(slot_variable_)

  slot_trainable = None
  if context.executing_eagerly():
    slot_tw_name = slot_name + '-' + str(optimizer_v2_legacy._var_key(variable))
  else:
    # In graph mode of former version, It only uses slot_name as name to
    # trainable wrappers of slots. So here set it the name to slot_name
    # for forward compatibility.
    slot_tw_name = slot_name

  def slot_trainable_create_(var_impl, scope_store_params, full_name_in,
                             slot_tw_name_in):
    if isinstance(var_impl, de.shadow_ops.ShadowVariable):
      slot_trainable = de.shadow_ops.ShadowVariable(
          params=scope_store_params,
          ids=var_impl.ids,
          exists=var_impl.exists,
          name=full_name_in,
          trainable=False,
      )
    else:
      _, slot_trainable = de.embedding_lookup(
          params=scope_store_params,
          ids=var_impl.ids,
          name=slot_tw_name_in,
          return_trainable=True,
      )
    return slot_trainable

  if strategy_devices:
    slot_variable_impl = tf_utils.ListWrapper([])
    for i, strategy_device in enumerate(strategy_devices):
      with ops.device(strategy_device):
        slot_tw_name_replica = slot_tw_name
        full_name_replica = full_name
        if i > 0:
          slot_tw_name_replica = "%s/replica_%d" % (slot_tw_name, i)
          full_name_replica = "%s/replica_%d" % (full_name, i)
        with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
          with tape.stop_recording():
            slot_variable_impl.as_list().append(
                slot_trainable_create_(variable.values[i],
                                       scope_store._vars[full_name],
                                       full_name_replica, slot_tw_name_replica))
    slot_trainable = de.DistributedVariableWrapper(
        variable.distribute_strategy, slot_variable_impl.as_list(),
        VariableAggregation.NONE,
        TrainableWrapperDistributedPolicy(VariableAggregation.NONE))
  else:
    slot_trainable = slot_trainable_create_(variable,
                                            scope_store._vars[full_name],
                                            full_name, slot_tw_name)

  return slot_trainable
