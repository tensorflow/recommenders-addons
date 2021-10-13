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
"""patch on tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons import embedding_variable as ev

try:
  from tensorflow.python.keras.initializers import initializers_v2 as kinit2
except ImportError:
  kinit2 = None
  pass  # for compatible with TF < 2.3.x

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import initializers as kinit1
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops as rvo
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import device_setter
from tensorflow.python.training import optimizer
from tensorflow.python.training import slot_creator

_PARTITION_SHAPE = 'partition_shape'


class _DenseDynamicEmbeddingTrainableProcessor(optimizer._OptimizableVariable):
  """Processor for dense DynamicEmbedding."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    # pylint: disable=protected-access
    # for better convergence:

    with ops.colocate_with(None, ignore_existing=True):
      _slots = [
          optimizer.get_slot(self._v, _s) for _s in optimizer.get_slot_names()
      ]
      self._v._track_optimizer_slots(_slots)

      with ops.control_dependencies([g]):
        v0 = self._v.read_value(do_prefetch=not self._v.params.bp_v2)
        s0 = [_s.read_value() for _s in _slots]
        _before = [v0] + s0

      if isinstance(g, ops.IndexedSlices):
        if self._v.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")

        with ops.control_dependencies(_before):
          _apply_op = optimizer._resource_apply_sparse_duplicate_indices(
              g.values, self._v, g.indices)
        with ops.control_dependencies([_apply_op]):
          _after = control_flow_ops.group(
              [self._v.update_op(v0=v0)] +
              [_s.update_op(v0=s0[si]) for si, _s in enumerate(_slots)])
          return _after

      with ops.control_dependencies(_before):
        _apply_op = optimizer._resource_apply_dense(g, self._v)
      if self._v.constraint is not None:
        with ops.control_dependencies([_apply_op]):
          return self._v.assign(self._v.constraint(self._v))
      else:
        with ops.control_dependencies([_apply_op]):
          _after = control_flow_ops.group(
              [self._v.update_op(v0=v0)] +
              [_s.update_op(v0=s0[si]) for si, _s in enumerate(_slots)])
        return _after


def _get_processor(v):
  """The processor of v."""
  if isinstance(v, de.TrainableWrapper):
    return _DenseDynamicEmbeddingTrainableProcessor(v)
  if context.executing_eagerly():
    if isinstance(v, ops.Tensor):
      return optimizer._TensorProcessor(v)
    else:
      return optimizer._DenseResourceVariableProcessor(v)
  if (rvo.is_resource_variable(v) and not v._in_graph_mode):  # pylint: disable=protected-access
    # True if and only if `v` was initialized eagerly.
    return optimizer._DenseResourceVariableProcessor(v)
  if isinstance(v, ev.EmbeddingVariable):
    return optimizer._DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return optimizer._DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return optimizer._RefVariableProcessor(v)
  if isinstance(v, ops.Tensor):
    return optimizer._TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)


def _create_slot_var(primary,
                     val,
                     scope,
                     validate_shape,
                     shape,
                     dtype,
                     *,
                     copy_xla_sharding=False):
  """Helper function for creating a slot variable."""

  # TODO(lukaszkaiser): Consider allowing partitioners to be set in the current
  # scope.
  current_partitioner = variable_scope.get_variable_scope().partitioner
  variable_scope.get_variable_scope().set_partitioner(None)
  # When init from val instead of callable initializer, the shape is expected to
  # be None, not <unknown> or any fully defined shape.
  shape = shape if callable(val) else None
  if rvo.is_resource_variable(primary):
    use_resource = True
  elif isinstance(primary, variables.RefVariable):
    use_resource = False
  else:
    use_resource = None
  if isinstance(primary, ev.EmbeddingVariable):
    slot = ev.get_variable(scope,
                           embedding_dim=shape[1:],
                           initializer=val,
                           trainable=False,
                           key_dtype=primary._ktype,
                           value_dtype=primary.dtype)
  else:
    slot = variable_scope.get_variable(scope,
                                       initializer=val,
                                       trainable=False,
                                       use_resource=use_resource,
                                       shape=shape,
                                       dtype=dtype,
                                       validate_shape=validate_shape)
  variable_scope.get_variable_scope().set_partitioner(current_partitioner)

  # pylint: disable=protected-access
  if isinstance(primary, variables.Variable) and primary._save_slice_info:
    # Primary is a partitioned variable, so we need to also indicate that
    # the slot is a partitioned variable.  Slots have the same partitioning
    # as their primaries.
    # For examples when using AdamOptimizer in linear model, slot.name
    # here can be "linear//weights/Adam:0", while primary.op.name is
    # "linear//weight". We want to get 'Adam' as real_slot_name, so we
    # remove "'linear//weight' + '/'" and ':0'.
    real_slot_name = slot.name[len(primary.op.name + "/"):-2]
    slice_info = primary._save_slice_info
    # support slot's shape not same as primary's shape
    # example: primary's shape = [10, 20, 30], slot's shape =
    # None, [], [10], [10, 20] or [10, 20, 30] is allowed
    # slot's shape = None or [10, 20, 30], set slot's slice_info same as primary
    # slot's shape = [], don't set slot's slice_info
    # slot's shape = [10] or [10, 20], set slot's slice_info according to ndims
    n = slot.shape.ndims
    if n is None or n > 0:
      slot._set_save_slice_info(
          variables.Variable.SaveSliceInfo(
              slice_info.full_name + "/" + real_slot_name,
              slice_info.full_shape[:n], slice_info.var_offset[:n],
              slice_info.var_shape[:n]))
  # pylint: enable=protected-access

  # Copy XLA sharding attributes from primary.
  if copy_xla_sharding:
    try:
      from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
      slot = xla_sharding.copy_sharding(primary, slot, use_sharding_op=False)
    except ImportError:
      tf_logging.warn("xla_sharding not found, maybe in tf version < 2.5")
      pass
  return slot


def device_function(self, op):
  """Choose a device for `op`.

    Args:
      op: an `Operation`.

    Returns:
      The device to use for the `Operation`.
    """
  # If we don't return early here, either merge_devices is True, or op.device
  # is empty (in which case merging is a no-op). So we can always merge below.
  if not self._merge_devices and op.device:
    return op.device

  current_device = pydev.DeviceSpec.from_string(op.device or "")

  # The ps_device will be used for specified ops (ps_ops) whenever it is
  # present and ps_tasks is non-zero. However, its task number will only be
  # set (using ps_strategy) if there is a job field in ps_device that won't be
  # changed by the job field (if present) in current_device.
  node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def

  # TODO(rhdong): `TrainableWrapper` is not multi-threads safe so we try to
  #  prevent to place the `TrainableWrapper` on PS,
  #  a less bad way of avoiding handle of `TrainableWrapper` be
  #  placed on the PS devices for node_def carries too little information to
  #  know if it was created by `TrainableWrapper` or not.
  if ("TrainableWrapper" not in node_def.name and self._ps_tasks
      and self._ps_device and node_def.op in self._ps_ops):
    ps_device = pydev.DeviceSpec.from_string(self._ps_device)

    current_job, ps_job = current_device.job, ps_device.job
    if ps_job and (not current_job or current_job == ps_job):
      ps_device = ps_device.replace(task=self._ps_strategy(op))

    ps_device = ps_device.make_merged_spec(current_device)
    return ps_device.to_string()
  worker_device = pydev.DeviceSpec.from_string(self._worker_device or "")
  worker_device = worker_device.make_merged_spec(current_device)
  return worker_device.to_string()


def _assert_float_dtype(dtype):
  dtype = dtypes.as_dtype(dtype)
  if not dtype.is_floating:
    raise ValueError("Expected floating point type, got %s." % dtype)
  return dtype


def _compute_fans_for_keras_init_v1_v2(shape):
  """ Making keras VarianceScaling initializers v1 & v2 support dynamic shape.
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


def __call__for_keras_init_v1(self, shape, dtype=None, partition_info=None):
  """ Making keras VarianceScaling initializers v1 support dynamic shape.
  """
  if dtype is None:
    dtype = self.dtype
  scale = self.scale
  scale_shape = shape
  if partition_info is not None:
    scale_shape = partition_info.full_shape
  fan_in, fan_out = _compute_fans_for_keras_init_v1_v2(scale_shape)
  fan_in = math_ops.cast(fan_in, dtype=dtype)
  fan_out = math_ops.cast(fan_out, dtype=dtype)
  if self.mode == "fan_in":
    scale /= math_ops.maximum(1., fan_in)
  elif self.mode == "fan_out":
    scale /= math_ops.maximum(1., fan_out)
  else:
    scale /= math_ops.maximum(1., (fan_in + fan_out) / 2.)
  if self.distribution == "normal" or self.distribution == "truncated_normal":
    # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    stddev = math_ops.sqrt(scale) / .87962566103423978
    return random_ops.truncated_normal(shape,
                                       0.0,
                                       stddev,
                                       dtype,
                                       seed=self.seed)
  elif self.distribution == "untruncated_normal":
    stddev = math_ops.sqrt(scale)
    return random_ops.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)
  else:
    limit = math_ops.sqrt(3.0 * scale)
    return random_ops.random_uniform(shape,
                                     -limit,
                                     limit,
                                     dtype,
                                     seed=self.seed)


def __call__for_keras_init_v2(self, shape, dtype=None, **kwargs):
  """ Making keras VarianceScaling initializers v2 support dynamic shape.
  """
  if hasattr(kinit2, "_validate_kwargs"):
    kinit2._validate_kwargs(self.__class__.__name__, kwargs)
  elif hasattr(self, "_validate_kwargs"):
    self._validate_kwargs(kwargs)

  if hasattr(kinit2, "_get_dtype"):
    dtype = _assert_float_dtype(kinit2._get_dtype(dtype))
  else:
    dtype = _assert_float_dtype(dtype)

  scale = self.scale
  fan_in, fan_out = _compute_fans_for_keras_init_v1_v2(shape)
  fan_in = math_ops.cast(fan_in, dtype=dtype)
  fan_out = math_ops.cast(fan_out, dtype=dtype)
  if _PARTITION_SHAPE in kwargs:
    shape = kwargs[_PARTITION_SHAPE]
  if self.mode == 'fan_in':
    scale /= math_ops.maximum(1., fan_in)
  elif self.mode == 'fan_out':
    scale /= math_ops.maximum(1., fan_out)
  else:
    scale /= math_ops.maximum(1., (fan_in + fan_out) / 2.)
  if self.distribution == 'truncated_normal':
    # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    stddev = math_ops.sqrt(scale) / .87962566103423978
    return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)
  elif self.distribution == 'untruncated_normal':
    stddev = math_ops.sqrt(scale)
    return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
  else:
    limit = math_ops.sqrt(3.0 * scale)
  return self._random_generator.random_uniform(shape, -limit, limit, dtype)


def patch_on_tf():
  optimizer._get_processor = _get_processor
  slot_creator._create_slot_var = _create_slot_var
  device_setter._ReplicaDeviceChooser.device_function = device_function
  if kinit1 is not None:
    kinit1.VarianceScaling.__call__ = __call__for_keras_init_v1
  if kinit2 is not None:
    kinit2.VarianceScaling.__call__ = __call__for_keras_init_v2
