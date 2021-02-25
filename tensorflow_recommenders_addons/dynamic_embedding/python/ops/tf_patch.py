# Copyright 2020 The TensorFlow Recommenders-Addpnons Authors.
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

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops as rvo
from tensorflow.python.ops import variables
from tensorflow.python.training import device_setter
from tensorflow.python.training import optimizer


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
      with ops.control_dependencies([g]):
        _before = [self._v.read_value()] + [_s.read_value() for _s in _slots]
      if isinstance(g, ops.IndexedSlices):
        if self._v.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")

        with ops.control_dependencies(_before):
          _apply_op = optimizer._resource_apply_sparse_duplicate_indices(
              g.values, self._v, g.indices)
        with ops.control_dependencies([_apply_op]):
          _after = control_flow_ops.group([self._v.update_op()] +
                                          [_s.update_op() for _s in _slots])
          return _after
      with ops.control_dependencies(_before):
        _apply_op = optimizer._resource_apply_dense(g, self._v)
      if self._v.constraint is not None:
        with ops.control_dependencies([_apply_op]):
          return self._v.assign(self._v.constraint(self._v))
      else:
        with ops.control_dependencies([_apply_op]):
          _after = control_flow_ops.group([self._v.update_op()] +
                                          [_s.update_op() for _s in _slots])
        return _after


def _get_processor(v):
  """The processor of v."""
  if context.executing_eagerly():
    if isinstance(v, ops.Tensor):
      return optimizer._TensorProcessor(v)
    else:
      return optimizer._DenseResourceVariableProcessor(v)
  if isinstance(v, de.TrainableWrapper):
    return _DenseDynamicEmbeddingTrainableProcessor(v)
  if (rvo.is_resource_variable(v) and not v._in_graph_mode):  # pylint: disable=protected-access
    # True if and only if `v` was initialized eagerly.
    return optimizer._DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return optimizer._DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return optimizer._RefVariableProcessor(v)
  if isinstance(v, ops.Tensor):
    return optimizer._TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)


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


def patch_on_tf():
  optimizer._get_processor = _get_processor
  device_setter._ReplicaDeviceChooser.device_function = device_function
