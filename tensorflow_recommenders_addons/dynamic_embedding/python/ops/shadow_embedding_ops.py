# Copyright 2021 The TensorFlow Recommenders-Addons Authors.
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

`The file will be introduced as `shadow_ops` under `dynamic_embedding`.
It is a submodule of `dynamic_embedding`.

In TensorFlow 2.x, tf.function is introduced to speedup the computation.
And also modular programming based on [tf.Module](https://www.tensorflow.org/guide/intro_to_modules)
are recommended because of the Pythonic style APIs. But APIs like
`embedding_lookup`, `embedding_lookup_unique`, `embedding_lookup_sparse`, and
`safe_embedding_lookup_sparse` in `dynamic_embedding`, are wrappers of
`embedding_lookup`. And it will create a TrainableWrapper object inside
the function, which doesn't meet the requirements of
[tf.function](https://www.tensorflow.org/guide/function)

The `shadow_ops` submodule is designed to support usage on `tf.function`
and modular style development, like keras.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.tracking import base as trackable


class ShadowVariable(de.TrainableWrapper):
  """
  ShadowVariable is a eager persistent twin of TrainableWrapper.

  ShadowVariable maps the sparse domain, which may reside cross multiple
  devices, as a projection on current device. Its value represents the activated
  part of the sparse domain. When lookup on sparse domain, it will fetch the
  lookup result to local, and could be regarded as trainable object to
  optimizers, like an ordinary variable. It supports the modular programming
  and [tf.function](https://www.tensorflow.org/guide/function).
  """

  def __init__(self,
               params,
               name='ShadowVariable',
               max_norm=None,
               trainable=True,
               distribute_strategy=None,
               **kwargs):
    """
    Create a ShadowVariable object.

    Args:
      params: A dynamic_embedding.Variable object represents the sparse domain.
      ids: If set, it needs to be a ResourceVariable, to keep the
        ids for backward computations. Otherwise the ShadowVariable will
        create ids variable buffer itself.
      name: Name of the ShadowVariable.
      max_norm: If not `None`, each values is clipped if its l2-norm is larger
        than this value.
      trainable: Bool. If true, the variable will be treated as trainable.
        Default is true.
      distribute_strategy: DistributeStrategy.

      **kwargs:
        model_mode: ModelMode of the option. Default is ModelMode.CURRENT_SETTING.
          often used internally.
        ids: A Buffer to store the feature ids. If None, it use a private one.
        exists: A Buffer to indicate whether the feature ids exist in sparse domain.
          If None, it use a private one.
    """
    if not context.executing_eagerly():
      raise NotImplementedError('Currently ShadowVariable is only allowed'
                                ' in eager mode.')

    self._name = name
    if not isinstance(params, de.Variable):
      raise TypeError('params must be de.Variable, but get %s' % type(params))
    self.params = params
    collections = kwargs.get('collections', None)
    ids = kwargs.get('ids', None)
    if ids is not None:
      kwargs.pop('ids')
    ids_name = self._name + '-ids'
    if ids is None:
      self.ids = resource_variable_ops.ResourceVariable(
          (),
          trainable=False,
          collections=collections,
          name=ids_name,
          dtype=self.params.key_dtype,
          shape=tensor_shape.TensorShape(None))
      self._track_trackable(self.ids, ids_name, overwrite=False)
    else:
      if not isinstance(ids, resource_variable_ops.ResourceVariable):
        raise TypeError('If ids is set, it needs to be a ResourceVariable')
      self.ids = ids

    model_mode = kwargs.get('model_mode', None)
    if model_mode:
      kwargs.pop('model_mode')
    else:
      model_mode = de.ModelMode.CURRENT_SETTING
    initial_value = array_ops.zeros(shape=(0, self.params.dim),
                                    dtype=self.params.value_dtype)

    if (distribute_strategy is not None) and (not isinstance(
        distribute_strategy, distribute_lib.StrategyBase)):
      raise TypeError('distribute_strategy must inherit from StrategyBase.')

    super(ShadowVariable,
          self).__init__(self.params,
                         self.ids,
                         max_norm=max_norm,
                         initial_value=initial_value,
                         dtype=self.params.value_dtype,
                         trainable=trainable,
                         collections=collections,
                         model_mode=model_mode,
                         distribute_strategy=distribute_strategy,
                         name=name)
    exists = kwargs.get('exists', None)
    exists_name = self._name + '-exists'
    if exists is None:
      self.exists = resource_variable_ops.ResourceVariable(
          (),
          trainable=False,
          collections=collections,
          name=exists_name,
          dtype=dtypes.bool,
          shape=tensor_shape.TensorShape(None))
      self._track_trackable(self.exists, exists_name, overwrite=False)
    else:
      self.exists = exists
    self.params._trainable_store[name] = self

  def prefetch_values(self, update=False):
    if self.params.bp_v2:
      r, exists = self.params.lookup(self.ids, return_exists=True)
      self.exists.assign(exists)
      self.prefetch_values_op = self.transform(r)
    else:
      self.prefetch_values_op = self.transform(self.params.lookup(self.ids))
    return self.prefetch_values_op

  def value(self, do_prefetch=False):
    """A cached operation which reads the value of this variable."""
    if self._cached_value is not None:
      return self._cached_value
    with ops.colocate_with(None, ignore_existing=True):
      return self._read_variable_op(do_prefetch=do_prefetch)

  def assign(self, value, use_locking=None, name=None, read_value=True):
    """
    Assigns a new value to this variable.
    To discriminate with ResourceVariable, the shadow always uses a
    variant space to hold the temporary embedding lookup buffer.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name to use for the assignment.
      read_value: A `bool`. Whether to read and return the new value of the
        variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    """
    # Note: not depending on the cached value here since this can be used to
    # initialize the variable.
    with resource_variable_ops._handle_graph(self.handle):
      value_tensor = ops.convert_to_tensor(value, dtype=self.dtype)
      assign_op = gen_resource_variable_ops.assign_variable_op(self.handle,
                                                               value_tensor,
                                                               name=name)
      if read_value:
        return self._lazy_read(assign_op)
    return assign_op

  def _reset_ids(self, ids):
    return self.ids.assign(ids, use_locking=True)

  def _gather_saveables_for_checkpoint(self):
    self._reset_ids(array_ops.zeros((0,), dtype=self.params.key_dtype))
    self.read_value(do_prefetch=True)
    for s in self._tracked_slots:
      s.read_value(do_prefetch=True)
    return {trackable.VARIABLE_VALUE_KEY: self}


def embedding_lookup(
    shadow,
    ids,
    partition_strategy=None,  # pylint: disable=unused-argument
    name=None,
    validate_indices=None,  # pylint: disable=unused-argument
):
  """
  Shadow version of dynamic_embedding.embedding_lookup. It use existed shadow
  variable to to embedding lookup, and store the result. No by-product will
  be introduced in this call. So it can be decorated by `tf.function`.

  Args:
    shadow: A ShadowVariable object.
    ids: A tensor with any shape as same dtype of params.key_dtype.
    partition_strategy: No used, for API compatiblity with `nn.emedding_lookup`.
    name: A name for the operation.
    validate_indices: No used, just for compatible with nn.embedding_lookup .

  Returns:
    A tensor with shape [shape of ids] + [dim],
      dim is equal to the value dim of params.
      containing the values from the params tensor(s) for keys in ids.
  """
  ids = ops.convert_to_tensor(ids)

  if shadow.ids.dtype != ids.dtype:
    raise ValueError('{} ids is not matched with ShadowVariable with ids'
                     ' {},'.format(ids.dtype, shadow.ids.dtype))

  with ops.name_scope(name, "shadow_embedding_lookup"):
    if de.ModelMode.CURRENT_SETTING == de.ModelMode.TRAIN:
      with ops.control_dependencies([shadow._reset_ids(ids)]):
        return shadow.read_value(do_prefetch=True)
    else:
      return shadow.params.lookup(ids)
