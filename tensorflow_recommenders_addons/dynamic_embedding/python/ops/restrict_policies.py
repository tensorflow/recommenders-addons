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
"""Policies for restricting the dynamic embedding variable"""
from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import make_partition

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging


class RestrictPolicy(object):
  """
  Base class of restrict policies. Never use this class directly, but
  instead of of its derived class.

  This class defines the rules for tracking and restricting the size of the
  `dynamic_embedding.Variable`. If the variable joins training via stateful
  optimizer, the policy also manage the slots of the optimizer. It could own
  a status to keep tracking the state of affairs of the features presented
  in sparse `dynamic_embedding.Variable`.

  `RestrictPolicy` requires a set of methods to be override, in its derived
  policies: `apply_update`, `apply_restriction`, `status`.

  * apply_update: keep tracking on the status of the sparse variable,
      specifically the attributes of each key in sparse variable.
  * apply_restriction: eliminate the features which are not legitimate.
  """

  def __init__(self, var):
    """
    Create a new RestrictPolicy.

    Args:
      var: A `dynamic_embedding.Variable` object to be restricted.
    """
    self.var = var
    self.params_in_slots = []

  def apply_update(self, ids):
    """
    Define the rule to update status, with tracking the
    changes in variable and slots.

    Args:
      ids: A Tensor. Keys appear in training. These keys in status
        variable will be updated if needed.

    Returns:
      An operation to update status.
    """
    raise NotImplementedError

  def apply_restriction(self, num_reserved, **kwargs):
    """
    Define the rule to restrict the size of the target variable. There are
    three kinds of variables are token into consideration: variable, status
    variable, and variables in slots. Number of `num_reserved` features will
    be kept in variable.

    Args:
      num_reserved: number of remained keys after restriction.
      **kwargs: (Optional) reserved keyword arguments.

    Returns:
      An operation to restrict the sizes of variable and variables in slots.
    """
    raise NotImplementedError

  @property
  def status(self):
    """
    Get status variable which save information about properties of features.
    """
    return None

  def _track_params_from_optimizer_slots(self, slots):
    for _s in slots:
      if isinstance(_s, de.TrainableWrapper):
        params = _s.params
      elif isinstance(_s, de.Variable):
        params = _s
      else:
        raise TypeError('slots should be dynamic_embedding.TrainableWrapper'
                        'or dynamic_embedding.Variable. But get {}'.format(
                            type(_s)))
      # TODO(Lifann) Use unique identifier instead of Python id().
      id_collection = [id(p) for p in self.params_in_slots]
      if id(params) not in id_collection:
        self.params_in_slots.append(params)


class TimestampRestrictPolicy(RestrictPolicy):
  """
  A derived policy to eliminate features in variable follow the
  `oldest-out-first` rule.
  """

  def __init__(self, var):
    """
    A timestamp status sparse variable is created. The timestamp status
    has same key_dtype as the target variable and value_dtype in int32,
    which indicates the timestamp value. The timestamp means a digital
    record of time. The later the time, the larger the timestamp.

    Args:
      var: A `dynamic_embedding.Variable` object to be restricted.
    """
    super(TimestampRestrictPolicy, self).__init__(var)
    scope = variable_scope.get_variable_scope()
    if scope.name:
      tstp_scope = scope.name + '/status'
    else:
      tstp_scope = 'status'
    tstp_name = self.var.name + '/timestamp'

    with ops.name_scope(tstp_scope, 'status', []) as unique_scope:
      if unique_scope:
        full_name = unique_scope + tstp_name
      else:
        full_name = tstp_name

      self.tstp_var = de.get_variable(full_name,
                                      key_dtype=self.var.key_dtype,
                                      value_dtype=dtypes.int32,
                                      dim=1,
                                      devices=self.var.devices,
                                      partitioner=self.var.partition_fn,
                                      trainable=False,
                                      init_size=self.var.init_size,
                                      kv_creator=self.var.kv_creator)

  def apply_update(self, ids):
    """
    Define the rule to update the timestamp status. If any feature shows up
    in training, then its timestamp will be updated.

    Args:
      ids: A Tensor. Keys appear in training. These keys in status variable
        will be updated if needed.

    Returns:
      An operation to update timestamp status.
    """
    keys = array_ops.reshape(ids, (-1,))
    fresh_tstp = array_ops.tile(
        array_ops.reshape(gen_logging_ops.timestamp(), [1]),
        array_ops.reshape(array_ops.size(keys), (-1,)),
    )
    fresh_tstp = math_ops.cast(fresh_tstp, dtypes.int32)
    fresh_tstp = array_ops.reshape(fresh_tstp, (-1, 1))
    update_tstp_op = self.tstp_var.upsert(keys, fresh_tstp)
    return update_tstp_op

  def apply_restriction(self, num_reserved, **kwargs):
    """
    Define the rule to restrict the size of the target variable by eliminating
    the oldest k features, and number of `num_reserved` feature will be kept.

    Args:
      num_reserved: int. Number of remained keys after restriction.
      **kwargs: (Optional) reserved keyword arguments.
        trigger: int. The triggered threshold to execute restriction. Default
          equals to `num_reserved`.

    Returns:
      An operation to restrict the sizes of variable and variables in slots.
    """
    if not isinstance(num_reserved, int):
      raise TypeError('num_reserved should be integer.')
    self._num_reserved = num_reserved
    trigger = kwargs.get('trigger', num_reserved)
    if not isinstance(trigger, int):
      raise TypeError('trigger should be integer.')

    is_oversize = math_ops.greater(self.var.size(), trigger)
    return control_flow_ops.cond(is_oversize, self._cond_restrict_fn,
                                 control_flow_ops.no_op)

  def _cond_restrict_fn(self):
    """
    This will only be execute when size of variable is larger than trigger.
    """
    restrict_var_ops, restrict_status_ops, restrict_slot_ops = [], [], []
    for i, dev in enumerate(self.tstp_var.devices):
      with ops.device(dev):
        partial_keys, partial_tstp = self.tstp_var.tables[i].export()
        partial_reserved = int(self._num_reserved / self.tstp_var.shard_num)
        partial_tstp = array_ops.reshape(partial_tstp, (-1,))
        first_dim = array_ops.shape(partial_tstp)[0]

        k_on_top = math_ops.cast(first_dim - partial_reserved,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_key_indices = nn_ops.top_k(-partial_tstp,
                                              k_on_top,
                                              sorted=False)
        removed_keys = array_ops.gather(partial_keys, removed_key_indices)
        restrict_var_ops.append(self.var.tables[i].remove(removed_keys))
        restrict_status_ops.append(self.tstp_var.tables[i].remove(removed_keys))
        for slot_param in self.params_in_slots:
          restrict_slot_ops.append(slot_param.tables[i].remove(removed_keys))
    return control_flow_ops.group(restrict_var_ops, restrict_status_ops,
                                  restrict_slot_ops)

  @property
  def status(self):
    return self.tstp_var


class FrequencyRestrictPolicy(RestrictPolicy):
  """
  A derived policy to eliminate features in variable follow the
  `lowest-occurrence-out-first` rule.
  """

  def __init__(self, var):
    """
    A frequency status sparse variable is created. The frequency status has
    same key_dtype as the target variable and value_dtype in `int32`, which
    indicates the occurrence times of the feature.

    Args:
      var: A `dynamic_embedding.Variable` object to be restricted.
    """
    super(FrequencyRestrictPolicy, self).__init__(var)
    self.init_count = constant_op.constant(0, dtypes.int32)

    scope = variable_scope.get_variable_scope()
    if scope.name:
      freq_scope = scope.name + '/status'
    else:
      freq_scope = 'status'
    freq_name = self.var.name + '/frequency'

    with ops.name_scope(freq_scope, 'status', []) as unique_scope:
      if unique_scope:
        full_name = unique_scope + freq_name
      else:
        full_name = freq_name

      self.freq_var = de.get_variable(full_name,
                                      key_dtype=self.var.key_dtype,
                                      value_dtype=dtypes.int32,
                                      dim=1,
                                      devices=self.var.devices,
                                      partitioner=self.var.partition_fn,
                                      trainable=False,
                                      init_size=self.var.init_size,
                                      kv_creator=self.var.kv_creator)

  def apply_update(self, ids):
    """
    Define the rule to update the frequency status. If any feature shows up,
    then its frequency value will be increased by 1.

    Args:
      ids: A Tensor. Keys appear in training. These keys in status variable
        will be updated if needed.

    Returns:
      An operation to update timestamp status.
    """
    update_status_ops = []
    keys = array_ops.reshape(ids, (-1,))
    partitioned_indices = self.var.partition_fn(keys, self.var.shard_num)
    partitioned_keys, _ = make_partition(keys, partitioned_indices,
                                         self.var.shard_num)
    for i, dev in enumerate(self.freq_var.devices):
      with ops.device(dev):
        feature_counts = self.freq_var.tables[i].lookup(
            partitioned_keys[i], dynamic_default_values=self.init_count)
        feature_counts += 1
        update_table_op = self.freq_var.tables[i].insert(
            partitioned_keys[i], feature_counts)
        update_status_ops.append(update_table_op)
    return control_flow_ops.group(update_status_ops)

  def apply_restriction(self, num_reserved, **kwargs):
    """
    Define the rule to restrict the size of the target variable by eliminating
    k features with least occurrence, and number of `num_reserved` features will
    be left.

    Args:
      num_reserved: int. Number of remained keys after restriction.
      **kwargs: (Optional) reserved keyword arguments.
        trigger: int. The triggered threshold to execute restriction. Default
          equals to `num_reserved`.

    Returns:
      An operation to do restriction.
    """
    if not isinstance(num_reserved, int):
      raise TypeError('num_reserved should be integer.')
    if num_reserved < 0:
      raise ValueError('num_reserved should be non-negative.')
    self._num_reserved = num_reserved
    trigger = kwargs.get('trigger', num_reserved)
    if not isinstance(trigger, int):
      raise TypeError('trigger should be integer.')

    is_oversize = math_ops.greater(self.var.size(), trigger)
    return control_flow_ops.cond(is_oversize, self._cond_restrict_fn,
                                 control_flow_ops.no_op)

  def _cond_restrict_fn(self):
    """
    This will only be execute when size of variable is larger than trigger.
    """
    restrict_var_ops, restrict_status_ops, restrict_slot_ops = [], [], []

    for i, dev in enumerate(self.freq_var.devices):
      with ops.device(dev):
        partial_keys, partial_counts = self.freq_var.tables[i].export()
        partial_reserved = int(self._num_reserved / self.freq_var.shard_num)
        partial_counts = array_ops.reshape(partial_counts, (-1,))
        first_dim = array_ops.shape(partial_counts)[0]

        k_on_top = math_ops.cast(first_dim - partial_reserved,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_key_indices = nn_ops.top_k(-partial_counts,
                                              k_on_top,
                                              sorted=False)
        removed_keys = array_ops.gather(partial_keys, removed_key_indices)

        restrict_var_ops.append(self.var.tables[i].remove(removed_keys))
        restrict_status_ops.append(self.freq_var.tables[i].remove(removed_keys))
        for slot_param in self.params_in_slots:
          restrict_slot_ops.append(slot_param.tables[i].remove(removed_keys))
    return control_flow_ops.group(restrict_var_ops, restrict_status_ops,
                                  restrict_slot_ops)

  @property
  def status(self):
    return self.freq_var
