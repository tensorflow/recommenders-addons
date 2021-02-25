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
"""Restrain dynamic embedding variable with specified size."""
from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable import _partition

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import optimizer
from tensorflow.python.util import nest


def _select_relative_trainable_wrappers(var, trainable_wrappers):
  if trainable_wrappers:
    chosen_tws = []
    tw_id_collection = [
        optimizer_v2._var_key(vtw) for vtw in var.trainable_wrappers
    ]
    for tw in trainable_wrappers:
      if optimizer_v2._var_key(tw) in tw_id_collection:
        chosen_tws.append(tw)
  else:
    chosen_tws = var.trainable_wrappers
  return chosen_tws


def _select_slot_vars(var, optmz):
  var_in_slots = []
  for tw in var.trainable_wrappers:
    slots = []
    for name in optmz.get_slot_names():
      try:
        _s = optmz.get_slot(tw, name)
        var_in_slots.append(_s.params)
      except:
        tf_logging.warn('trainable_wrapper {} not'
                        ' applied.'.format(tw._unique_id))
  return var_in_slots


def _flatten(fold):
  return [item for sublist in fold for item in sublist]


class RestrictPolicy(object):
  """
  RestrictPolicy is a base class for recording and restricting the
  size of the `dynamic_embedding.Variable`. If the variable joins
  training via stateful optimizer, the policy also manage the slots
  of the optimizer.

  `RestrictPolicy` requires a set of interfaces `create_status`,
  `update_rule`, and `restrict_rule`.

  1. `create_status` make a record on the strategy defined informantion
    for comparing the importance of evey feature.
  2. `update_rule` provides a method to keep the status created by
    `create_status` tracing on dynamic of training.
  3. `restrict_rule` provides a method to eliminate features which are
    not qualified for certain standard.
  """

  def __init__(self, var, optmz, **kwargs):
    """
    Create a `RestrictPolicy` object with giving variable and optimizer.

    Args:
      var: An `dynamic_embedding.Variable` object.
      optmz: A `tf.train.Optimizer` object.
      **kwargs: Optional keyword arguments.
    """
    if not isinstance(var, de.Variable):
      raise TypeError("Parameter var type should be {}, but get {} "
                      "instead.".format(type(de.Variable), type(var)))
    if not isinstance(optmz, (optimizer.Optimizer, optimizer_v2.OptimizerV2)):
      raise TypeError("Parameter optmz type should be Optimizer or "
                      "OptimizerV2, but get {} instead.".format(type(optmz)))

    self.var = var
    self.optmz = optmz
    self.create_status(**kwargs)

  def create_status(self, **kwargs):
    """
    Create variable's status. Generally its value is related to
    property of the keys, such as life-span or occurrence counts.
    And its keys have same dtype as the target variable.
    """
    raise NotImplementedError

  def update_rule(self, trainable_wrappers=None, **kwargs):
    """
    Define the rule to update status, which was created in method
    `create_status`.

    Args:
      trainable_wrappers: A list of `TrainableWrapper` objects. The
        status of variable is only updated by trainable_wrappers, to
        avoid consuming unrelated indices when embedding is shared by
        multiple optimizers, and only parts of then work. `None` means
        update all embeddings.
      **kwargs: Optional keyword arguments.

    Returns:
      An operation updates trainable_wrappers.
    """
    raise NotImplementedError

  def restrict_rule(self, **kwargs):
    """
    Define the rule to restrict the size of the target variable. There
    usually 3 kinds of variables are related: the target variable, the
    status variable, and the slots in optimizer.
    """
    raise NotImplementedError


class TimestampRestrictPolicy(RestrictPolicy):
  """
  An derived policy to eliminate features in variable follow
  the "oldest-out-first" rule.
  """

  def __init__(self, var, optmz, **kwargs):
    """
    Args:
      var: An `dynamic_embedding.Variable` object.
      optmz: A `tf.train.Optimizer` object.
      **kwargs: Unused.
    """
    self.tstp_var = None
    self.residue = 0
    super(TimestampRestrictPolicy, self).__init__(var, optmz)

  def create_status(self, **kwargs):
    """
    Create a timestamp status in `name_scope` protection, since
    the name of an optimizer is not guaranteed to be unique.

    Args:
      **kwargs: Unused.
    """
    scope = variable_scope.get_variable_scope()
    if scope.name:
      tstp_scope = scope.name + '/' + 'timestamp_status'
    else:
      tstp_scope = 'timestamp_status'
    status_name = self.var.name + '/' + self.optmz._name

    with ops.name_scope(tstp_scope, 'timestamp_status', []) as unique_scope:
      if unique_scope:
        full_name = unique_scope + status_name
      else:
        full_name = status_name

      self.tstp_var = de.get_variable(
          full_name,
          key_dtype=self.var.key_dtype,
          value_dtype=dtypes.int32,
          dim=1,
          devices=self.var.devices,
          partitioner=self.var.partition_fn,
          trainable=False,
      )

  def update_rule(self, trainable_wrappers=None, **kwargs):
    """
    Define the rule to update timestamp status of the variable.
    It will upsert a set of keys into the timestamp status. It
    will consume a set of inputs data. Make sure it uses the same
    inputs data as the training step.

    Args:
      trainable_wrappers: A list of `TrainableWrapper` objects. The
        status of variable is only updated by trainable_wrappers, to
        avoid consuming unrelated indices when embedding is shared by
        multiple optimizers, and only parts of then work. `None` means
        update all embeddings.
      **kwargs: Unused.

    Returns:
      An operation updates trainable_wrappers.
    """
    chosen_tws = _select_relative_trainable_wrappers(self.var,
                                                     trainable_wrappers)
    update_ops = []
    for tw in chosen_tws:
      keys = array_ops.reshape(tw.ids, (-1,))
      fresh_tstp = array_ops.tile(
          array_ops.reshape(gen_logging_ops.timestamp(), [1]),
          array_ops.reshape(array_ops.size(keys), (-1,)),
      )
      fresh_tstp = math_ops.cast(fresh_tstp, dtypes.int32)
      fresh_tstp = array_ops.reshape(fresh_tstp, (-1, 1))
      tstp_update = self.tstp_var.upsert(keys, fresh_tstp)

      update_ops.append(tstp_update)

    return control_flow_ops.group(update_ops)

  def restrict_rule(self, **kwargs):
    """
    Define the rule to restrict the size of the target
    variable by eliminating the oldest k features.

    Args:
      **kwargs: Keyword arguments, including
        residue: int. The leftover number of features after restriction. 
        trigger: int. The trigger threshold to execute the restriction.
          If the number of feature is less than `trigger` in real time,
          nothing happens. The restriction will only be executed when the
          feature number is greater than `trigger`. Default equals to
          `residue`.

    Returns:
      An restriction operation.
    """
    try:
      self.residue = kwargs['residue']
    except:
      raise KeyError('TimestampRestrictPolicy requires'
                     '`residue` keyword in restrict_rule.')
    if not isinstance(self.residue, int):
      raise TypeError('TimestampRestrictPolicy requires integer `residue`.')
    if self.residue < 0:
      raise ValueError('TimestampRestrictPolicy requires'
                       'non-negative `residue` value.')
    trigger = kwargs.get('trigger', self.residue)
    if not isinstance(trigger, int):
      raise TypeError('`trigger` needs to be an integer.')

    is_overflow = math_ops.greater(self.var.size(), trigger)
    return control_flow_ops.cond(is_overflow, self._cond_restrict,
                                 self._cond_no_op)

  def _cond_restrict(self):
    var_in_slots = _select_slot_vars(self.var, self.optmz)
    restrict_var_ops, restrict_status_ops, restrict_slot_ops = [], [], []

    for idx, dev in enumerate(self.tstp_var.devices):
      with ops.device(dev):
        partial_keys, partial_tstp = self.tstp_var.tables[idx].export()
        partial_residue = int(self.residue / self.tstp_var.shard_num)
        partial_tstp = array_ops.reshape(partial_tstp, (-1,))
        first_dim = array_ops.shape(partial_tstp)[0]

        k_on_top = math_ops.cast(first_dim - partial_residue,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_key_indices = nn_ops.top_k(-partial_tstp,
                                              k_on_top,
                                              sorted=False)
        removed_keys = array_ops.gather(partial_keys, removed_key_indices)

        restrict_var_ops.append(self.var.tables[idx].remove(removed_keys))
        restrict_status_ops.append(
            self.tstp_var.tables[idx].remove(removed_keys))
        for slot_var in var_in_slots:
          restrict_slot_ops.append(slot_var.tables[idx].remove(removed_keys))

    return control_flow_ops.group(restrict_var_ops, restrict_status_ops,
                                  restrict_slot_ops)

  def _cond_no_op(self):
    return control_flow_ops.no_op()


class FrequencyRestrictPolicy(RestrictPolicy):
  """
  An derived policy to eliminate features in variable
  follow the "lowest-occurrence-out-first" rule.
  """

  def __init__(self, var, optmz, **kwargs):
    """
    Args:
      var: An `dynamic_embedding.Variable` object.
      optmz: A `tf.train.Optimizer` object.
      **kwargs: Unused.
    """
    self.freq_var = None
    self.residue = 0
    self.default_count = constant_op.constant(0, dtypes.int32)
    super(FrequencyRestrictPolicy, self).__init__(var, optmz)

  def create_status(self, **kwargs):
    """
    Create a frequency status in `name_scope` protection, since
    the name of an optimizer is not guaranteed to be unique.

    Args:
      **kwargs: Unused.
    """
    scope = variable_scope.get_variable_scope()
    if scope.name:
      freq_scope = scope.name + '/frequency_status'
    else:
      freq_scope = 'frequency_status'
    status_name = self.var.name + '/' + self.optmz._name

    with ops.name_scope(freq_scope, 'frequency_status', []) as unique_scope:
      if unique_scope:
        full_name = unique_scope + status_name
      else:
        full_name = status_name

      self.freq_var = de.get_variable(
          full_name,
          key_dtype=self.var.key_dtype,
          value_dtype=dtypes.int32,
          dim=1,
          devices=self.var.devices,
          partitioner=self.var.partition_fn,
          initializer=self.default_count,
          trainable=False,
      )

  def update_rule(self, trainable_wrappers=None, **kwargs):
    """
    Define the rule to update frequency status of the variable.
    It will upsert a set of keys into the frequency status. It
    will consume a set of inputs data. Make sure it uses the same
    inputs data as the training step.

    Args:
      trainable_wrappers: A list of `TrainableWrapper` objects. The
        status of variable is only updated by trainable_wrappers, to
        avoid consuming unrelated indices when embedding is shared by
        multiple optimizers, and only parts of then work. `None` means
        update all embeddings.
      **kwargs: Unused.

    Returns:
      An operation updates trainable_wrappers.
    """
    chosen_tws = _select_relative_trainable_wrappers(self.var,
                                                     trainable_wrappers)
    update_ops = []
    for tw in chosen_tws:
      keys = array_ops.reshape(tw.ids, (-1,))
      partitioned_indices = self.var.partition_fn(keys, self.var.shard_num)
      partitioned_keys, _ = _partition(keys, partitioned_indices,
                                       self.var.shard_num)

      for idx, dev in enumerate(self.freq_var.devices):
        with ops.device(dev):
          feature_counts = self.freq_var.tables[idx].lookup(
              partitioned_keys[idx], dynamic_default_values=self.default_count)
          feature_counts += 1
          update_table_op = self.freq_var.tables[idx].insert(
              partitioned_keys[idx], feature_counts)
          update_ops.append(update_table_op)

    return control_flow_ops.group(update_ops)

  def restrict_rule(self, **kwargs):
    """
    Define the rule to restrict the size of the target
    variable by eliminating k features of least occurrence.

    Args:
      **kwargs: Keyword arguments, including
        residue: int. The leftover number of features after restriction. 
        trigger: int. The trigger threshold to execute the restriction.
          If the number of feature is less than `trigger` in real time,
          nothing happens. The restriction will only be executed when the
          feature number is greater than `trigger`. Default equals to
          `residue`.

    Returns:
      An restriction operation.
    """
    try:
      self.residue = kwargs['residue']
    except:
      raise KeyError('FrequencyRestrictPolicy requires'
                     '`residue` keyword in restrict_rule.')
    if not isinstance(self.residue, int):
      raise TypeError('FrequencyRestrictPolicy requires integer `residue`.')
    if self.residue < 0:
      raise ValueError('FrequencyRestrictPolicy requires'
                       'non-negative `residue` value.')

    trigger = kwargs.get('trigger', self.residue)
    if not isinstance(trigger, int):
      raise TypeError('`trigger` needs to an integer.')

    is_overflow = math_ops.greater(self.var.size(), trigger)
    return control_flow_ops.cond(is_overflow, self._cond_restrict,
                                 self._cond_no_op)

  def _cond_restrict(self):
    var_in_slots = _select_slot_vars(self.var, self.optmz)
    restrict_var_ops, restrict_status_ops, restrict_slot_ops = [], [], []

    for idx, dev in enumerate(self.freq_var.devices):
      with ops.device(dev):
        partial_keys, partial_counts = self.freq_var.tables[idx].export()
        partial_residue = int(self.residue / self.freq_var.shard_num)
        partial_counts = array_ops.reshape(partial_counts, (-1,))
        first_dim = array_ops.shape(partial_counts)[0]

        k_on_top = math_ops.cast(first_dim - partial_residue,
                                 dtype=dtypes.int32)
        k_on_top = math_ops.maximum(k_on_top, 0)
        _, removed_key_indices = nn_ops.top_k(-partial_counts,
                                              k_on_top,
                                              sorted=False)
        removed_keys = array_ops.gather(partial_keys, removed_key_indices)

        restrict_var_ops.append(self.var.tables[idx].remove(removed_keys))
        restrict_status_ops.append(
            self.freq_var.tables[idx].remove(removed_keys))
        for slot_var in var_in_slots:
          restrict_slot_ops.append(slot_var.tables[idx].remove(removed_keys))

    return control_flow_ops.group(restrict_var_ops, restrict_status_ops,
                                  restrict_slot_ops)

  def _cond_no_op(self):
    return control_flow_ops.no_op()


class VariableRestrictor(object):
  """
  Restrictor to constrain the feature number in `dynamic_embedding.Variable`
  objects. The restrictor can keep track on training procedure, which may
  leads to enormous growth of the variables, while new features are constantly
  fed into training. Meanwhile, stateful optimizers also keep these features
  in memory.

  In very large scale training, the feature space may overwhelm the physical
  memory. The `VariableRestrictor` is designed to track the status of the
  variables and optimizers. And It restricts the size of these variables
  and sparse slots in optimizers.

  Use an independent restrictor to enable flexible management on variable and
  optimizer slots, without aggregating the tables in variables and optimizers.

  ### graph mode example:
  ```python
  var = dynamic_embedding.get_variable(...)
  ...
  optmizer = tf.train.AdagradOptimizer(0.1)
  restrictor = de.VariableRestrictor(var_list=[var],
                                     optimizer_list=[optimizer],
                                     policy=de.TimestampRestrictPolicy)

  update_op = restrictor.update()
  restrict_op = restrictor.restrict(residue=20000, trigger=25000)

  with session.Session() as sess:
    for step in range(num_ter):
      sess.run([training_operations, update_op])
      if step % 1000 == 0:
        sess.run(restrict_op)
  ```

  ### eager mode example:
  ```python
  var = dynamic_embedding.get_variable(...)
  optmizer = tf.keras.adam.Adam(0.1)
  restrictor = de.VariableRestrictor(var_list=[var],
                                     optimizer_list=[optimizer],
                                     policy=de.TimestampRestrictPolicy)
  ...

  optimizer.minimize(loss_fn, var_fn)
  restrictor.update(trainable_wrappers=trainables)

  if step % 100 == 0:
    restrictor.restrict(residue=1000, trigger=1200)
  ```
  """

  def __init__(
      self,
      var_list=None,
      optimizer_list=None,
      policy=TimestampRestrictPolicy,
  ):
    """
    Create a restrictor for tracking training procedure, with growth of
    variables and slots in optimizers. The restrictor provides operation
    to update the tracking status, and operation to restrict the variables
    and slots in optimizers.

    Args:
      var_list: List of `dynamic_embedding.Variable` objects.
      optimizer_list: List of optimizers.
      policy: A derived `RestrictPolicy` class which defines the rules for
        updating and restriction.
    """
    if not issubclass(policy, RestrictPolicy):
      raise TypeError('policy must be inherited from `RestrictPolicy`.')

    self.var_list = var_list
    self.optimizer_list = optimizer_list
    self.policy_group = {}

    for var in self.var_list:
      if not self.policy_group.get(var, None):
        self.policy_group[var] = []
      for optmz in self.optimizer_list:
        self.policy_group[var].append(policy(var, optmz))

  def update(self, trainable_wrappers=None, **kwargs):
    """
    Update the restrictor's status.

    Args:
      trainable_wrappers: A list of `TrainableWrapper` objects. The
        status of variable is only updated by trainable_wrappers, to
        avoid consuming unrelated indices when embedding is shared by
        multiple optimizers, and only parts of then work. `None` means
        update all embeddings.
      **kwargs: Optional keyword arguments.

    Returns:
      An operation updates trainable_wrappers.
    """
    update_ops = []
    policies = _flatten(self.policy_group.values())
    for pol in policies:
      update_ops.append(
          pol.update_rule(trainable_wrappers=trainable_wrappers, **kwargs))
    return control_flow_ops.group(update_ops)

  def restrict(self, **kwargs):
    """
    Call restriction for every variable and optimizer slots.

    Args
      **kwargs: keyword arguments passed to `restrict_rule` in policy.

    Returns:
      An operation to restrict variables and slots.
    """
    restrict_ops = []
    policies = _flatten(self.policy_group.values())

    for pol in policies:
      restrict_ops.append(pol.restrict_rule(**kwargs))
    return control_flow_ops.group(restrict_ops)
