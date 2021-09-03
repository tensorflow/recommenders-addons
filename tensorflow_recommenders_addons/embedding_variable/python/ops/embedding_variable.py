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
# See the License for the specific language governing permissions and # limitations under the License.
# ==============================================================================
"""embedding variables interface"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import sys
import threading
import traceback

import six
from six import iteritems
from six.moves import xrange, zip  # pylint: disable=redefined-builtin

from tensorflow_recommenders_addons import embedding_variable as ev
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


class _PartitionInfo(object):
  """Holds partition info used by initializer functions."""

  __slots__ = ["_full_shape", "_var_offset"]

  def __init__(self, full_shape, var_offset):
    """Constructor.

    Args:
      full_shape: Tuple or list of `int` indicating the full combined shape of
        the partitioned variables.
      var_offset: Tuple or list of `int` specifying offset of this partition
        with respect to the full variable for each dimension.

    Raises:
      TypeError: If `full_shape` or `var_offset` is not a sequence.
      ValueError: If `full_shape` or `var_offset` differ in length. If
        `var_offset` exceeds `full_shape` in any dimension.
    """
    if not isinstance(full_shape, collections_abc.Sequence) or isinstance(
        full_shape, six.string_types):
      raise TypeError(
          "`full_shape` must be a sequence (like tuple or list) instead of " +
          type(full_shape).__name__)

    if not isinstance(var_offset, collections_abc.Sequence) or isinstance(
        var_offset, six.string_types):
      raise TypeError(
          "`var_offset` must be a sequence (like tuple or list) instead of " +
          type(var_offset).__name__)

    if len(var_offset) != len(full_shape):
      raise ValueError(
          "Expected equal length, but `var_offset` is of length {} while "
          "full_shape is of length {}.".format(len(var_offset),
                                               len(full_shape)))

    for offset, shape in zip(var_offset, full_shape):
      if offset < 0 or offset >= shape:
        raise ValueError(
            "Expected 0 <= offset < shape but found offset={}, shape={} for "
            "var_offset={}, full_shape={}".format(offset, shape, var_offset,
                                                  full_shape))

    self._full_shape = full_shape
    self._var_offset = var_offset

  @property
  def full_shape(self):
    return self._full_shape

  @property
  def var_offset(self):
    return self._var_offset

  def single_offset(self, shape):
    """Returns the offset when the variable is partitioned in at most one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the offset in the dimension along which the variable is
       partitioned. Returns 0 if the variable is not being partitioned.

    Raises:
      ValueError: Depending on self.single_slice_dim().
    """

    single_slice_dim = self.single_slice_dim(shape)
    # If this variable is not being partitioned at all, single_slice_dim() could
    # return None.
    if single_slice_dim is None:
      return 0
    return self.var_offset[single_slice_dim]

  def single_slice_dim(self, shape):
    """Returns the slice dim when the variable is partitioned only in one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the dimension that the variable is partitioned in, or
      `None` if the variable doesn't seem to be partitioned at all.

    Raises:
      TypeError: If `shape` is not a sequence.
      ValueError: If `shape` is not the same length as `self.full_shape`. If
        the variable is partitioned in more than one dimension.
    """
    if not isinstance(shape, collections_abc.Sequence) or isinstance(
        shape, six.string_types):
      raise TypeError(
          "`shape` must be a sequence (like tuple or list) instead of " +
          type(shape).__name__)

    if len(shape) != len(self.full_shape):
      raise ValueError(
          "Expected equal length, but received shape={} of length {} while "
          "self.full_shape={} is of length {}.".format(shape, len(shape),
                                                       self.full_shape,
                                                       len(self.full_shape)))

    for i in xrange(len(shape)):
      if self.var_offset[i] + shape[i] > self.full_shape[i]:
        raise ValueError(
            "With self.var_offset={}, a partition of shape={} would exceed "
            "self.full_shape={} in dimension {}.".format(
                self.var_offset, shape, self.full_shape, i))

    slice_dim = None
    for i in xrange(len(shape)):
      if shape[i] == self.full_shape[i]:
        continue
      if slice_dim is not None:
        raise ValueError(
            "Cannot use single_slice_dim() with shape={} and "
            "self.full_shape={} since slice dim could be either dimension {} "
            "or {}.".format(shape, self.full_shape, i, slice_dim))
      slice_dim = i

    return slice_dim


def _call_partitioner(partitioner, shape, dtype):
  """Call partitioner validating its inputs/output.

  Args:
    partitioner: a function mapping `Tensor` shape and dtype to a list of
      partitions.
    shape: shape of the `Tensor` to partition, must have at least two
      dimensions.
    dtype: dtype of the elements in the `Tensor`.

  Returns:
    A list with elements >=1 and exactly one >1. The index of that
    element corresponds to the partitioning axis.
  """
  if not shape.is_fully_defined():
    raise ValueError("Shape of a new partitioned variable must be "
                     "fully defined, but instead was %s." % (shape,))
  if shape.ndims < 1:
    raise ValueError("A partitioned Variable must have rank at least 1, "
                     "shape: %s" % shape)

  slicing = partitioner(shape=shape, dtype=dtype)
  if not isinstance(slicing, collections_abc.Sequence):
    raise ValueError("Partitioner must return a sequence, but saw: %s" %
                     slicing)
  if len(slicing) != shape.ndims:
    raise ValueError(
        "Partitioner returned a partition list that does not match the "
        "Variable's rank: %s vs. %s" % (slicing, shape))
  if any(p < 1 for p in slicing):
    raise ValueError("Partitioner returned zero partitions for some axes: %s" %
                     slicing)
  if sum(p > 1 for p in slicing) > 1:
    raise ValueError("Can only slice a variable along one dimension: "
                     "shape: %s, partitioning: %s" % (shape, slicing))
  return slicing


# TODO(slebedev): could be inlined, but
# `_VariableStore._get_partitioned_variable` is too complex even
# without this logic.
def _get_slice_dim_and_num_slices(slicing):
  """Get slicing dimension and number of slices from the partitioner output."""
  for slice_dim, num_slices in enumerate(slicing):
    if num_slices > 1:
      break
  else:
    # Degenerate case: no partitioning applied.
    slice_dim = 0
    num_slices = 1
  return slice_dim, num_slices


def _iter_slices(full_shape, num_slices, slice_dim):
  """Slices a given a shape along the specified dimension."""
  num_slices_with_excess = full_shape[slice_dim] % num_slices
  offset = [0] * len(full_shape)
  min_slice_len = full_shape[slice_dim] // num_slices
  for i in xrange(num_slices):
    shape = full_shape[:]
    shape[slice_dim] = min_slice_len + bool(i < num_slices_with_excess)
    yield offset[:], shape
    offset[slice_dim] += shape[slice_dim]


class _ReuseMode(enum.Enum):
  AUTO_REUSE = 1


AUTO_REUSE = _ReuseMode.AUTO_REUSE

VariableSynchronization = variables.VariableSynchronization  # pylint: disable=invalid-name
VariableAggregation = variables.VariableAggregation  # pylint: disable=invalid-name


class _EmbeddingVariableStore(object):
  """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """

  __slots__ = ["_var_store"]

  def __init__(self, var_store):
    """Create a embdding variable store.
        reuse original var_store """
    self._var_store = var_store

  def get_variable(
      self,
      name,
      shape=None,  # embedding_dim
      dtype=dtypes.float32,
      ktype=dtypes.int64,
      initializer=None,
      regularizer=None,
      reuse=None,
      trainable=None,
      collections=None,
      caching_device=None,
      partitioner=None,
      validate_shape=True,
      constraint=None,
      synchronization=VariableSynchronization.AUTO,
      aggregation=VariableAggregation.NONE):
    """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If a partitioner is provided, a `PartitionedVariable` is returned.
    Accessing this object as a `Tensor` returns the shards concatenated along
    the partition axis.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: Initializer for the variable.
      regularizer: A (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables. When eager execution is enabled  this argument is always
        forced to be False.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`
        defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys to add the `Variable` to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the `Variable` reside, to
        deduplicate copying through `Switch` and other conditional statements.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates
        instead an experimental ResourceVariable which has well-defined
        semantics. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be true.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. The signature
        of `custom_getter` should match that of this method,
        but the most future-proof version will allow for changes: `def
          custom_getter(getter, *args, **kwargs)`.  Direct access to
        all `get_variable` parameters is also allowed: `def
          custom_getter(getter, name, *args, **kwargs)`.  A simple identity
        custom getter that simply creates variables with modified names is:
          ```python
        def custom_getter(getter, name, *args, **kwargs): return getter(name +
          '_suffix', *args, **kwargs) ```
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

    Returns:
      The created or existing `Variable` (or `PartitionedVariable`, if a
      partitioner was used).

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
      RuntimeError: when eager execution is enabled and not called from an
        EagerVariableStore.
    """

    # Note that it's fine to reuse eager variables whose initialization was
    # lifted from a function-building graph into the eager context (that's why
    # the following clause is not wrapped in an `init_scope`); lifted variables
    # are tracked by the graph's `VariableStore`.
    if context.executing_eagerly():
      if not self._var_store._store_eager_variables and reuse:
        raise RuntimeError(
            "When eager execution is enabled variable reuse is only supported"
            " when an EagerVariableStore is active. See the documentation on"
            " EagerVariableStore for example usage.")
      if self._var_store._store_eager_variables:
        reuse = AUTO_REUSE

    # If a *_ref type is passed in an error would be triggered further down the
    # stack. We prevent this using base_dtype to get a non-ref version of the
    # type, before doing anything else. When _ref types are removed in favor of
    # resources, this line can be removed.
    try:
      dtype = dtype.base_dtype
      ktype = ktype.base_dtype
    except AttributeError:
      # .base_dtype not existing means that we will try and use the raw dtype
      # which was passed in - this might be a NumPy type which is valid.
      pass

    # This is the main logic of get_variable.  However, custom_getter
    # may override this logic.  So we save it as a callable and pass
    # it to custom_getter.
    # Note: the parameters of _true_getter, and their documentation, match
    # *exactly* item-for-item with the docstring of this method.
    def _true_getter(  # pylint: disable=missing-docstring
        var_store,
        name,
        shape=None,
        dtype=dtypes.float32,
        ktype=dtypes.int64,
        initializer=None,
        regularizer=None,
        reuse=None,
        trainable=None,
        collections=None,
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        constraint=None,
        synchronization=VariableSynchronization.AUTO,
        aggregation=VariableAggregation.NONE):
      is_scalar = (shape is not None
                   and isinstance(shape, collections_abc.Sequence)
                   and not shape)
      # Partitioned variable case
      if partitioner is not None and not is_scalar:
        if not callable(partitioner):
          raise ValueError("Partitioner must be callable, but received: %s" %
                           partitioner)
        with ops.name_scope(None):
          return self._get_partitioned_variable(var_store=var_store,
                                                name=name,
                                                shape=shape,
                                                dtype=dtype,
                                                ktype=ktype,
                                                initializer=initializer,
                                                regularizer=regularizer,
                                                reuse=reuse,
                                                trainable=trainable,
                                                collections=collections,
                                                caching_device=caching_device,
                                                partitioner=partitioner,
                                                validate_shape=validate_shape,
                                                constraint=constraint,
                                                synchronization=synchronization,
                                                aggregation=aggregation)

      # Special case for partitioned variable to allow reuse without having to
      # specify partitioner.
      if (reuse is True and partitioner is None
          and name in self._var_store._partitioned_vars):
        return self._get_partitioned_variable(var_store=var_store,
                                              name=name,
                                              shape=shape,
                                              dtype=dtype,
                                              ktype=dtypes.int64,
                                              initializer=initializer,
                                              regularizer=regularizer,
                                              reuse=reuse,
                                              trainable=trainable,
                                              collections=collections,
                                              caching_device=caching_device,
                                              partitioner=None,
                                              validate_shape=validate_shape,
                                              constraint=constraint,
                                              synchronization=synchronization,
                                              aggregation=aggregation)

      # Single variable case
      if "%s/part_0" % name in var_store._vars:
        raise ValueError(
            "No partitioner was provided, but a partitioned version of the "
            "variable was found: %s/part_0. Perhaps a variable of the same "
            "name was already created with partitioning?" % name)

      return self._get_single_variable(var_store=var_store,
                                       name=name,
                                       shape=shape,
                                       dtype=dtype,
                                       ktype=ktype,
                                       initializer=initializer,
                                       regularizer=regularizer,
                                       reuse=reuse,
                                       trainable=trainable,
                                       collections=collections,
                                       caching_device=caching_device,
                                       validate_shape=validate_shape,
                                       constraint=constraint,
                                       synchronization=synchronization,
                                       aggregation=aggregation)

    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            synchronization, aggregation, trainable, name))

    return _true_getter(self._var_store,
                        name,
                        shape=shape,
                        dtype=dtype,
                        ktype=ktype,
                        initializer=initializer,
                        regularizer=regularizer,
                        reuse=reuse,
                        trainable=trainable,
                        collections=collections,
                        caching_device=caching_device,
                        partitioner=partitioner,
                        validate_shape=validate_shape,
                        constraint=constraint,
                        synchronization=synchronization,
                        aggregation=aggregation)

  def _get_partitioned_variable(self,
                                var_store,
                                name,
                                partitioner,
                                shape=None,
                                dtype=dtypes.float32,
                                ktype=dtypes.int64,
                                initializer=None,
                                regularizer=None,
                                reuse=None,
                                trainable=None,
                                collections=None,
                                caching_device=None,
                                validate_shape=True,
                                constraint=None,
                                synchronization=VariableSynchronization.AUTO,
                                aggregation=VariableAggregation.NONE):
    """Gets or creates a sharded variable list with these parameters.

    The `partitioner` must be a callable that accepts a fully defined
    `TensorShape` and returns a sequence of integers (the `partitions`).
    These integers describe how to partition the given sharded `Variable`
    along the given dimension.  That is, `partitions[1] = 3` means split
    the `Variable` into 3 shards along dimension 1.  Currently, sharding along
    only one axis is supported.

    If the list of variables with the given name (prefix) is already stored,
    we return the stored variables. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If the initializer is a callable, then it will be called for each
    shard.  Otherwise the initializer should match the shape of the entire
    sharded Variable, and it will be sliced accordingly for each shard.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: the name of the new or existing sharded variable.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and `dtype` of the Variable to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      shape: shape of the new or existing sharded variable.
      dtype: type of the new or existing sharded variable (defaults to
        `DT_FLOAT`).
      initializer: initializer for the sharded variable.
      regularizer: a (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      collections: List of graph collections keys to add the Variable to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
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

    Returns:
      A `PartitionedVariable` object.

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        when violating reuse during variable creation, or if an existing
        sharded variable exists for the given name but with different sharding.
    """
    initializing_from_value = initializer is not None and isinstance(
        initializer, ops.Tensor)
    if name in var_store._vars:
      raise ValueError(
          "A partitioner was provided, but an unpartitioned version of the "
          "variable was found: %s.  Perhaps a variable of the same name was "
          "already created without partitioning?" % name)

    shape = tensor_shape.as_shape(shape)
    if initializing_from_value:
      shape = shape.merge_with(initializer.get_shape())

    shape_t = tensor_shape.as_shape([sys.maxsize]).concatenate(shape)
    fd_partition_num = partitioner(shape=shape_t, dtype=dtype)[0]
    shape = tensor_shape.as_shape([fd_partition_num]).concatenate(shape)
    partitions = None
    if not reuse or partitioner:
      partitions = _call_partitioner(partitioner, shape, dtype)

    if name in var_store._partitioned_vars:
      if reuse is False:
        raise ValueError(
            "Partitioned variable with name %s already exists. Did you mean to "
            "set reuse=True or reuse=tf.AUTO_REUSE in VarScope?" % name)

      existing_var = var_store._partitioned_vars[name]
      if not shape.is_compatible_with(existing_var.get_shape()):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified shape %s "
            "and found shape %s." % (name, shape, existing_var.get_shape()))
      if not dtype.is_compatible_with(existing_var.dtype):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified dtype %s "
            "and found dtype %s." % (name, dtype.name, existing_var.dtype.name))

      # pylint: disable=protected-access
      if (partitions is not None
          and existing_var._get_partitions() != partitions):
        raise ValueError(
            "Trying to reuse partitioned variable %s, but specified partitions "
            "%s and found partitions %s." %
            (name, partitions, existing_var._get_partitions()))
      # pylint: enable=protected-access

      return existing_var

    if reuse is True:
      raise ValueError("PartitionedVariable %s does not exist, or was not "
                       "created with tf.get_variable(). Did you mean to set "
                       "reuse=False or reuse=tf.AUTO_REUSE in VarScope?" % name)

    slice_dim, num_slices = _get_slice_dim_and_num_slices(partitions)

    if "%s/part_0" % name in var_store._vars:
      if "%s/part_%d" % (name, num_slices - 1) not in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but %s/part_%d was not." %
            (num_slices, name, name, num_slices - 1))
      if "%s/part_%d" % (name, num_slices) in self._vars:
        raise ValueError(
            "Partitioner returned a different partitioning than what was "
            "already found.  Partitioner returned %d shards, and shard "
            "%s/part_0 was found, but so was the extra shard %s/part_%d." %
            (num_slices, name, name, num_slices))

    vs = []
    for i, (var_offset, var_shape) in enumerate(
        _iter_slices(shape.as_list(), num_slices, slice_dim)):
      partition_info = _PartitionInfo(full_shape=shape.as_list(),
                                      var_offset=var_offset)
      var_full_name = "%s/part_%d" % (name, i)
      with ops.name_scope(var_full_name + "/PartitionedInitializer",
                          skip_on_eager=False):
        # Create the tensor to initialize the variable with default value.
        if initializer is None:
          init, initializing_from_value = self._get_default_initializer(
              name=name, shape=shape, dtype=dtype)
          if initializing_from_value:
            init_shape = None
          else:
            init_shape = var_shape
        elif callable(initializer):
          init = initializer
          init_shape = var_shape
        elif isinstance(initializer, ops.Tensor):
          init = array_ops.slice(initializer, var_offset, var_shape)
          # Use the dtype of the given tensor.
          dtype = init.dtype.base_dtype
          init_shape = None
        else:
          init = ops.convert_to_tensor(initializer, dtype=dtype)
          init = array_ops.slice(init, var_offset, var_shape)
          init_shape = None
      init_shape = shape.as_list()[1:]
      with ops.name_scope(None):
        var = self._get_single_variable(name=var_full_name,
                                        var_store=var_store,
                                        shape=init_shape,
                                        dtype=dtype,
                                        ktype=ktype,
                                        initializer=init,
                                        partition_info=partition_info,
                                        regularizer=regularizer,
                                        reuse=reuse,
                                        trainable=trainable,
                                        collections=collections,
                                        caching_device=caching_device,
                                        validate_shape=validate_shape,
                                        constraint=constraint,
                                        synchronization=synchronization,
                                        aggregation=aggregation)

      # pylint: disable=protected-access
      var._set_save_slice_info(
          variables.Variable.SaveSliceInfo(name, shape.as_list(), var_offset,
                                           var_shape))
      vs.append(var)
      # pylint: enable=protected-access

    partitioned_var = variables.PartitionedVariable(name=name,
                                                    shape=shape,
                                                    dtype=dtype,
                                                    variable_list=vs,
                                                    partitions=partitions)
    if not context.executing_eagerly() or var_store._store_eager_variables:
      var_store._partitioned_vars[name] = partitioned_var
    return partitioned_var

  def _get_single_variable(self,
                           var_store,
                           name,
                           shape=None,
                           dtype=dtypes.float32,
                           ktype=dtypes.int64,
                           initializer=None,
                           regularizer=None,
                           partition_info=None,
                           reuse=None,
                           trainable=None,
                           collections=None,
                           caching_device=None,
                           validate_shape=True,
                           constraint=None,
                           synchronization=VariableSynchronization.AUTO,
                           aggregation=VariableAggregation.NONE):
    """Get or create a single Variable (e.g.

    a shard or entire variable).

    See the documentation of get_variable above (ignore partitioning components)
    for details.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.
      initializer: see get_variable.
      regularizer: see get_variable.
      partition_info: _PartitionInfo object.
      reuse: see get_variable.
      trainable: see get_variable.
      collections: see get_variable.
      caching_device: see get_variable.
      validate_shape: see get_variable.
      constraint: see get_variable.
      synchronization: see get_variable.
      aggregation: see get_variable.

    Returns:
      A Variable.  See documentation of get_variable above.

    Raises:
      ValueError: See documentation of get_variable above.
    """
    # Set to true if initializer is a constant.
    initializing_from_value = False
    if initializer is not None and not callable(initializer):
      initializing_from_value = True
    if shape is not None and initializing_from_value:
      raise ValueError("If initializer is a constant, do not specify shape.")

    dtype = dtypes.as_dtype(dtype)
    shape = tensor_shape.as_shape(shape)

    if name in var_store._vars:
      # Here we handle the case when returning an existing variable.
      if reuse is False:
        var = var_store._vars[name]
        err_msg = ("Variable %s already exists, disallowed."
                   " Did you mean to set reuse=True or "
                   "reuse=tf.AUTO_REUSE in VarScope?" % name)
        # ResourceVariables don't have an op associated with so no traceback
        if isinstance(var, resource_variable_ops.ResourceVariable):
          raise ValueError(err_msg)
        tb = var.op.traceback[::-1]
        # Throw away internal tf entries and only take a few lines. In some
        # cases the traceback can be longer (e.g. if someone uses factory
        # functions to create variables) so we take more than needed in the
        # default case.
        tb = [x for x in tb if "tensorflow/python" not in x[0]][:5]
        raise ValueError("%s Originally defined at:\n\n%s" %
                         (err_msg, "".join(traceback.format_list(tb))))
      found_var = var_store._vars[name]
      if not shape.is_compatible_with(found_var.get_shape()):
        raise ValueError("Trying to share variable %s, but specified shape %s"
                         " and found shape %s." %
                         (name, shape, found_var.get_shape()))
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError("Trying to share variable %s, but specified dtype %s"
                         " and found dtype %s." %
                         (name, dtype_str, found_type_str))
      return found_var

    # The code below handles only the case of creating a new variable.
    if reuse is True:
      raise ValueError("Variable %s does not exist, or was not created with "
                       "tf.get_variable(). Did you mean to set "
                       "reuse=tf.AUTO_REUSE in VarScope?" % name)

    # Create the tensor to initialize the variable with default value.
    if initializer is None:
      initializer, initializing_from_value = self._get_default_initializer(
          name=name, shape=shape, dtype=dtype)
    # Enter an init scope when creating the initializer.
    with ops.init_scope():
      if initializing_from_value:
        init_val = initializer
        variable_dtype = None
      else:
        # Instantiate initializer if provided initializer is a type object.
        if tf_inspect.isclass(initializer):
          initializer = initializer()
        if shape.is_fully_defined():
          if "partition_info" in tf_inspect.getargspec(initializer).args:
            init_val = functools.partial(initializer,
                                         shape.as_list(),
                                         dtype=dtype,
                                         partition_info=partition_info)
          else:
            init_val = functools.partial(initializer,
                                         shape.as_list(),
                                         dtype=dtype)
          variable_dtype = dtype.base_dtype
        elif _needs_no_arguments(initializer):
          init_val = initializer
          variable_dtype = None
        else:
          raise ValueError("The initializer passed is not valid. It should "
                           "be a callable with no arguments and the "
                           "shape should not be provided or an instance of "
                           "`tf.keras.initializers.*' and `shape` should be "
                           "fully defined.")

    # Create the variable.
    v = ev.EmbeddingVariable(
        embedding_dim=shape,
        initializer=initializer,
        trainable=trainable,
        collections=collections,
        name=name,
        ktype=ktype,
        vtype=dtype,
        #variable_def=None,
        #import_scope=None,
        #distribute_strategy=None,
        caching_device=caching_device,
        #validate_shape=validate_shape,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation)
    if context.executing_eagerly() and var_store._store_eager_variables:
      if collections:
        ops.add_to_collections(collections, v)
      else:
        ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, v)
      if trainable:
        ops.add_to_collection(ops.GraphKeys.TRAINABLE_VARIABLES, v)

    if not context.executing_eagerly() or var_store._store_eager_variables:
      # In eager mode we do not want to keep default references to Variable
      # objects as this will prevent their memory from being released.
      var_store._vars[name] = v
    logging.vlog(1, "Created variable %s with shape %s and init %s", v.name,
                 format(shape), initializer)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:

      def make_regularizer_op():
        with ops.colocate_with(v):
          with ops.name_scope(name + "/Regularizer/"):
            return regularizer(v)

      if regularizer(v) is not None:
        lazy_eval_tensor = _LazyEvalTensor(make_regularizer_op)
        ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES,
                              lazy_eval_tensor)

    return v

  # Initialize variable when no initializer provided
  def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
    """Provide a default initializer and a corresponding value.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.

    Returns:
      initializer and initializing_from_value. See get_variable above.

    Raises:
      ValueError: When giving unsupported dtype.
    """
    del shape
    # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
    if dtype.is_floating:
      initializer = init_ops.glorot_uniform_initializer()
      initializing_from_value = False
    # If dtype is DT_INT/DT_UINT, provide a default value `zero`
    # If dtype is DT_BOOL, provide a default value `FALSE`
    elif (dtype.is_integer or dtype.is_unsigned or dtype.is_bool
          or dtype == dtypes.string):
      initializer = init_ops.zeros_initializer()
      initializing_from_value = False
    # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
    else:
      raise ValueError("An initializer for variable %s of %s is required" %
                       (name, dtype.base_dtype))

    return initializer, initializing_from_value


@tf_export("embedding_variable.get_variable")
def get_variable(
    name,  # unique,
    embedding_dim,
    key_dtype=dtypes.int64,
    value_dtype=dtypes.float32,
    initializer=None,
    regularizer=None,
    reuse=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    constraint=None):
  if key_dtype == dtypes.int64 or key_dtype == dtypes.int32:
    invalid_key = -1
  else:
    raise ValueError("Not support key_dtype: %s, only support int64/int32" %
                     key_dtype)
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer()

  scope = variable_scope.get_variable_scope()
  scope_store = variable_scope._get_default_variable_store()

  if regularizer is None:
    regularizer = scope._regularizer
  if caching_device is None:
    caching_device = scope._caching_device
  if partitioner is None:
    partitioner = scope._partitioner
  if not context.executing_eagerly():
    if reuse is None:
      reuse = scope._reuse
  else:
    reuse = AUTO_REUSE

  full_name = scope.name + "/" + name if scope.name else name
  # Variable names only depend on variable_scope (full_name here),
  # not name_scope, so we reset it below for the time of variable creation.
  with ops.name_scope(None):
    dtype = value_dtype
    # Check that `initializer` dtype and `dtype` are consistent before
    # replacing them with defaults.
    if (dtype is not None and initializer is not None
        and not callable(initializer)):
      init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
      if init_dtype != dtype:
        raise ValueError("Initializer type '%s' and explicit dtype '%s' "
                         "don't match." % (init_dtype, dtype))
    if initializer is None:
      initializer = scope._initializer
    if constraint is None:
      constraint = scope._constraint
    if dtype is None:
      dtype = scope._dtype
    if invalid_key is None:
      invalid_key = -1
    ev_store = _EmbeddingVariableStore(scope_store)
    return ev_store.get_variable(full_name,
                                 shape=embedding_dim,
                                 dtype=value_dtype,
                                 ktype=key_dtype,
                                 initializer=initializer,
                                 regularizer=regularizer,
                                 reuse=reuse,
                                 trainable=trainable,
                                 collections=collections,
                                 caching_device=caching_device,
                                 partitioner=partitioner,
                                 validate_shape=True,
                                 constraint=None)
