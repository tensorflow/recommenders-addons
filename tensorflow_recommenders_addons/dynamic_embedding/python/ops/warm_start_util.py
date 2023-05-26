# Copyright 2022 The TensorFlow Recommenders-Addons Authors.
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
"""warm-start util"""

import collections
import six
import re

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.util.tf_export import tf_export

from tensorflow_recommenders_addons import dynamic_embedding as de


def _get_de_variables(vars_to_warm_start):
  if isinstance(vars_to_warm_start,
                six.string_types) or vars_to_warm_start is None:
    list_of_vars = ops.get_collection(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES,
                                      scope=vars_to_warm_start)
  elif isinstance(vars_to_warm_start, list):
    if all(isinstance(v, six.string_types) for v in vars_to_warm_start):
      list_of_vars = []
      for v in vars_to_warm_start:
        list_of_vars += ops.get_collection(
            de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES, scope=v)
    elif all(isinstance(v, de.Variable) for v in vars_to_warm_start):
      list_of_vars = vars_to_warm_start
    else:
      raise ValueError("If `vars_to_warm_start` is a list, it must be a "
                       "`de.Variable` or `str`.  Given types are {}".format(
                           type(vars_to_warm_start)))
  else:
    raise ValueError("`vars_to_warm_start must be a `list` or `str`.  Given "
                     "type is {}".format(type(vars_to_warm_start)))

  de_variables = []
  for v in list_of_vars:
    t = [v] if not isinstance(v, list) else v
    de_variables.append(v)

  return de_variables


def warm_start(ckpt_to_initialize_from,
               vars_to_warm_start=".*",
               var_name_to_prev_var_name=None):
  """Warm-starts de.Variable using the given settings.

    Args:
      ckpt_to_initialize_from: [Required] A string specifying the directory with
        checkpoint file(s) or path to checkpoint from which to warm-start the
        model parameters.
      vars_to_warm_start: [Optional] One of the following:
        - A regular expression (string) that captures which variables to
          warm-start (see tf.compat.v1.get_collection).  This expression will only
          consider variables in the TRAINABLE_VARIABLES collection -- if you need
          to warm-start non_TRAINABLE vars (such as optimizer accumulators or
          batch norm statistics), please use the below option.
        - A list of strings, each a regex scope provided to
          tf.compat.v1.get_collection with GLOBAL_VARIABLES (please see
          tf.compat.v1.get_collection).  For backwards compatibility reasons,
          this is separate from the single-string argument type.
        - A list of Variables to warm-start.  If you do not have access to the
          `Variable` objects at the call site, please use the above option.
        - `None`, in which case only TRAINABLE variables specified in
          `var_name_to_vocab_info` will be warm-started.
        Defaults to `'.*'`, which warm-starts all variables in the
        TRAINABLE_VARIABLES collection.  Note that this excludes variables such
        as accumulators and moving statistics from batch norm.

    Raises:
      ValueError: If saveable's spec.name not match pattern 
        defined by de.Variable._make_name.
    """

  def _replace_var_in_spec_name(spec_name, var_name):

    def _replace(m):
      return '{}_mht_{}of{}'.format(var_name, m.groups()[1], m.groups()[2])

    out = re.sub(r'(\w+)_mht_(\d+)of(\d+)', _replace, spec_name)
    if out is None:
      raise ValueError(
          "Invalid sepc name, should match `{}_mht_{}of{}`, given %s" %
          spec_name)
    return out

  logging.info("Warm-starting from: {}".format(ckpt_to_initialize_from))

  de_variables = _get_de_variables(vars_to_warm_start)
  if not var_name_to_prev_var_name:
    var_name_to_prev_var_name = {}

  ckpt_file = checkpoint_utils._get_checkpoint_filename(ckpt_to_initialize_from)
  assign_ops = []
  for variable in de_variables:
    var_name = variable.name
    prev_var_name = var_name_to_prev_var_name.get(var_name)
    if prev_var_name:
      logging.debug("Warm-start variable: {}: prev_var_name: {}".format(
          var_name, prev_var_name or "Unchanged"))
    else:
      prev_var_name = var_name

    try:
      saveables = saveable_object_util.validate_and_slice_inputs([variable])
    except AttributeError:
      saveables_dict = saveable_object_util.op_list_to_dict([variable])
      saveables = saveable_object_util.validate_and_slice_inputs(saveables_dict)
    for saveable in saveables:
      restore_specs = []
      for spec in saveable.specs:
        restore_specs.append((_replace_var_in_spec_name(spec.name,
                                                        prev_var_name),
                              spec.slice_spec, spec.dtype))

      names, slices, dtypes = zip(*restore_specs)
      # Load tensors in cuckoo_hashtable op's device
      with ops.colocate_with(saveable.op._resource_handle.op):
        saveable_tensors = io_ops.restore_v2(ckpt_file, names, slices, dtypes)
        assign_ops.append(saveable.restore(saveable_tensors, None))

  return control_flow_ops.group(assign_ops)


class WarmStartHook(SessionRunHook):
  """Warm-start hook for tf.estimator.Estimator
  """

  def __init__(self,
               ckpt_to_initialize_from,
               vars_to_warm_start,
               var_name_to_prev_var_name=None):
    """Initializes a `WarmStartHook`

    Args:
      ckpt_to_initialize_from: [Required] A string specifying the directory with
        checkpoint file(s) or path to checkpoint from which to warm-start the
        model parameters.
      vars_to_warm_start: [Optional] One of the following:
        - A regular expression (string) that captures which variables to
          warm-start (see tf.compat.v1.get_collection).  This expression will only
          consider variables in the TRAINABLE_VARIABLES collection -- if you need
          to warm-start non_TRAINABLE vars (such as optimizer accumulators or
          batch norm statistics), please use the below option.
        - A list of strings, each a regex scope provided to
          tf.compat.v1.get_collection with GLOBAL_VARIABLES (please see
          tf.compat.v1.get_collection).  For backwards compatibility reasons,
          this is separate from the single-string argument type.
        - A list of Variables to warm-start.  If you do not have access to the
          `Variable` objects at the call site, please use the above option.
        - `None`, in which case only TRAINABLE variables specified in
          `var_name_to_vocab_info` will be warm-started.
        Defaults to `'.*'`, which warm-starts all variables in the
        TRAINABLE_VARIABLES collection.  Note that this excludes variables such
        as accumulators and moving statistics from batch norm.

    Raises:
      ValueError: If saveable's spec.name not match pattern 
        defined by de.Variable._make_name.
    """
    self._ckpt_to_initialize_from = ckpt_to_initialize_from
    self._vars_to_warm_start = vars_to_warm_start
    self._var_name_to_prev_var_name = var_name_to_prev_var_name

  def begin(self):
    self._restore_op = warm_start(
        ckpt_to_initialize_from=self._ckpt_to_initialize_from,
        vars_to_warm_start=self._vars_to_warm_start,
        var_name_to_prev_var_name=self._var_name_to_prev_var_name)

  def after_create_session(self, session, coord):
    session.run(self._restore_op)
