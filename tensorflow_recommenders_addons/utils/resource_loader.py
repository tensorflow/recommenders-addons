# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities similar to tf.python.platform.resource_loader."""

import os
import pkg_resources
import tensorflow as tf
import warnings

abi_warning_already_raised = False
SKIP_CUSTOM_OPS = False


def get_required_tf_version():
  try:
    pkg = pkg_resources.get_distribution("tensorflow-recommenders-addons")
  except pkg_resources.DistributionNotFound:
    try:
      pkg = pkg_resources.get_distribution("tensorflow-recommenders-addons-gpu")
    except pkg_resources.DistributionNotFound:
      # Force return for 'Test with bazel' on CI.
      warnings.warn(
          "Fail to get TFRA package information, if you are running on "
          "bazel test mode, please ignore this warning, \nor you should check "
          "TFRA installation.",
          UserWarning,
      )
      return tf.__version__

  pkg_info = pkg.requires()
  for x in pkg_info:
    if x.name in ["tensorflow", "tensorflow-gpu"]:
      return x.specs[0][1]
  assert False, "Fail to get required TensorFlow version of TFRA!"


def get_devices(device_type="GPU"):
  if hasattr(tf.config, "list_physical_devices"):
    return tf.config.list_physical_devices(device_type)
  elif hasattr(tf.config, "experimental_list_devices"):
    # for compatible with TensorFlow1.x
    devices_list = tf.config.experimental_list_devices()
    return [d for d in devices_list if ":{}".format(device_type.upper()) in d]
  else:
    warnings.warn(
        "You are currently using TensorFlow {} which TFRA cann't get the devices correctly.\n"
        "So we strongly recommend that you use the version supported by the TFRA statement "
        "To do that, refer to the readme: "
        "https://github.com/tensorflow/recommenders-addons"
        "".format(tf.__version__,),
        UserWarning,
    )
    return []


def get_project_root():
  """Returns project root folder."""
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path_to_datafile(path):
  """Get the path to the specified file in the data dependencies.

    The path is relative to tensorflow_recommenders_addons/

    Args:
      path: a string resource path relative to tensorflow_recommenders_addons/
    Returns:
      The path to the specified data file
    """
  root_dir = get_project_root()
  return os.path.join(root_dir, path.replace("/", os.sep))


class LazySO:

  def __init__(self, relative_path):
    self.relative_path = relative_path
    self._ops = None

  @property
  def ops(self):
    if SKIP_CUSTOM_OPS:
      import pytest

      pytest.skip("Skipping the test because a custom ops "
                  "was being loaded while --skip-custom-ops was set.")
    if self._ops is None:
      self.display_warning_if_incompatible()
      self._ops = tf.load_op_library(get_path_to_datafile(self.relative_path))
    return self._ops

  def display_warning_if_incompatible(self):
    global abi_warning_already_raised
    if abi_is_compatible() or abi_warning_already_raised:
      return
    required_tf_version = get_required_tf_version()
    warnings.warn(
        "You are currently using TensorFlow {} and trying to load a custom op ({})."
        "\n"
        "TFRA has compiled its custom ops against TensorFlow {}, "
        "and there are no compatibility guarantees between the two versions. "
        "\n"
        "This means that you might get 'Symbol not found' when loading the custom op, "
        "or other kind of low-level errors.\n If you do, do not file an issue "
        "on Github. This is a known limitation."
        "\n\n"
        "You can also change the TensorFlow version installed on your system. "
        "You would need a TensorFlow version equal to {}. \n"
        "Note that nightly versions of TensorFlow, "
        "as well as non-pip TensorFlow like `conda install tensorflow` or compiled "
        "from source are not supported."
        "\n\n"
        "The last solution is to compile the TensorFlow Recommenders-Addons "
        "with the TensorFlow installed on your system. "
        "To do that, refer to the readme: "
        "https://github.com/tensorflow/recommenders-addons"
        "".format(
            tf.__version__,
            self.relative_path,
            required_tf_version,
            required_tf_version,
        ),
        UserWarning,
    )
    abi_warning_already_raised = True


def abi_is_compatible():
  if "dev" in tf.__version__:
    return False

  required_tf_version = get_required_tf_version()
  return tf.__version__ == required_tf_version


def prefix_op_name(op_name):
  """
  In order to keep compatibility of existing models,
  we cannot change the OP naming rule directly by replacing "TFRA>" to "Tfra",
  So we had to add prefix to OP name according to the TF verison.

  Args:
    op_name: original OP name
  Returns:
    OP name with prefix
  """
  major_tf_version = int(tf.__version__.split(".")[0])
  _prefix = "TFRA>" if major_tf_version >= 2 else "Tfra"
  return _prefix + op_name


def get_tf_version_triple():
  tf_version_triple = tf.__version__.split(".")
  return tf_version_triple
