# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
import glob
import os

from typedapi import ensure_api_is_typed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def test_api_typed():
  modules_list = []
  # Files within this list will be exempt from verification.
  exception_list = []
  help_message = (
      "You can also take a look at the section about it in the CONTRIBUTING.md:\n"
      "https://github.com/tensorflow/recommenders-addons/blob/master/CONTRIBUTING.md#about-type-hints"
  )
  ensure_api_is_typed(modules_list,
                      exception_list,
                      init_only=True,
                      additional_message=help_message)


def test_case_insensitive_filesystems():
  # Make sure BASE_DIR is project root.
  # If it doesn't, we probably computed the wrong directory.
  if not os.path.isdir(os.path.join(BASE_DIR,
                                    "tensorflow_recommenders_addons")):
    raise AssertionError("BASE_DIR = {} is not project root".format(BASE_DIR))

  for dirpath, dirnames, filenames in os.walk(BASE_DIR, followlinks=True):
    lowercase_directories = [x.lower() for x in dirnames]
    lowercase_files = [x.lower() for x in filenames]

    lowercase_dir_contents = lowercase_directories + lowercase_files
    if len(lowercase_dir_contents) != len(set(lowercase_dir_contents)):
      raise AssertionError("Files with same name but different case detected "
                           "in directory: {}".format(dirpath))


def get_lines_of_source_code(allowlist=None):
  allowlist = allowlist or []
  source_dir = os.path.join(BASE_DIR, "tensorflow_recommenders_addons")
  for path in glob.glob(source_dir + "/**/*.py", recursive=True):
    if in_allowlist(path, allowlist):
      continue
    with open(path) as f:
      for line_idx, line in enumerate(f):
        yield path, line_idx, line


def in_allowlist(file_path, allowlist):
  for allowed_file in allowlist:
    if file_path.endswith(allowed_file):
      return True
  return False


def test_no_tf_control_dependencies():
  # TODO: remove all elements of the list and remove the allowlist
  # This allowlist should not grow. Do not add elements to this list.
  allowlist = [
      # "tensorflow_recommenders_addons/**/*",
  ]
  for file_path, line_idx, line in get_lines_of_source_code(allowlist):

    if "tf.control_dependencies(" in line:

      raise NameError(
          "The usage of a tf.control_dependencies() function call was found in "
          "file {} at line {}:\n\n"
          "   {}\n"
          "In TensorFlow 2.x, in a function decorated "
          "with `@tf.function` the dependencies are controlled automatically"
          " thanks to Autograph. \n"
          "TensorFlow Recommenders Addons aims to be written with idiomatic TF 2.x code. \n"
          "As such, using tf.control_dependencies() is not allowed in the codebase. \n"
          "Decorate your function with @tf.function instead. \n"
          "You can take a look at \n"
          "https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md#program-order-semantics--control-dependencies"
          "".format(file_path, line_idx, line))
