#!/bin/bash -eu
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

PIP="$1"
PIP_INSTALL=("${PIP}" "install" "--prefer-binary" --upgrade)

if [[ ! -x "$(which "${PIP}")" ]]; then
  # Python2 doesn't ship with pip by default.
  PYTHON="${PIP/pip/python}"
  wget "https://bootstrap.pypa.io/get-pip.py"
  "${PYTHON}" "get-pip.py"
  rm "get-pip.py"
fi

PACKAGES=(
  "auditwheel"
  "wheel"
  "setuptools"
  "virtualenv"
  "six"
  "future"
  "absl-py"
  "werkzeug"
  "bleach"
  "markdown"
  "protobuf"
  "numpy"
  "scipy"
  "scikit-learn"
  "pandas"
  "psutil"
  "py-cpuinfo"
  "lazy-object-proxy"
  "pylint"
  "pycodestyle"
  "portpicker"
  "grpcio"
  "astor"
  "gast"
  "termcolor"
  "keras_preprocessing"
  "h5py"
  "tf-estimator-nightly"
  "tb-nightly"
  "argparse"
  "dm-tree"
  "dill"
  "tblib"
)

# tf.mock require the following for python2:
if [[ "${PIP}" == *pip2* ]]; then
  PACKAGES+=("mock")
fi

# Get the latest version of pip so it recognize manylinux2010
"${PIP}" "install" "--upgrade" "pip"

"${PIP_INSTALL[@]}" "${PACKAGES[@]}"

