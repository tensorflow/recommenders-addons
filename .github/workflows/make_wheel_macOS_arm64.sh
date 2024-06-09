#!/bin/bash
#
# Making wheel for macOS arm64 architecture
set -e -x

export TF_NEED_CUDA=0
export IGNORE_HKV="--ignore=./tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/hkv_hashtable_ops_test.py"
export IGNORE_REDIS="--ignore=./tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/redis_table_ops_test.py"
export IGNORE_REDIS_VAR="--ignore=./tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/redis_table_variable_test.py"
export USE_BAZEL_VERSION='5.1.1'

# For TensorFlow version 2.12 or earlier:
export PROTOBUF_VERSION=3.19.6
export TF_NAME="tensorflow-macos"

python --version
python -m pip install --default-timeout=1000 delocate==0.10.3 wheel setuptools
# For TensorFlow version 2.13 or later:
if [[ "$TF_VERSION" =~ ^2\.1[3-9]\.[0-9]$ ]] ; then
  export PROTOBUF_VERSION=4.23.4
  export TF_NAME="tensorflow"
fi

python -m pip install \
  --platform=macosx_12_0_arm64 \
  --target=$(python -c 'import site; print(site.getsitepackages()[0])') \
  --upgrade \
  --only-binary=:all: \
  protobuf~=$PROTOBUF_VERSION $TF_NAME==$TF_VERSION

# For tensorflow version 2.11.0, and python version = 3.10 and 3.9
# numpy>=1.20 is required, then it will install numpy 2.0 and cause errors.
PYTHON_VERSION=$(python -V | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$TF_VERSION" == "2.11.0" && ( "$PYTHON_VERSION" == "3.9" || "$PYTHON_VERSION" == "3.10" ) ]]; then
  python -m pip install numpy==1.26.4 --force-reinstall
fi

python configure.py

bazel build \
  --cpu=darwin_arm64 \
  --copt -mmacosx-version-min=12.0 \
  --linkopt -mmacosx-version-min=12.0 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts "--plat-name macosx_12_0_arm64 $NIGHTLY_FLAG"
delocate-wheel -w wheelhouse -v --ignore-missing-dependencies artifacts/*.whl

# Test
pip install --default-timeout=1000 -r tools/install_deps/pytest.txt
cp ./bazel-bin/tensorflow_recommenders_addons/dynamic_embedding/core/_*_ops.so ./tensorflow_recommenders_addons/dynamic_embedding/core/
python -m pytest -v -s --functions-durations=20 --modules-durations=5 $IGNORE_HKV $IGNORE_REDIS $IGNORE_REDIS_VAR $SKIP_CUSTOM_OP_TESTS ./tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/

# Clean
bazel clean