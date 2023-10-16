#!/bin/bash
#
# Making wheel for macOS arm64 architecture
# Requirements: 
# MacOS Monterey 12.0.0 +, Tensorflow-macos 2.6.0 - 2.11.0, ARM64 Apple Silicon, Bazel 5.1.1
# Please don't install tensorflow-metal, it may cause incorrect GPU devices detection issue.
set -e -x

export TF_NEED_CUDA=0
if [ -z $HOROVOD_VERSION ] ; then
  export HOROVOD_VERSION='0.28.1'
fi

# For TensorFlow version 2.12 or earlier:
export PROTOBUF_VERSION=3.19.6
export TF_NAME="tensorflow-macos"

python --version
python -m pip install --default-timeout=1000 delocate==0.10.3 wheel setuptools
# For TensorFlow version 2.13 or later:
if [[ "$TF_VERSION" =~ ^2\.1[3-9]\.[0-9]$ ]] ; then
  export PROTOBUF_VERSION=3.20.3
  export TF_NAME="tensorflow"
fi

python -m pip install \
  --platform=macosx_12_0_arm64 \
  --target=$(python -c 'import site; print(site.getsitepackages()[0])') \
  --upgrade \
  --only-binary=:all: \
  protobuf~=$PROTOBUF_VERSION $TF_NAME==$TF_VERSION

python configure.py
# Setting DYLD_LIBRARY_PATH to help delocate finding tensorflow after the rpath invalidation
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(python -c 'import configure; print(configure.get_tf_shared_lib_dir())')

bazel build \
  --cpu=darwin_arm64 \
  --copt -mmacosx-version-min=12.0 \
  --linkopt -mmacosx-version-min=12.0 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts "--plat-name macosx_11_0_arm64 $NIGHTLY_FLAG"
delocate-wheel -w wheelhouse -v --ignore-missing-dependencies artifacts/*.whl