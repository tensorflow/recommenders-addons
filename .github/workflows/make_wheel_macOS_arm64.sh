#!/bin/bash
#
# Making wheel for macOS arm64 architecture
# Requirements: 
# MacOS Monterey 12.0.0 +, Tensorflow-macos 2.5.0, ARM64 Apple Silicon, Bazel 4.1.0 +
# Please don't install tensorflow-metal, it may cause incorrect GPU devices detection issue.
set -e -x

# Install CPU version
export TF_NEED_CUDA=0

python --version

RUN python -m pip install --upgrade protobuf==3.20.0
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

# Output the wheel file to the artifacts directory
bazel-bin/build_pip_pkg artifacts "--plat-name macosx_12_0_arm64 $NIGHTLY_FLAG"
delocate-wheel -w wheelhouse artifacts/*.whl
