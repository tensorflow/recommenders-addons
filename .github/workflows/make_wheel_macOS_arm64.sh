#!/bin/bash
#
# Making wheel for macOS arm64 architecture
# Requirements: 
# MacOS Monterey 12.0.0 +, Tensorflow-macos 2.5.0, ARM64 Apple Silicon, Bazel 4.1.0 +, Python 3.8 or 3.9
# Please don't install tensorflow-metal, it may cause incorrect GPU devices detection issue.
set -e -x

python --version

python -m pip install --default-timeout=1000 delocate==0.9.1 wheel==0.37.0 setuptools==50.0.0 tensorflow-macos==$TF_VERSION
python -m pip install --upgrade protobuf==3.19.6 numpy==1.19.5

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
bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG
delocate-wheel -w wheelhouse artifacts/*.whl
