#!/bin/bash
#
# Making wheel for macOS arm64 architecture
# Requirements: 
# MacOS Monterey 12.0.0 +, Tensorflow-macos 2.6.0 or 2.8.0, ARM64 Apple Silicon, Bazel 4.1.0 +
# Please don't install tensorflow-metal, it may cause incorrect GPU devices detection issue.
set -e -x

python --version

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
