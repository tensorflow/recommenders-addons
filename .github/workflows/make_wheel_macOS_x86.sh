set -e -x

export TF_NEED_CUDA=0
if [ -z $HOROVOD_VERSION ] ; then
  export HOROVOD_VERSION='0.28.1'
fi
python --version

# For TensorFlow version 2.12 or earlier:
export PROTOBUF_VERSION=3.19.6
if [[ "$TF_VERSION" =~ ^2\.1[3-9]\.[0-9]$ ]] ; then
  export PROTOBUF_VERSION=3.20.3
fi

brew install open-mpi
python -m pip install --default-timeout=1000 delocate==0.10.3 wheel==0.36.2 setuptools tensorflow==$TF_VERSION
python -m pip install tensorflow-io
python -m pip install --upgrade protobuf~=$PROTOBUF_VERSION

bash tools/docker/install/install_horovod.sh $HOROVOD_VERSION --only-cpu

# Test
bash tools/testing/build_and_run_tests.sh $SKIP_CUSTOM_OP_TESTS

# Clean
bazel clean

# Build
python configure.py

bazel build \
  -c opt \
  --copt -mmacosx-version-min=10.15 \
  --linkopt -mmacosx-version-min=10.15 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

# Setting DYLD_LIBRARY_PATH to help delocate finding tensorflow after the rpath invalidation
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(python -c 'import configure; print(configure.get_tf_shared_lib_dir())')
delocate-wheel -w wheelhouse -v --ignore-missing-dependencies artifacts/*.whl

