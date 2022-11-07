set -e -x

export TF_NEED_CUDA=0
if [ -z $HOROVOD_VERSION ] ; then
  export HOROVOD_VERSION='0.23.0'
fi
python --version

brew install open-mpi

python -m pip install --default-timeout=1000 delocate==0.9.1 wheel==0.36.2 setuptools tensorflow==$TF_VERSION
python -m pip install tensorflow-io
python -m pip install --upgrade protobuf==3.20.0

bash tools/docker/install/install_horovod.sh $HOROVOD_VERSION --only-cpu

bash tools/testing/build_and_run_tests.sh

bazel build \
  -c opt \
  --copt -mmacosx-version-min=10.13 \
  --linkopt -mmacosx-version-min=10.13 \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg

bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG
delocate-wheel -w wheelhouse artifacts/*.whl

