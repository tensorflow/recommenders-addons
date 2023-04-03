set -e -x

export TF_NEED_CUDA=0
export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/"

if [ -z $HOROVOD_VERSION ] ; then
  export HOROVOD_VERSION='0.23.0'
fi

python -m pip install --default-timeout=1000 wheel setuptools tensorflow==$TF_VERSION horovod==$HOROVOD_VERSION
bash ./tools/testing/build_and_run_tests.sh

python -m pip install --upgrade protobuf==3.19.6

python configure.py

bazel.exe build  --no-cache \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg
bazel-bin/build_pip_pkg wheelhouse $NIGHTLY_FLAG
