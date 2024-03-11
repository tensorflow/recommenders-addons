set -e -x

export TF_NEED_CUDA=0
export BAZEL_VC="C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/"

if [ -z $HOROVOD_VERSION ] ; then
  export HOROVOD_VERSION='0.28.1'
fi

python -m pip install --default-timeout=1000 wheel setuptools tensorflow==$TF_VERSION horovod==$HOROVOD_VERSION
python -m pip install tensorflow-io
# For TensorFlow version 2.12 or earlier:
export PROTOBUF_VERSION=3.19.6
if [[ "$TF_VERSION" =~ ^2\.1[3-9]\.[0-9]$ ]] ; then
  export PROTOBUF_VERSION=4.23.4
fi

python -m pip install --upgrade protobuf~=$PROTOBUF_VERSION

bash ./tools/testing/build_and_run_tests.sh

python configure.py

bazel.exe build  --no-cache \
  --noshow_progress \
  --noshow_loading_progress \
  --verbose_failures \
  --test_output=errors \
  build_pip_pkg
bazel-bin/build_pip_pkg wheelhouse $NIGHTLY_FLAG
