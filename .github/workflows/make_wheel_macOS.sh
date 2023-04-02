set -e -x

export TF_NEED_CUDA=0

python --version
python -m pip install --default-timeout=1000 delocate==0.9.1 wheel==0.37.0 setuptools==50.0.0 tensorflow==$TF_VERSION
python -m pip install --upgrade protobuf==3.19.6

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

