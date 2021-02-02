set -e -x

if [ "$TF_NEED_CUDA" == "1" ]; then
  CUDA_FLAG="--crosstool_top=//build_deps/toolchains/gcc7_manylinux2010-nvcc-cuda10.1:toolchain"
fi

bazel build $CUDA_FLAG //tensorflow_recommenders_addons/...
cp ./bazel-bin/tensorflow_recommenders_addons/dynamic_embedding/core/_*_ops.so ./tensorflow_recommenders_addons/dynamic_embedding/core/
cp ./bazel-bin/tensorflow_recommenders_addons/embedding_variable/core/_*_ops.so ./tensorflow_recommenders_addons/embedding_variable/core/
