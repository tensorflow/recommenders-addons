set -e -x

bazel build --cxxopt="-w" --copt="-w" --jobs 4 $CUDA_FLAG //tensorflow_recommenders_addons/...
cp ./bazel-bin/tensorflow_recommenders_addons/dynamic_embedding/core/_*_ops.so ./tensorflow_recommenders_addons/dynamic_embedding/core/
cp ./bazel-bin/tensorflow_recommenders_addons/embedding_variable/core/_*_ops.so ./tensorflow_recommenders_addons/embedding_variable/core/
