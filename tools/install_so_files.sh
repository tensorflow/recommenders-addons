set -e -x

bazel --output_user_root=./mnt build --local_ram_resources=4096 --cxxopt="-w" --copt="-w" $CUDA_FLAG //tensorflow_recommenders_addons/...
mv ./bazel-bin/tensorflow_recommenders_addons/dynamic_embedding/core/_*_ops.so ./tensorflow_recommenders_addons/dynamic_embedding/core/
