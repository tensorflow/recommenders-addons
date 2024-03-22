# usage: bash tools/run_gpu_tests.sh

set -x -e

export DOCKER_BUILDKIT=1
docker build \
       -f tools/docker/build_wheel.Dockerfile \
       --target tfra_gpu_tests \
       --build-arg TF_VERSION=2.15.1 \
       --build-arg TF_NEED_CUDA=1 \
       --build-arg TF_NAME="tensorflow" \
       --build-arg PY_VERSION=3.9 \
       -t tfra_gpu_tests ./
docker run --rm -t -v cache_bazel:/root/.cache/bazel --gpus=all tfra_gpu_tests
