# usage: bash tools/run_sanity_check.sh

set -e

DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/sanity_check.Dockerfile \
    --build-arg USE_BAZEL_VERSION=${USE_BAZEL_VERSION:-"5.1.1"} \
    ./