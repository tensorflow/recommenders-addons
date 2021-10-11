# usage: bash tools/run_build.sh
# by default uses docker buildkit.
# to disable it:
# DOCKER_BUILDKIT=0 bash tools/run_build.sh
set -e

DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/sanity_check.Dockerfile \
    --build-arg USE_BAZEL_VERSION=${USE_BAZEL_VERSION:-"3.7.2"} \
    --target=${1} ./
