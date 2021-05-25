set -e -x

df -h
docker info
# to get more disk space
rm -rf /usr/share/dotnet &

if [ $TF_NEED_CUDA -eq "1" ] ; then
  export TF_NAME='tensorflow-gpu'
else
  export TF_NAME='tensorflow'
fi

DOCKER_BUILDKIT=1 docker build \
    -f tools/docker/build_wheel.Dockerfile \
    --output type=local,dest=wheelhouse \
    --build-arg PY_VERSION \
    --build-arg TF_VERSION \
    --build-arg TF_NAME \
    --build-arg TF_NEED_CUDA \
    --build-arg NIGHTLY_FLAG \
    --build-arg NIGHTLY_TIME \
    ./
