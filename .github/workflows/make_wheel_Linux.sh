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

if [ `echo "$TF_CUDA_VERSION == 11"|bc` -eq 1 ] ; then
  export BUILD_IMAGE="tfra/nosla-cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython"
elif [ `echo "$TF_CUDA_VERSION == 10"|bc` -eq 1 ] ; then
  export BUILD_IMAGE='tfra/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010-multipython'
else
  echo "No suitable Build Image found!"
  exit 1
fi

DOCKER_BUILDKIT=1 docker build --no-cache \
    -f tools/docker/build_wheel.Dockerfile \
    --output type=local,dest=wheelhouse \
    --build-arg PY_VERSION \
    --build-arg TF_VERSION \
    --build-arg TF_NAME \
    --build-arg TF_NEED_CUDA \
    --build-arg TF_CUDA_VERSION \
    --build-arg TF_CUDNN_VERSION \
    --build-arg BUILD_IMAGE \
    --build-arg NIGHTLY_FLAG \
    --build-arg NIGHTLY_TIME \
    ./
