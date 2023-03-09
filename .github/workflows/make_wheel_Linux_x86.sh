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

if [ $TF_VERSION == "2.6.3" ]  || [ $TF_VERSION == "2.8.3" ] ; then
  export BUILD_IMAGE="tfra/nosla-cuda11.2.1-cudnn8-ubuntu20.04-manylinux2014-multipython"
  export TF_CUDA_VERSION="11.2"
  export TF_CUDNN_VERSION="8.1"
elif [ $TF_VERSION == "2.4.1" ] ; then
  export BUILD_IMAGE='tfra/nosla-cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython'
  export TF_CUDA_VERSION="11.0"
  export TF_CUDNN_VERSION="8.0"
elif [ $TF_VERSION == "1.15.2" ] ; then
  export BUILD_IMAGE='tfra/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010-multipython'
  export TF_CUDA_VERSION="10.0"
  export TF_CUDNN_VERSION="7.6"
else
  echo "TF_VERSION is invalid: $TF_VERSION!"
  exit 1
fi

if [ -z $HOROVOD_VERSION ] ; then
  export HOROVOD_VERSION='0.23.0'
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
    --build-arg HOROVOD_VERSION \
    --build-arg BUILD_IMAGE \
    --build-arg NIGHTLY_FLAG \
    --build-arg NIGHTLY_TIME \
    ./
