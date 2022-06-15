#syntax=docker/dockerfile:1.1.5-experimental
ARG IMAGE_TYPE
ARG PY_VERSION

# Currenly all of our dev images are GPU capable but at a cost of being quite large.
# See https://github.com/tensorflow/build/pull/47
FROM tensorflow/build:latest-python$PY_VERSION as dev_container
ARG TF_PACKAGE
ARG TF_VERSION
ARG PY_VERSION

RUN pip uninstall $TF_PACKAGE -y
RUN pip install --default-timeout=1000 $TF_PACKAGE==$TF_VERSION

COPY tools/install_deps /install_deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /install_deps/yapf.txt \
    -r /install_deps/pytest.txt \
    -r /install_deps/typedapi.txt \
    -r /tmp/requirements.txt

RUN bash /install_deps/buildifier.sh
RUN bash /install_deps/clang-format.sh

COPY tools/docker/install/install_bazel.sh /install/
RUN /install/install_bazel.sh "5.1.1"

ENV ADDONS_DEV_CONTAINER="1"

RUN apt-get update && apt-get install -y \
      openssh-client \
      sudo \
      rsync \
      cmake

COPY tools/docker/install/install_openmpi.sh /install/
RUN /install/install_openmpi.sh "4.1.1"

COPY tools/docker/install/install_nccl.sh /install/
RUN /install/install_nccl.sh "2.8.4-1+cuda11.2"

COPY tools/docker/install/install_horovod.sh /install/
RUN /install/install_horovod.sh "0.23.0"

# write default env for user
RUN echo "export TF_VERSION=$TF_VERSION" >> ~/.bashrc
RUN echo "export PY_VERSION=$PY_VERSION" >> ~/.bashrc
RUN echo "export TF_NEED_CUDA=1" >> ~/.bashrc
RUN echo "export TF_CUDA_VERSION=11.2" >> ~/.bashrc
RUN echo "export TF_CUDNN_VERSION=8.1" >> ~/.bashrc
RUN echo "export CUDA_TOOLKIT_PATH='/usr/local/cuda'" >> ~/.bashrc
RUN echo "export CUDNN_INSTALL_PATH='/usr/lib/x86_64-linux-gnu'" >> ~/.bashrc

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*