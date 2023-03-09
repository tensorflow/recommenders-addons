# Dockerfile to build a manylinux 2010 compliant cross-compiler.
#
# Builds a devtoolset gcc/libstdc++ that targets manylinux 2010 compatible
# glibc (2.12) and system libstdc++ (4.4).
#
# To push a new version, run:
# $ docker build -f cuda11.2.1-cudnn8-ubuntu20.04-manylinux2014-multipython.Dockerfile . \
#  --tag "tfra/nosla-cuda11.2.1-cudnn8-ubuntu20.04-manylinux2014-multipython"
# $ docker push tfra/nosla-cuda11.2.1-cudnn8-ubuntu20.04-manylinux2014-multipython

FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04 as devtoolset

RUN chmod 777 /tmp/
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      cpio \
      file \
      flex \
      g++ \
      make \
      patch \
      rpm2cpio \
      unar \
      wget \
      xz-utils \
      libjpeg-dev \
      zlib1g-dev \
      libgflags-dev \
      libsnappy-dev \
      libbz2-dev \
      liblz4-dev \
      libzstd-dev \
      openssh-client \
      && \
    rm -rf /var/lib/apt/lists/*

ADD devtoolset/fixlinks.sh fixlinks.sh
ADD devtoolset/build_devtoolset.sh build_devtoolset.sh
ADD devtoolset/rpm-patch.sh rpm-patch.sh

# Set up a sysroot for glibc 2.12 / libstdc++ 4.4 / devtoolset-7 in /dt7.
RUN /build_devtoolset.sh devtoolset-7 /dt7
# Set up a sysroot for glibc 2.12 / libstdc++ 4.4 / devtoolset-8 in /dt8.
RUN /build_devtoolset.sh devtoolset-8 /dt8

# TODO(klimek): Split up into two different docker images.
FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
COPY --from=devtoolset /dt7 /dt7
COPY --from=devtoolset /dt8 /dt8

# Install TensorRT.
RUN echo \
    deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 / \
    > /etc/apt/sources.list.d/nvidia-ml.list \
      && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80 && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*

# Copy and run the install scripts.
ARG DEBIAN_FRONTEND=noninteractive

COPY install/install_bootstrap_deb_packages.sh /install/
RUN /install/install_bootstrap_deb_packages.sh

COPY install/install_deb_packages.sh /install/
RUN /install/install_deb_packages.sh

# Install additional packages needed for this image:
# - dependencies to build Python from source
# - patchelf, as it is required by auditwheel
RUN apt-get update && apt-get install -y \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    libncurses5-dev \
    libnss3-dev \
    libreadline-dev \
    patchelf \
    gcc-multilib \
      && \
    rm -rf /var/lib/apt/lists/*

RUN chmod 777 /tmp/
WORKDIR /tmp/

COPY install/install_nccl.sh /install/
RUN /install/install_nccl.sh "2.8.4-1+cuda11.2"

COPY install/install_rocksdb.sh /install/
RUN /install/install_rocksdb.sh "6.22.1"

COPY install/install_bazel.sh /install/
RUN /install/install_bazel.sh "5.1.1"

COPY install/build_and_install_python.sh /install/
RUN /install/build_and_install_python.sh "3.7.7"
RUN /install/build_and_install_python.sh "3.8.2"
RUN /install/build_and_install_python.sh "3.9.7"
RUN /install/build_and_install_python.sh "3.10.6"

COPY install/install_pip_packages_by_version.sh /install/
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.10"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.9"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.8"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.7"

COPY install/use_devtoolset_7.sh /install/
RUN /install/use_devtoolset_7.sh

COPY install/install_openmpi.sh /install/
RUN /install/install_openmpi.sh "4.1.1"

# clean
RUN rm -rf /tmp/*
