# Dockerfile to build a manylinux 2010 compliant cross-compiler.
#
# Builds a devtoolset gcc/libstdc++ that targets manylinux 2010 compatible
# glibc (2.12) and system libstdc++ (4.4).
#
# To push a new version, run:
# $ docker build -f cuda11.2-cudnn8.1-ubuntu18.04-manylinux2010-multipython.Dockerfile . \
#  --tag "tfra/nosla-cuda11.2-cudnn8.1-ubuntu18.04-manylinux2010-multipython"
# $ docker push tfra/nosla-cuda11.2-cudnn8.1-ubuntu18.04-manylinux2010-multipython

FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04 as devtoolset

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
FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
COPY --from=devtoolset /dt7 /dt7
COPY --from=devtoolset /dt8 /dt8

RUN chmod 777 /tmp/

# Install TensorRT.
RUN echo \
    deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 / \
    > /etc/apt/sources.list.d/nvidia-ml.list \
      && \
    apt-get update && apt-get install -y \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-plugin-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0 \
      && \
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
      && \
    rm -rf /var/lib/apt/lists/*

COPY install/install_bazel.sh /install/
RUN /install/install_bazel.sh "3.7.2"

COPY install/build_and_install_python.sh /install/
RUN /install/build_and_install_python.sh "3.6.9"
RUN /install/build_and_install_python.sh "3.7.7"
RUN /install/build_and_install_python.sh "3.8.2"
RUN /install/build_and_install_python.sh "3.9.7"

COPY install/install_pip_packages_by_version.sh /install/
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.9"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.8"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.6"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.7"

ENV CLANG_VERSION="r7f6f9f4cf966c78a315d15d6e913c43cfa45c47c"
COPY install/install_latest_clang.sh /install/
RUN /install/install_latest_clang.sh
