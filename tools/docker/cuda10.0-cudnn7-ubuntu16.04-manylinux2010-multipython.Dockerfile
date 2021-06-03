# Dockerfile to build a manylinux 2010 compliant cross-compiler.
#
# Builds a devtoolset gcc/libstdc++ that targets manylinux 2010 compatible
# glibc (2.12) and system libstdc++ (4.4).
#
# To push a new version, run:
# $ docker build -f cuda10.0-cudnn7-ubuntu16.04-manylinux2010-multipython.Dockerfile .\
#  --tag "tfra/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010-multipython"
# $ docker push tfra/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010-multipython

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 as devtoolset

RUN chmod 777 /tmp/
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      cpio \
      file \
      flex \
      g++ \
      make \
      rpm2cpio \
      unar \
      wget \
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
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
COPY --from=devtoolset /dt7 /dt7
COPY --from=devtoolset /dt8 /dt8

# Install TensorRT.
RUN apt-get update && apt-get install -y \
    libnvinfer-dev=5.1.5-1+cuda10.0 \
    libnvinfer5=5.1.5-1+cuda10.0 \
      && \
    rm -rf /var/lib/apt/lists/*

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

# Copy and run the install scripts.
ENV CLANG_VERSION="r373795"
COPY install/*.sh /install/
ARG DEBIAN_FRONTEND=noninteractive
RUN /install/install_bootstrap_deb_packages.sh
RUN /install/install_deb_packages.sh
RUN /install/install_latest_clang.sh
RUN /install/install_bazel.sh "0.26.1"

# Install many python
COPY install/build_and_install_python.sh /install/
RUN /install/build_and_install_python.sh "3.5.9"
RUN /install/build_and_install_python.sh "3.6.9"
RUN /install/build_and_install_python.sh "3.7.7"

COPY install/install_pip_packages_by_version.sh /install/
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.5"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.6"
RUN /install/install_pip_packages_by_version.sh "/usr/local/bin/pip3.7"