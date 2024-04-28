#syntax=docker/dockerfile:1.1.5-experimental
ARG IMAGE_TYPE
ARG TF_VERSION
ARG PY_VERSION
ARG TF_NEED_CUDA
ARG TF_NAME
ARG HOROVOD_VERSION
ARG BUILD_IMAGE
ARG PROTOBUF_VERSION

# Currenly all of our dev images are GPU capable but at a cost of being quite large.
# See https://github.com/tensorflow/build/pull/47
FROM ${BUILD_IMAGE} as dev_container

RUN echo "#! /usr/bin/python2.7" >> /usr/bin/lsb_release2
RUN cat /usr/bin/lsb_release >> /usr/bin/lsb_release2
RUN mv /usr/bin/lsb_release2 /usr/bin/lsb_release

ARG PY_VERSION
RUN ln -sf /usr/local/bin/python$PY_VERSION /usr/bin/python

ENV PATH=/dt8/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/dt8/user/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/dt8/user/lib:${LD_LIBRARY_PATH}
ENV MANPATH=/dt8/user/share/man:${LD_LIBRARY_PATH}
ENV INFOPATH=/dt8/user/share/info

ARG TF_VERSION
ARG TF_NAME
ARG HOROVOD_VERSION
ARG PROTOBUF_VERSION

RUN python -m pip install --upgrade pip
RUN python -m pip install --default-timeout=1000 $TF_NAME==$TF_VERSION

COPY tools/docker/install/install_horovod.sh /install/
RUN /install/install_horovod.sh $HOROVOD_VERSION

COPY tools/install_deps/ ./
COPY tools/docker/install/install_pytest.sh /install/
RUN bash /install/install_pytest.sh

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN python -m pip install tensorflow-io

RUN python -m pip install --upgrade protobuf==$PROTOBUF_VERSION

COPY tools/install_deps/yapf.txt ./
RUN pip install -r ./yapf.txt

RUN pip install setuptools

COPY tools/install_deps/buildifier.sh ./buildifier.sh
RUN bash buildifier.sh

COPY tools/install_deps/clang-format.sh ./clang-format.sh
RUN bash clang-format.sh

ARG IMAGE_TYPE
ARG TF_VERSION
ARG PY_VERSION
ARG TF_NEED_CUDA
ARG TF_CUDA_VERSION
ARG TF_CUDNN_VERSION
ARG TF_NAME
ARG HOROVOD_VERSION
ARG BUILD_IMAGE
ARG PROTOBUF_VERSION

# write default env for user
RUN echo "export PATH=$PATH" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc
RUN echo "export MANPATH=$MANPATH" >> ~/.bashrc
RUN echo "export INFOPATH=$INFOPATH" >> ~/.bashrc
RUN echo "export TF_VERSION=$TF_VERSION" >> ~/.bashrc
RUN echo "export PY_VERSION=$PY_VERSION" >> ~/.bashrc
RUN echo "export TF_NAME=$TF_NAME" >> ~/.bashrc
RUN echo "export IMAGE_TYPE=$IMAGE_TYPE" >> ~/.bashrc
RUN echo "export HOROVOD_VERSION=$HOROVOD_VERSION" >> ~/.bashrc
RUN echo "export TF_NAME=$TF_NAME" >> ~/.bashrc
RUN echo "export PROTOBUF_VERSION=$PROTOBUF_VERSION" >> ~/.bashrc
RUN echo "export TF_NEED_CUDA=1" >> ~/.bashrc
RUN echo "export TF_CUDA_VERSION=$TF_CUDA_VERSION" >> ~/.bashrc
RUN echo "export TF_CUDNN_VERSION=$TF_CUDNN_VERSION" >> ~/.bashrc
RUN echo "export CUDA_TOOLKIT_PATH='/usr/local/cuda'" >> ~/.bashrc
RUN echo "export CUDNN_INSTALL_PATH='/usr/lib/x86_64-linux-gnu'" >> ~/.bashrc
