#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
ARG PY_VERSION
ARG TF_NEED_CUDA
ARG TF_NAME
ARG HOROVOD_VERSION
ARG BUILD_IMAGE
ARG PROTOBUF_VERSION
FROM ${BUILD_IMAGE} as base_install

# Required for setuptools v50.0.0
# https://setuptools.readthedocs.io/en/latest/history.html#v50-0-0
# https://github.com/pypa/setuptools/issues/2352
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Fix presented in
# https://stackoverflow.com/questions/44967202/pip-is-showing-error-lsb-release-a-returned-non-zero-exit-status-1/44967506
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

RUN if [ "$TF_VERSION" = "2.11.0" ] && ([ "$PY_VERSION" = "3.9" ] || [ "$PY_VERSION" = "3.10" ]); then \
        pip install numpy==1.26.4 --force-reinstall; \
    fi

COPY tools/docker/install/install_horovod.sh /install/
RUN /install/install_horovod.sh $HOROVOD_VERSION

COPY tools/install_deps/ ./
COPY tools/docker/install/install_pytest.sh /install/
RUN bash /install/install_pytest.sh

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN python -m pip install tensorflow-io

RUN python -m pip install --upgrade protobuf==$PROTOBUF_VERSION

COPY ./ /recommenders-addons
WORKDIR /recommenders-addons

# -------------------------------------------------------------------
FROM base_install as tfra_gpu_tests
CMD ["bash", "tools/testing/build_and_run_tests.sh $SKIP_CUSTOM_OP_TESTS"]

# -------------------------------------------------------------------
FROM base_install as make_wheel
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME
ARG TF_NEED_CUDA
ARG TF_CUDA_VERSION
ARG TF_CUDNN_VERSION
ARG HOROVOD_VERSION
ARG PROTOBUF_VERSION
ENV TF_NEED_CUDA=$TF_NEED_CUDA
ENV TF_CUDA_VERSION=$TF_CUDA_VERSION
ENV TF_CUDNN_VERSION=$TF_CUDNN_VERSION

RUN python -m pip install --upgrade pip
RUN python configure.py

RUN bash tools/testing/build_and_run_tests.sh $SKIP_CUSTOM_OP_TESTS && \
    bazel build --local_ram_resources=4096 \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2014_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------

FROM python:$PY_VERSION as test_wheel_in_fresh_environment

ARG TF_VERSION
ARG TF_NAME
ARG PROTOBUF_VERSION

RUN python -m pip install --upgrade pip
RUN python -m pip install --default-timeout=1000 $TF_NAME==$TF_VERSION
RUN python -m pip install --upgrade protobuf==$PROTOBUF_VERSION

COPY --from=make_wheel /recommenders-addons/wheelhouse/ /recommenders-addons/wheelhouse/
RUN pip install /recommenders-addons/wheelhouse/*.whl

RUN PYTHON_VERSION=$(python -V | cut -d' ' -f2 | cut -d'.' -f1,2) && \
    if [ "$TF_VERSION" = "2.11.0" ] && ([ "$PYTHON_VERSION" = "3.9" ] || [ "$PYTHON_VERSION" = "3.10" ]); then \
        pip install numpy==1.26.4 --force-reinstall; \
    fi
RUN python -c "import tensorflow_recommenders_addons as tfra; print(tfra.register_all())"


# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /recommenders-addons/wheelhouse/ .
