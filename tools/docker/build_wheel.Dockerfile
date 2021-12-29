#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
ARG PY_VERSION
ARG TF_NEED_CUDA
ARG TF_NAME
ARG BUILD_IMAGE
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

# Use devtoolset-7 as tool chain
RUN rm -r /usr/bin/gcc*
ENV PATH=/dt7/usr/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/dt7/user/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/dt7/user/lib:${LD_LIBRARY_PATH}
ENV MANPATH=/dt7/user/share/man:${LD_LIBRARY_PATH}
ENV INFOPATH=/dt7/user/share/info
RUN ln -sf /dt7/usr/bin/cc /usr/bin/gcc
RUN ln -sf /dt7/usr/bin/gcc /usr/bin/gcc
RUN ln -sf /dt7/usr/bin/g++ /usr/bin/g++

ARG TF_VERSION
ARG TF_NAME

RUN python -m pip install --default-timeout=1000 $TF_NAME==$TF_VERSION

COPY tools/install_deps/ /install_deps
RUN python -m pip install -r /install_deps/pytest.txt

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY ./ /recommenders-addons
WORKDIR /recommenders-addons

# -------------------------------------------------------------------
FROM base_install as tfra_gpu_tests
CMD ["bash", "tools/testing/build_and_run_tests.sh"]

# -------------------------------------------------------------------
FROM base_install as make_wheel
ARG NIGHTLY_FLAG
ARG NIGHTLY_TIME
ARG TF_NEED_CUDA
ARG TF_CUDA_VERSION
ARG TF_CUDNN_VERSION
ENV TF_NEED_CUDA=$TF_NEED_CUDA
ENV TF_CUDA_VERSION=$TF_CUDA_VERSION
ENV TF_CUDNN_VERSION=$TF_CUDNN_VERSION

RUN python configure.py

RUN bash tools/testing/build_and_run_tests.sh && \
    bazel build \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------

FROM python:$PY_VERSION as test_wheel_in_fresh_environment

ARG TF_VERSION
ARG TF_NAME
RUN python -m pip install --default-timeout=1000 $TF_NAME==$TF_VERSION

COPY --from=make_wheel /recommenders-addons/wheelhouse/ /recommenders-addons/wheelhouse/
RUN pip install /recommenders-addons/wheelhouse/*.whl

RUN python -c "import tensorflow_recommenders_addons as tfra; print(tfra.register_all())"

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /recommenders-addons/wheelhouse/ .
