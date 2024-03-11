#syntax=docker/dockerfile:1.1.5-experimental
FROM python:3.9 as build_wheel

ARG TF_VERSION="2.15.1"
ARG PY_VERSION="3.9"
ARG MPI_VERSION="4.1.1"
ARG HOROVOD_VERSION="0.28.1"

RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 tensorflow==$TF_VERSION

RUN python -m pip install --upgrade protobuf==3.20.3

RUN apt-get update && apt-get install -y sudo rsync cmake openmpi-bin libopenmpi-dev

COPY tools/install_deps/install_bazelisk.sh /install/
RUN bash /install/install_bazelisk.sh

COPY tools/docker/install/install_horovod.sh /install/
RUN  /install/install_horovod.sh $HOROVOD_VERSION --only-cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY tools/install_deps/ ./
COPY tools/docker/install/install_pytest.sh /install/
RUN bash /install/install_pytest.sh
RUN pip install pytest-cov

COPY ./ /recommenders-addons
WORKDIR recommenders-addons

RUN python -m pip install tensorflow-io

RUN python -m pip install --upgrade protobuf==3.20.3
RUN python -m pip install numpy==1.26.4 --force-reinstall

RUN python configure.py
RUN pip install -e ./
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/install_so_files.sh
RUN pytest -v -s -n auto --durations=25 --doctest-modules ./tensorflow_recommenders_addons \
    --cov=tensorflow_recommenders_addons ./tensorflow_recommenders_addons/

RUN bazel build --enable_runfiles build_pip_pkg
RUN bazel-bin/build_pip_pkg artifacts


FROM python:3.9

ARG TF_VERSION="2.15.1"

RUN pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY --from=0 /recommenders-addons/artifacts /artifacts

RUN python -m pip install --upgrade protobuf==3.20.3

RUN pip install /artifacts/tensorflow_recommenders_addons-*.whl

# check that we didnd't forget to add a py file to
# The corresponding BUILD file.
# Also test that the wheel works in a fresh environment
RUN python -c "import tensorflow_recommenders_addons"
