#syntax=docker/dockerfile:1.1.5-experimental
FROM python:3.7 as build_wheel

ARG TF_VERSION=2.8.3
ARG MPI_VERSION="4.1.1"
ARG HOROVOD_VERSION="0.23.0"

RUN pip install --default-timeout=1000 tensorflow-cpu==$TF_VERSION

RUN python -m pip install --upgrade protobuf==3.20.0

RUN apt-get update && apt-get install -y sudo rsync cmake openmpi-bin libopenmpi-dev

COPY tools/install_deps/install_bazelisk.sh /install/
RUN bash /install/install_bazelisk.sh

COPY tools/docker/install/install_horovod.sh /install/
RUN  /install/install_horovod.sh $HOROVOD_VERSION --only-cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY tools/install_deps/pytest.txt ./
RUN pip install -r pytest.txt pytest-cov

COPY ./ /recommenders-addons
WORKDIR recommenders-addons

RUN python -m pip install tensorflow-io

RUN python -m pip install --upgrade protobuf==3.20.0

RUN python configure.py
RUN pip install -e ./
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/install_so_files.sh
RUN pytest -v -s -n auto --durations=25 --doctest-modules ./tensorflow_recommenders_addons \
    --cov=tensorflow_recommenders_addons ./tensorflow_recommenders_addons/

RUN bazel build --enable_runfiles build_pip_pkg
RUN bazel-bin/build_pip_pkg artifacts


FROM python:3.7

COPY tools/install_deps/tensorflow-cpu.txt ./
RUN pip install --default-timeout=1000 --upgrade --force-reinstall -r tensorflow-cpu.txt

COPY --from=0 /recommenders-addons/artifacts /artifacts

RUN python -m pip install --upgrade protobuf==3.20.0

RUN pip install /artifacts/tensorflow_recommenders_addons-*.whl

# check that we didnd't forget to add a py file to
# The corresponding BUILD file.
# Also test that the wheel works in a fresh environment
RUN python -c "import tensorflow_recommenders_addons"
