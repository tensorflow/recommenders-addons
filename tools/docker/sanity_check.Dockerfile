#syntax=docker/dockerfile:1.1.5-experimental
# -------------------------------
FROM python:3.9 as yapf-test

COPY tools/install_deps/yapf.txt ./
RUN pip install -r yapf.txt
COPY ./ /recommenders-addons
WORKDIR /recommenders-addons

RUN python -m pip install --upgrade protobuf==3.20.3
RUN python tools/check_python_format.py
RUN touch /ok.txt

# -------------------------------
FROM python:3.9 as source_code_test

COPY tools/install_deps /install_deps
COPY tools/install_deps/pytest.txt ./
COPY tools/docker/install/install_pytest.sh /install/
RUN bash /install/install_pytest.sh
RUN --mount=type=cache,id=cache_pip,target=/root/.cache/pip \
    cd /install_deps && pip install \
    --default-timeout=1000 \
    -r tensorflow-cpu.txt \
    -r typedapi.txt

RUN apt-get update && apt-get install -y sudo rsync cmake
COPY tools/install_deps/install_bazelisk.sh .bazelversion ./
RUN bash install_bazelisk.sh

COPY ./ /recommenders-addons
RUN pip install -e /recommenders-addons

WORKDIR /recommenders-addons

RUN python -m pip install tensorflow-io

RUN python -m pip install --upgrade protobuf==3.20.3

RUN python configure.py
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/install_so_files.sh

RUN pytest -v -s /recommenders-addons/tools/testing/
RUN touch /ok.txt

# -------------------------------
FROM python:3.9 as valid_build_files

COPY tools/install_deps/tensorflow.txt ./
RUN pip install --default-timeout=1000 -r tensorflow.txt

RUN apt-get update && apt-get install sudo
COPY tools/install_deps/install_bazelisk.sh .bazelversion ./
RUN bash install_bazelisk.sh

COPY ./ /recommenders-addons
WORKDIR /recommenders-addons

RUN python -m pip install --upgrade protobuf==3.20.3

RUN python ./configure.py
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bazel build --nobuild -- //tensorflow_recommenders_addons/...
RUN touch /ok.txt

# -------------------------------
FROM python:3.9-alpine as clang-format

RUN apk add --no-cache git
RUN git clone https://github.com/gabrieldemarmiesse/clang-format-lint-action.git
WORKDIR ./clang-format-lint-action
RUN git checkout 1044fee

COPY ./ /recommenders-addons
RUN python run-clang-format.py \
               -r \
               --cli-args=--style=google \
               --clang-format-executable ./clang-format/clang-format9 \
               /recommenders-addons/tensorflow_recommenders_addons
RUN touch /ok.txt

# -------------------------------
# Bazel code format
FROM alpine:3.11 as check-bazel-format

RUN apk update && apk add bash
COPY ./tools/install_deps/buildifier.sh ./
RUN sh buildifier.sh

COPY ./ /recommenders-addons
RUN buildifier -mode=diff -r /recommenders-addons
RUN touch /ok.txt

# -------------------------------
# docs tests
FROM python:3.9 as docs_tests

COPY tools/install_deps/tensorflow.txt ./
RUN pip install --default-timeout=1000 -r tensorflow.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY tools/install_deps/doc_requirements.txt ./
RUN pip install -r doc_requirements.txt

RUN apt-get update && apt-get install -y sudo rsync cmake
COPY tools/install_deps/install_bazelisk.sh .bazelversion ./
RUN bash install_bazelisk.sh

COPY ./ /recommenders-addons
WORKDIR /recommenders-addons

RUN python -m pip install --upgrade protobuf==3.20.3

RUN python configure.py
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/install_so_files.sh

RUN pip install --no-deps -e .
RUN python tools/docs/build_docs.py
RUN touch /ok.txt

# -------------------------------
# test the editable mode
FROM python:3.9 as test_editable_mode

COPY tools/install_deps/tensorflow.txt ./
RUN pip install --default-timeout=1000 -r tensorflow.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY tools/install_deps/pytest.txt ./
RUN pip install -r pytest.txt

RUN apt-get update && apt-get install -y sudo rsync cmake
COPY tools/install_deps/install_bazelisk.sh .bazelversion ./
RUN bash install_bazelisk.sh

COPY ./ /recommenders-addons
WORKDIR /recommenders-addons

RUN python -m pip install --upgrade protobuf==3.20.3

RUN python configure.py
RUN --mount=type=cache,id=cache_bazel,target=/root/.cache/bazel \
    bash tools/install_so_files.sh
RUN pip install --no-deps -e .
RUN pytest -v -s -n auto ./tensorflow_recommenders_addons/dynamic_embedding
RUN touch /ok.txt

# -------------------------------
# ensure that all checks were successful
# this is necessary if using docker buildkit
# with "export DOCKER_BUILDKIT=1"
# otherwise dead branch elimination doesn't
# run all tests
FROM scratch

COPY --from=0 /ok.txt /ok0.txt
COPY --from=1 /ok.txt /ok1.txt
COPY --from=2 /ok.txt /ok2.txt
COPY --from=3 /ok.txt /ok3.txt
COPY --from=4 /ok.txt /ok4.txt
COPY --from=5 /ok.txt /ok5.txt
COPY --from=6 /ok.txt /ok6.txt
