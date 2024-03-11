#!/usr/bin/env bash

set -x -e
docker build \
    -f tools/docker/dev_container.Dockerfile \
    --build-arg TF_VERSION=2.15.1 \
    --build-arg TF_PACKAGE=tensorflow \
    --build-arg PY_VERSION=$PY_VERSION \
    --build-arg HOROVOD_VERSION=$HOROVOD_VERSION \
    --no-cache \
    --target dev_container \
    -t tfra/dev_container:latest-python$PY_VERSION ./