#!/usr/bin/env bash

set -x -e
docker build \
    -f tools/docker/dev_container.Dockerfile \
    --build-arg TF_VERSION=2.8.3 \
    --build-arg TF_PACKAGE=tensorflow-gpu \
    --build-arg PY_VERSION=$PY_VERSION \
    --build-arg HOROVOD_VERSION="0.23.0" \
    --no-cache \
    --target dev_container \
    -t tfra/dev_container:latest-python$PY_VERSION ./