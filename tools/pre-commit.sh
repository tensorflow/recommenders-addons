#!/usr/bin/env bash
# usage: bash tools/pre-commit.sh


set -e

if [ -z "${TFRA_DEV_CONTAINER}" ]; then
  export DOCKER_BUILDKIT=1
  docker build -t tf_recommenders_addons_formatting -f tools/docker/pre-commit.Dockerfile .

  export MSYS_NO_PATHCONV=1
  docker run --rm -t -v "$(pwd -P):/recommenders-addons" tf_recommenders_addons_formatting
else
  python tools/pre_commit_format.py
fi
