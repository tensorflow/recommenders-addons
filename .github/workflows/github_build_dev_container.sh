#!/usr/bin/env bash
# export PY_VERSION=3.8 # or other python version
# sh ./.github/workflows/github_build_dev_container.sh

set -x -e

df -h
docker info
# to get more disk space
rm -rf /usr/share/dotnet &

tools/build_dev_container.sh
