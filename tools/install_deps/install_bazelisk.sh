#!/usr/bin/env bash
set -x -e

# Downloads bazelisk to ${output_dir} as `bazel`.
date

output_dir=${1:-"/usr/local/bin"}

mkdir -p "${output_dir}"
wget --progress=dot:mega -O ${output_dir}/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-$([ $(uname -m) = "aarch64" ] && echo "arm64" || echo "amd64")

chmod u+x "${output_dir}/bazel"

if [[ ! ":$PATH:" =~ :${output_dir}/?: ]]; then
    PATH="${output_dir}:$PATH"
fi

which bazel
date
