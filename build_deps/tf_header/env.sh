#! /bin/bash
#
# Copyright 2021.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

PYTHON_PATH=$HOME/python-3.7
TENSORFLOW_VERSION=1.15.2
BAZEL_DIR=$HOME/bazel-0.24.1
BAZEL_CACHE_DIR=$HOME/bazel-cache
# TODO(kimi): 860b4776177384b48cc611bdf8c7a91c may change?
_BAZEL_CACHE_DIR=$BAZEL_CACHE_DIR/_bazel_root/860b4776177384b48cc611bdf8c7a91c
TENSORFLOW_DIR=$HOME/tensorflow
TENSORFLOW_GENFILE_DIR=$_BAZEL_CACHE_DIR/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflo
OUTPUT_INCLUDE_DIR=$(dirname "${BASH_SOURCE[0]}")/$TENSORFLOW_VERSION/tensorflow

function use() {
    if test "x$1" == x; then
        echo 'Use a software by changing PATH, LD_LIBRARY_PATH, MANPATH...'
        echo 'Usage: use path'
        return 1
    fi
    DIR=$1
    if test -d $DIR; then
        echo Use $DIR
        if test -f $DIR/_bashrc; then
            source $DIR/_bashrc
        fi
        if test -d $DIR/bin; then
            export PATH=$DIR/bin:$PATH
        fi
        if test -d $DIR/lib; then
            export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH
        fi
        if test -d $DIR/lib64; then
            export LD_LIBRARY_PATH=$DIR/lib64:$LD_LIBRARY_PATH
        fi
        if test -d $DIR/share/man; then
            export MANPATH=$DIR/share/man:$LD_LIBRARY_PATH
        fi
        if test -d $DIR/share/info; then
            export INFOPATH=$DIR/share/info:$LD_LIBRARY_PATH
        fi
    fi
}

use $PYTHON_PATH
use $BAZEL_DIR