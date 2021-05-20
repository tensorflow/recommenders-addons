#! /bin/bash
#
# Copyright 2021.
# Author: Yafei Zhang (kimmyzhang@tencent.com)
#

set -e
cd $(dirname $0)
source env.sh

CACHE_DIR=$_BAZEL_CACHE_DIR
TMP_DIR=$(pwd)/tmp

function install() {
    local src=$1
    local dest=$2
    echo Installing $src $dest
    mkdir -p $(dirname $dest)
    cp $src $dest
    chmod 644 $dest
}

function link() {
    local src=$1
    local dest=$2
    echo Linking $src to $dest...
    mkdir -p $(dirname $dest)
    rm -rf $dest
    ln -s $src $dest
}

# 1. construct all possible include files.
rm -rf $TMP_DIR
mkdir $TMP_DIR
cd $TMP_DIR
# 1.1. third party
link $CACHE_DIR/external/com_google_absl/absl absl
link $CACHE_DIR/external/eigen_archive/Eigen Eigen
link $CACHE_DIR/external/eigen_archive/unsupported/Eigen unsupported/Eigen
# TODO(kimi): external/aws/aws-cpp-sdk-core/include/aws/core/SDKConfig.h?
link $CACHE_DIR/external/aws/aws-cpp-sdk-core external/aws/aws-cpp-sdk-core
link $CACHE_DIR/external/aws/aws-cpp-sdk-kinesis external/aws/aws-cpp-sdk-kinesis
link $CACHE_DIR/external/aws/aws-cpp-sdk-s3 external/aws/aws-cpp-sdk-s3
link $CACHE_DIR/external/boringssl external/boringssl
link $CACHE_DIR/external/com_googlesource_code_re2 external/com_googlesource_code_re2
link $CACHE_DIR/external/curl external/curl
link $CACHE_DIR/external/double_conversion external/double_conversion
link $CACHE_DIR/external/farmhash_archive external/farmhash_archive
link $CACHE_DIR/external/gif_archive external/gif_archive
link $CACHE_DIR/external/highwayhash external/highwayhash
link $CACHE_DIR/external/jpeg external/jpeg
link $CACHE_DIR/external/local_config_cuda external/local_config_cuda
link $CACHE_DIR/external/nsync external/nsync
link $CACHE_DIR/external/zlib_archive external/zlib_archive
link $CACHE_DIR/external/com_google_protobuf/src/google/protobuf google/protobuf
link $CACHE_DIR/external/jsoncpp_git/include include
link $TENSORFLOW_DIR/third_party/eigen3/Eigen third_party/eigen3/Eigen
link $TENSORFLOW_DIR/third_party/eigen3/unsupported/Eigen third_party/eigen3/unsupported/Eigen
# 1.2. tensorflow
rm -rf tensorflow && mkdir tensorflow
files=$(find $TENSORFLOW_DIR/tensorflow -regextype posix-extended -regex ".*\.(h|inc)")
for file in $files; do
    install $file ${file/$TENSORFLOW_DIR\/}
done

# 1.3. tensorflow gen file
files=$(find -L $TTENSORFLOW_GENFILE_DIR/tensorflow/ -regextype posix-extended -regex ".*\.(h|inc)" -printf "%P\n")
for file in $files; do
    install $TTENSORFLOW_GENFILE_DIR/tensorflow/$file ./tensorflow/$file
done
# 2. copy *.h to $OUTPUT_INCLUDE_DIR
cd -
rm -rf $OUTPUT_INCLUDE_DIR
mkdir $OUTPUT_INCLUDE_DIR

files=$(find -L $TMP_DIR/ -regextype posix-extended -regex ".*\.(h|inc)" -printf "%P\n")
for file in $files; do
    install $TMP_DIR/$file $OUTPUT_INCLUDE_DIR/$file
done
files=$(find -L $TMP_DIR/ -type f -name "*" -printf "%P\n" |grep -i "Eigen")
for file in $files; do
    install $TMP_DIR/$file $OUTPUT_INCLUDE_DIR/$file
done

cd -