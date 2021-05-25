# Extract Tensorflow headers TensorFlow compiling dirctory
-----------------

For some C++ headers needed by TFRA are not included in installation directory of old version Tensorflow.
If you want to compile TFRA with these TF, extracting headers from TensorFlow compiling dirctory is needed. Do the following:

1. Compile TensorFlow with version you need from source
- [Guide of Build Tensorflow from source](https://www.tensorflow.org/install/source)

2. Modify configuration in env.sh
**Note:** The BAZEL_CACHE_DIR name is created randomly, you need to change it.
`_BAZEL_CACHE_DIR=$BAZEL_CACHE_DIR/_bazel_root/860b4776177384b48cc611bdf8c7a91c`

3. Extract headers
```shell script
./mkinclude.sh
```

4. Copy $OUTPUT_INCLUDE_DIR to direcory below:
```shell script
cp -r $_BAZEL_CACHE_DIR $TFRA_HOME/build_deps/tf_header/$TF_VERSION/
```

5. Start compiling TFRA
```shell script
PY_VERSION="3.6" TF_VERSION="1.15.2" TF_NEED_CUDA=0 sh .github/workflows/make_wheel_Linux.sh
```


