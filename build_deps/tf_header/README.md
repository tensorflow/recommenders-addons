# Extract headers from TensorFlow compiling directory
-----------------

For some C++ headers needed by TFRA are not included in installation directory of old version Tensorflow.
If you want to compile TFRA with these TF, it is necessary for extracting headers from TensorFlow compiling dirctory. 
Do as below:

1. Compile TensorFlow with version you need to work with
- [Guide of Build Tensorflow from source](https://www.tensorflow.org/install/source)

2. Modify `BAZEL_CACHE_DIR` in [./env.sh](./env.sh), for `$BAZEL_CACHE_DIR` is created randomly, 
so change it after step 1 :
`_BAZEL_CACHE_DIR=$BAZEL_CACHE_DIR/_bazel_root/860b4776177384b48cc611bdf8c7a91c`

3. Extract headers by executing [./mkinclude.sh](./mkinclude.sh)
```shell script
./mkinclude.sh
```

4. Copy `$OUTPUT_INCLUDE_DIR` to TFRA compile direcory
```shell script
cp -r $_BAZEL_CACHE_DIR $TFRA_HOME/build_deps/tf_header/$TF_VERSION/
```

5. Start compiling TFRA
```shell script
PY_VERSION="3.7" TF_VERSION="1.15.2" TF_NEED_CUDA=0 sh .github/workflows/make_wheel_Linux.sh
```
