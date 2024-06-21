# A Horovod synchronous distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/1m-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens1m-ratings)
- model: DNN
- Running mode: Graph mode and Keras by using Horovod AllToAll Embedding as model parameters parallelism
- enable gpu by `python3 -m pip install tensorflow[and-cuda]`
- `HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod`
- recommend to use nv docker image `nvcr.io/nvidia/tensorflow:24.02-tf2-py3`
- run `rm -rf model_dir/ export_dir/` to clean up the model and export directory before running the script
## start train:
By default, this shell will start a train task with N workers as GPU number on local machine.

```shell
sh start.sh
```
