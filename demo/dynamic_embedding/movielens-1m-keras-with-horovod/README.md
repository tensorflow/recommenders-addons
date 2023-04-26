# A Horovod synchronous distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/1m-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens1m-ratings)
- model: DNN
- Running mode: Graph mode and Keras by using Horovod AllToAll Embedding as model parameters parallelism

## start train:
By default, this shell will start a train task with N workers as GPU number on local machine.

```shell
sh start.sh
```
