# A distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/1m-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens1m-ratings)
- model: DNN
- Running mode: Graph mode by using MonitoredTrainingSession

## start train:
By default, this shell will start a train task with 2 PS and 2 workers on local machine.

```shell
sh start.sh
```

## stop train:

```shell
sh stop.sh
```