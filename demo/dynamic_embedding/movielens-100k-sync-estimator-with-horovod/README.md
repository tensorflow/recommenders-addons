# A distributed synchronous training demo based on Horovod for `tfra.dynamic_embedding`:

- dataset: [movielen/100k-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens100k-ratings)
- model: DNN
- Running API: using estimator APIs

## Requirements
- Horovod Version: 0.23.0
- OpenMPI Version: 4.1.2
## start train:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
sh train.sh

## start export for serving:
By default, this shell will start a export for serving task with 1 PS and 1 workers and 1 chief on local machine.
sh export.sh

## stop.train
run sh stop.sh