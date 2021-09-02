# A distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/100k-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens100k-ratings)
- model: DNN
- Running API: using estimator APIs

## About TFRA-redis configuration
For more details, please visit file 'tensorflow_recommenders_addons/dynamic_embedding/core/kernels/redis_impl/README.md'

## start train:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
But if you only use TFRA-redis backend, we actually will not save model parameters in PS.
sh train.sh

## start export for serving:
By default, this shell will start a export for serving task with 1 PS and 1 workers and 1 chief on local machine.
But if you only use TFRA-redis backend, we actually will not save model parameters in PS.
sh export.sh

## stop.train
run sh stop.sh