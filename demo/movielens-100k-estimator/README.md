# A distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/100k-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens100k-ratings)
- model: DNN
- Running API: using estimator APIs

# model dir info
checkpoint dir is ./ckpt
export dir is ./export_dir

## start train:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
sh train.sh

## start export for serving:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
sh export.sh

## stop.train
run sh stop.sh