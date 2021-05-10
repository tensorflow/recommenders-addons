# A distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/100k-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens100k-ratings)
- model: DNN
- Running API: using estimator APIs

## start train:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
1. open three terminal
2. terminal chief run: sh chief.sh train
3. terminal worker run: sh worker.sh train
4. terminal ps run: sh ps.sh train

## start export for serving:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
1. open three terminal
2. terminal chief run: sh chief.sh serving
3. terminal worker run: sh worker.sh serving
4. terminal ps run: sh ps.sh serving

## stop.train
run sh stop.sh