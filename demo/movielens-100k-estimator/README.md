# A distributed training demo for `tfra.dynamic_embedding`:

- dataset: [movielen/100k-ratings](https://www.tensorflow.org/datasets/catalog/movielens#movielens100k-ratings)
- model: DNN
- Running API: using estimator APIs

## start train:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
1. open three terminal
2. terminal chief run: sh chief.sh
3. terminal worker run: sh worker.sh
4. terminal ps run: sh ps.sh

## stop.train
run sh stop.sh