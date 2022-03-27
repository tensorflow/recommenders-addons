# movielens-1m-keras

This is a demo for using keras API to build a recommender system to predict the user's rating on a movie, with support of [dynamic_embedding](https://github.com/tensorflow/recommenders-addons/blob/master/docs/api_docs/tfra/dynamic_embedding.md).

The training and testing data are from [movielens](https://www.tensorflow.org/datasets/catalog/movielens) dataset.

```bash
# train
python movielens-1m-keras.py --mode=train --epochs=1 --steps_per_epoch=20000

# export model for inference
python movielens-1m-keras.py --mode=export

# Run test
python movielens-1m-keras.py --mode=test --test_steps=100 --test_batch=1024
```


