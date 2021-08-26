# A [dynamic_embedding](https://github.com/tensorflow/recommenders-addons/blob/master/docs/api_docs/tfra/dynamic_embedding.md) demo based on [amazon_us_reviews/Digital_Video_Games_v1_00](https://www.tensorflow.org/datasets/catalog/amazon_us_reviews)

We use [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) APIs to train a model and predict whether if the digital video games are purchased verifiedly.

In the demo, we expect to show how to use [dynamic_embedding.Variable](https://github.com/tensorflow/recommenders-addons/blob/master/docs/api_docs/tfra/dynamic_embedding/Variable.md) to represent embedding layers, train the model with growth of `Variable`, and restrict the `Variable` when it grows too large.


## Start training and export model:
```bash
python main.py --mode=train  --export_dir="export"
```
It will produce a model to `export_dir`.

## Inference:
```bash
python main.py --mode=test  --export_dir="export" --batch_size=10
```
It will print accuracy to the prediction on verified purchase of the digital video games.