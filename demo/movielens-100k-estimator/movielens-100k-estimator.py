import json
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense

import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra

from absl import app
from absl import flags

flags.DEFINE_string('model_dir', "./ckpt", 'export_dir')
flags.DEFINE_string('export_dir', "./export_dir", 'export_dir')
flags.DEFINE_string('mode', "train", 'train or export')

FLAGS = flags.FLAGS


def input_fn():
  ratings = tfds.load("movielens/100k-ratings", split="train")
  ratings = ratings.map(
      lambda x: {
          "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
          "user_id": tf.strings.to_number(x["user_id"], tf.int64),
          "user_rating": x["user_rating"]
      })
  shuffled = ratings.shuffle(1_000_000,
                             seed=2021,
                             reshuffle_each_iteration=False)
  dataset = shuffled.batch(256)
  return dataset


def model_fn(features, labels, mode, params):
  embedding_size = 32
  movie_id = features["movie_id"]
  user_id = features["user_id"]
  rating = features["user_rating"]

  is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False

  if params["run_status"] == 'online':
    tfra.dynamic_embedding.enable_inference_mode()

  if is_training:
    ps_list = [
        "/job:ps/replica:0/task:{}/CPU:0".format(i)
        for i in range(params["ps_num"])
    ]
    initializer = tf.keras.initializers.RandomNormal(-1, 1)
  else:
    ps_list = [
        "/job:localhost/replica:0/task:0/CPU:0" for _ in range(params["ps_num"])
    ]
    initializer = tf.keras.initializers.Zeros()

  user_embeddings = tfra.dynamic_embedding.get_variable(
      name="user_dynamic_embeddings",
      dim=embedding_size,
      devices=ps_list,
      initializer=initializer)
  movie_embeddings = tfra.dynamic_embedding.get_variable(
      name="moive_dynamic_embeddings",
      dim=embedding_size,
      devices=ps_list,
      initializer=initializer)

  user_id_val, user_id_idx = tf.unique(tf.concat(user_id, axis=0))
  user_id_weights, user_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup(
      params=user_embeddings,
      ids=user_id_val,
      name="user-id-weights",
      return_trainable=True)
  user_id_weights = tf.gather(user_id_weights, user_id_idx)

  movie_id_val, movie_id_idx = tf.unique(tf.concat(movie_id, axis=0))
  movie_id_weights, movie_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup(
      params=movie_embeddings,
      ids=movie_id_val,
      name="movie-id-weights",
      return_trainable=True)
  movie_id_weights = tf.gather(movie_id_weights, movie_id_idx)

  embeddings = tf.concat([user_id_weights, movie_id_weights], axis=1)
  d0 = Dense(256,
             activation='relu',
             kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
             bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
  d1 = Dense(64,
             activation='relu',
             kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
             bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
  d2 = Dense(1,
             kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
             bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
  dnn = d0(embeddings)
  dnn = d1(dnn)
  dnn = d2(dnn)
  out = tf.reshape(dnn, shape=[-1])
  loss = tf.keras.losses.MeanSquaredError()(rating, out)
  predictions = {"out": out}

  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)
    train_op = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions_for_net = {"out": out}
    export_outputs = {
        "predict_export_outputs":
            tf.estimator.export.PredictOutput(outputs=predictions_for_net)
    }
    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions_for_net,
                                      export_outputs=export_outputs)


def train(model_dir, ps_num, run_status):
  model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                        save_summary_steps=100,
                                        save_checkpoints_steps=100,
                                        save_checkpoints_secs=None,
                                        keep_checkpoint_max=2)

  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     params={
                                         "run_status": run_status,
                                         "ps_num": ps_num
                                     },
                                     config=model_config)

  train_spec = tf.estimator.TrainSpec(input_fn=input_fn)

  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def serving_input_receiver_dense_fn():
  input_spec = {
      "movie_id": tf.constant([1], tf.int64),
      "user_id": tf.constant([1], tf.int64),
      "user_rating": tf.constant([1.0], tf.float32)
  }
  return tf.estimator.export.build_raw_serving_input_receiver_fn(input_spec)


def export_for_serving(model_dir, export_dir, ps_num, run_status):
  model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                        save_summary_steps=100,
                                        save_checkpoints_steps=100,
                                        save_checkpoints_secs=None)

  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     params={
                                         "run_status": run_status,
                                         "ps_num": ps_num
                                     },
                                     config=model_config)

  estimator.export_saved_model(export_dir, serving_input_receiver_dense_fn())


def main(argv):
  del argv
  tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
  task_name = tf_config.get('task', {}).get('type')
  task_idx = tf_config.get('task', {}).get('index')

  print("tf_config is {}".format(tf_config))

  ps_num = len(tf_config["cluster"]["ps"])

  if FLAGS.mode == "train":
    train(FLAGS.model_dir, ps_num, run_status="offline")
  if FLAGS.mode == "serving" and task_name == "chief" and int(task_idx) == 0:
    export_for_serving(FLAGS.model_dir,
                       FLAGS.export_dir,
                       ps_num,
                       run_status="online")


if __name__ == "__main__":
  app.run(main)
