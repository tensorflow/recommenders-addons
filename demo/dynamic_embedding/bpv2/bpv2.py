#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os, json
import numpy as np
import tensorflow as tf
import tensorflow_recommenders_addons as tfra

from absl import app

batch_size = 128
vocab_size = 10000
embed_size = 64


def make_word_index():
  word_index = tf.keras.datasets.imdb.get_word_index()
  word_index = {k: (v + 3) for k, v in word_index.items()}
  word_index["<PAD>"] = 0
  word_index["<START>"] = 1
  word_index["<UNK>"] = 2  # unknown
  word_index["<UNUSED>"] = 3
  reverse_word_index = dict([
      (value, key) for (key, value) in word_index.items()
  ])
  return word_index, reverse_word_index


def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])


word_index, reverse_word_index = make_word_index()


def get_data():
  (train_data, train_labels), (test_data,
                               test_labels) = tf.keras.datasets.imdb.load_data(
                                   num_words=10000, path='imdb-0')
  train_data = tf.keras.preprocessing.sequence.pad_sequences(
      train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
  test_data = tf.keras.preprocessing.sequence.pad_sequences(
      test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
  x_val = train_data[:1024]
  x_train = train_data[:1024]
  y_val = train_labels[:1024]
  y_train = train_labels[:1024]
  return x_train, y_train, x_val, y_val


x_train, y_train, x_val, y_val = get_data()


def input_fn_train():
  dataset = tf.data.Dataset.from_tensor_slices(({'x': x_train}, y_train))
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  return dataset


def input_fn_val():
  dataset = tf.data.Dataset.from_tensor_slices(({'x': x_val}, y_val))
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  return dataset


def model_fn(features, labels, mode):
  x = features['x']
  x = tf.reshape(x, [-1])
  w = tfra.dynamic_embedding.get_variable(
      name='w',
      devices=["/job:ps/replica:0/task:0/CPU:0"],
      initializer=tf.random_normal_initializer(0, 0.5),
      dim=embed_size,
      bp_v2=True,  # this is the only thing you need to do to enable bpv2
      key_dtype=tf.int32)
  e = tfra.dynamic_embedding.embedding_lookup_unique(params=w, ids=x, name='a')
  e = tf.reshape(e, [-1, 256, embed_size])

  embmean = tf.reduce_mean(e, axis=1)
  fc = tf.compat.v1.layers.dense(embmean, 16, activation=tf.nn.relu)
  logits = tf.compat.v1.layers.dense(fc, 2, activation=None)
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
  }

  y = tf.one_hot(tf.cast(labels, tf.int32), 2, 1, 0)
  loss = tf.compat.v1.losses.softmax_cross_entropy(y, logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    opt = tf.compat.v1.train.AdamOptimizer(0.01)
    opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(opt)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  else:
    eval_metric_ops = {
        "accuracy":
            tf.metrics.accuracy(labels=labels,
                                predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(argv):
  del argv
  tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')

  config = tf.estimator.RunConfig(save_checkpoints_steps=None,
                                  save_checkpoints_secs=tf.int64.max,
                                  model_dir=None,
                                  log_step_count_steps=100)
  classifier = tf.estimator.Estimator(model_fn=model_fn, config=config)
  tf.estimator.train_and_evaluate(
      classifier,
      train_spec=tf.estimator.TrainSpec(input_fn=input_fn_train,
                                        max_steps=3000),
      eval_spec=tf.estimator.EvalSpec(input_fn=input_fn_val, steps=1000))


if __name__ == "__main__":
  app.run(main)
