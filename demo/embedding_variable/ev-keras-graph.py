import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra

tf.compat.v1.disable_eager_execution()

ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(
    lambda x: {
        "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
        "user_id": tf.strings.to_number(x["user_id"], tf.int64),
        "user_rating": x["user_rating"]
    })

tf.random.set_seed(2021)
shuffled = ratings.shuffle(100_000, seed=2021, reshuffle_each_iteration=False)

dataset_train = shuffled.take(100_000).batch(256)

iterator = tf.compat.v1.data.make_one_shot_iterator(dataset_train)
dataset_train = iterator.get_next()


class NCFModel(tf.keras.Model):

  def __init__(self):
    super(NCFModel, self).__init__()
    self.embedding_size = 32
    self.d0 = Dense(
        256,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d1 = Dense(
        64,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.d2 = Dense(
        1,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.user_embeddings = tfra.embedding_variable.EmbeddingVariable(
        name="user_dynamic_embeddings",
        ktype=tf.int64,
        embedding_dim=self.embedding_size,
        initializer=tf.keras.initializers.RandomNormal(-1, 1))
    self.movie_embeddings = tfra.embedding_variable.EmbeddingVariable(
        name="moive_dynamic_embeddings",
        embedding_dim=self.embedding_size,
        ktype=tf.int64,
        initializer=tf.keras.initializers.RandomNormal(-1, 1))
    self.loss = tf.keras.losses.MeanSquaredError()

  def call(self, batch):
    movie_id = batch["movie_id"]
    user_id = batch["user_id"]
    rating = batch["user_rating"]

    user_id_val, user_id_idx = tf.unique(user_id)
    user_id_weights = tf.nn.embedding_lookup(params=self.user_embeddings,
                                             ids=user_id_val,
                                             name="user-id-weights")
    user_id_weights = tf.gather(user_id_weights, user_id_idx)
    movie_id_val, movie_id_idx = tf.unique(movie_id)
    movie_id_weights = tf.nn.embedding_lookup(params=self.movie_embeddings,
                                              ids=movie_id_val,
                                              name="movie-id-weights")
    movie_id_weights = tf.gather(movie_id_weights, movie_id_idx)

    embeddings = tf.concat([user_id_weights, movie_id_weights], axis=1)
    dnn = self.d0(embeddings)
    dnn = self.d1(dnn)
    dnn = self.d2(dnn)
    out = tf.reshape(dnn, shape=[-1])
    loss = self.loss(rating, out)
    return loss


model = NCFModel()
loss = model(dataset_train)
optimizer = tfra.embedding_variable.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

epoch = 10
with tf.compat.v1.Session() as sess:
  for i in range(epoch):
    loss_t, _ = sess.run([loss, train_op])
    print("epoch:", i, "loss:", loss_t)
