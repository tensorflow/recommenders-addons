import tensorflow as tf
import tensorflow_datasets as tfds
import feature
import numpy as np
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow_recommenders_addons import dynamic_embedding as de


class VideoGameDnn(tf.keras.Model):
  """
  A DNN model for training. Sequential `Dense` layers are piled
  on a embedding layer. Adam optimizer is used here.
  """

  def __init__(self, batch_size=1, embedding_size=1):
    super(VideoGameDnn, self).__init__()
    self.batch_size = batch_size
    self.embedding_size = embedding_size

    # Create embedding variable by `tfra.dynamic_embedding` API.
    self.embedding_store = de.get_variable(
        'video_feature_embedding',
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        dim=embedding_size,
        devices=['/CPU:0'],
        initializer=tf.keras.initializers.RandomNormal(-0.1, 0.1),
        trainable=True,
        restrict_policy=de.TimestampRestrictPolicy)

    # Create dense layers.
    self.dnn0 = tf.keras.layers.Dense(
        64,
        activation='relu',
        use_bias=True,
        bias_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.L1(0.01),
        bias_regularizer=tf.keras.regularizers.L1(0.02),
    )
    self.dnn1 = tf.keras.layers.Dense(
        16,
        activation='relu',
        use_bias=True,
        bias_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.L1(0.01),
        bias_regularizer=tf.keras.regularizers.L1(0.02),
    )
    self.dnn2 = tf.keras.layers.Dense(1, use_bias=False)
    self.embedding_trainables = []

    # Create optimizer.
    self.optmz = de.DynamicEmbeddingOptimizer(tf.keras.optimizers.Adam(0.01))

    # Metric observer.
    self._auc = tf.metrics.AUC()

  @staticmethod
  def lookup_sparse_weights(model, features, name='lookup_sparse_weights'):
    if not isinstance(model, VideoGameDnn):
      raise TypeError('Only serve VideoGameDnn model.')
    embed, tw = de.embedding_lookup_unique(model.embedding_store,
                                           features,
                                           name=name,
                                           return_trainable=True)
    if not model.embedding_trainables:
      model.embedding_trainables.append(tw)
    return embed

  @staticmethod
  def embedding_fn(model, x):
    """
    Funcion to lookup the embedding. It was made static because we
    need to reuse it in somewhere else.
    """
    batch_size = tf.shape(x)[0]
    x = tf.reshape(x, (-1,))
    embed_w = model.lookup_sparse_weights(model, x)
    embeds = []
    for name, encoder in feature.FEATURE_AND_ENCODER.items():
      mask = encoder.match_category(x)
      indices = tf.where(mask)
      categorical_w = tf.gather(embed_w, indices)
      categorical_w = tf.reshape(categorical_w,
                                 (batch_size, -1, model.embedding_size))
      categorical_w = tf.reduce_sum(categorical_w, axis=1)
      embeds.append(
          tf.reshape(categorical_w, (batch_size, model.embedding_size)))
    embeds = tf.concat(embeds, axis=1)
    return embeds

  def dnn_net(self, x):
    out = x
    out = self.dnn0(x)
    out = self.dnn1(out)
    out = self.dnn2(out)
    return out

  def call(self, x):
    """
    `call` method override whom in `tf.keras.Model`.
    """
    embed = self.embedding_fn(self, x)
    logits = self.dnn_net(embed)
    preds = tf.nn.sigmoid(logits)
    return preds

  def train(self, features, labels):
    """
    Train model with input features and labels. Here it uses
    `GradientTape` for clipping the gradients to avoid explosion
    of parameter. `optimizer.minimize` is also supported.
    """
    with tf.GradientTape() as tape:
      preds = self(features)
      preds = tf.reshape(preds, (-1))
      labels = tf.cast(labels, dtype=tf.float32)
      loss = tf.keras.losses.MeanSquaredError()(preds, labels)
      grads = tape.gradient(loss, self.trainable_variables)
      grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
    self.optmz.apply_gradients(zip(grads, self.trainable_variables))
    self._auc.update_state(labels, preds)
    return loss, self._auc.result()


class VideoGameDnnInference(tf.keras.Model):
  """
  Model built for inference. It copies dense and sparse variables
  from a training model, and slightly removes the `embedding_lookup`
  by replacing `lookup_sparse_weights` method, because we don't need
  the local trainables in inference.
  """

  def __init__(self, model):
    super(VideoGameDnnInference, self).__init__()
    self.embedding_size = model.embedding_size
    self.embedding_store = model.embedding_store
    self.embedding_fn = model.embedding_fn
    self.dnn0 = model.dnn0
    self.dnn1 = model.dnn1
    self.dnn2 = model.dnn2
    self.dnn_net = model.dnn_net

  @staticmethod
  def lookup_sparse_weights(model, features):
    return model.embedding_store.lookup(features)

  def call(self, x):
    x = self.embedding_fn(self, x)
    out = self.dnn_net(x)
    return tf.nn.sigmoid(out)
