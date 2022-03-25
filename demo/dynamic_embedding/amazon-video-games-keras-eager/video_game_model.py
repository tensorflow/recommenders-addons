import tensorflow as tf
import feature

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

    # Create embedding as first layer.
    self.embedding_layer = de.keras.layers.FieldWiseEmbedding(
        self.embedding_size,
        feature.NUM_FEATURE_SLOTS + 1,
        initializer=tf.keras.initializers.RandomNormal(0.0, 0.05),
        restrict_policy=de.FrequencyRestrictPolicy,
        slot_map_fn=feature.get_category,
        bp_v2=False,
        combiner='mean',
        name='video_embedding')
    self.embedding_store = self.embedding_layer.params

    # Create dense layers.
    self.flat = tf.keras.layers.Flatten()
    self.dnn0 = tf.keras.layers.Dense(
        64,
        activation='relu',
        use_bias=True,
    )
    self.dnn1 = tf.keras.layers.Dense(
        16,
        activation='relu',
        use_bias=True,
    )
    self.dnn2 = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)

  @tf.function(
      input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
  def call(self, x):
    """
    `call` method override from `tf.keras.Model`.
    """
    x = self.embedding_layer(x)
    x = self.flat(x)
    x = self.dnn0(x)
    x = self.dnn1(x)
    preds = self.dnn2(x)
    return preds

  def get_config(self):
    return {
        'batch_size: ': self.batch_size,
        'embedding_size': self.embedding_size
    }
