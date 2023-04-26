import os
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import flags
from absl import app
from tensorflow_recommenders_addons import dynamic_embedding as de

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  #VERY IMPORTANT!

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

# Horovod: initialize Horovod.
hvd.init()

if hvd.rank() > 0:
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Horovod: pin GPU to be used to process local rank (one GPU per process)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[hvd.local_rank()],
                                         True)

# optimal performance
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
tf.config.experimental.set_synchronous_execution(False)

flags.DEFINE_string('mode', 'train', 'Select the running mode: train or test.')
flags.DEFINE_string('model_dir', 'model_dir',
                    'Directory where checkpoint stores.')
flags.DEFINE_string('export_dir', 'export_dir',
                    'Directory where model stores for inference.')
flags.DEFINE_integer('steps_per_epoch', 20000, 'Number of training steps.')
flags.DEFINE_integer('epochs', 1, 'Number of training epochs.')
flags.DEFINE_integer('embedding_size', 32,
                     'Embedding size for users and movies')
flags.DEFINE_integer('test_steps', 128, 'Embedding size for users and movies')
flags.DEFINE_integer('test_batch', 1024, 'Embedding size for users and movies')
FLAGS = flags.FLAGS

input_spec = {
    'user_id': tf.TensorSpec(shape=[
        None,
    ], dtype=tf.int64, name='user_id'),
    'movie_id': tf.TensorSpec(shape=[
        None,
    ], dtype=tf.int64, name='movie_id')
}


# Construct input function
def input_fn():
  ds = tfds.load("movielens/1m-ratings",
                 split="train",
                 data_dir="/dataset",
                 download=False)
  ids = ds.map(
      lambda x: {
          "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
          "movie_genres": tf.cast(x["movie_genres"][0], tf.int32),
          "user_id": tf.strings.to_number(x["user_id"], tf.int64),
          "user_gender": tf.cast(x["user_gender"], tf.int32),
      })
  ratings = ds.map(lambda x: {"user_rating": x["user_rating"]})
  dataset = tf.data.Dataset.zip((ids, ratings))
  shuffled = dataset.shuffle(1_000_000,
                             seed=2021,
                             reshuffle_each_iteration=False)
  dataset = shuffled.repeat(1).batch(4096)
  return dataset


class DualChannelsDeepModel(tf.keras.Model):

  def __init__(self,
               user_embedding_size=1,
               movie_embedding_size=1,
               embedding_initializer=None,
               is_training=True):

    if not is_training:
      de.enable_inference_mode()

    super(DualChannelsDeepModel, self).__init__()
    self.user_embedding_size = user_embedding_size
    self.movie_embedding_size = movie_embedding_size

    if embedding_initializer is None:
      embedding_initializer = tf.keras.initializers.Zeros()

    self.user_embedding = de.keras.layers.SquashedEmbedding(
        user_embedding_size,
        initializer=embedding_initializer,
        name='user_embedding')
    self.movie_embedding = de.keras.layers.SquashedEmbedding(
        movie_embedding_size,
        initializer=embedding_initializer,
        name='movie_embedding')

    self.dnn1 = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.dnn2 = tf.keras.layers.Dense(
        16,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.dnn3 = tf.keras.layers.Dense(
        5,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
    self.bias_net = tf.keras.layers.Dense(
        5,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))

  @tf.function
  def call(self, features):
    user_id = tf.reshape(features['user_id'], (-1, 1))
    movie_id = tf.reshape(features['movie_id'], (-1, 1))
    user_latent = self.user_embedding(user_id)
    movie_latent = self.movie_embedding(movie_id)
    latent = tf.concat([user_latent, movie_latent], axis=1)

    x = self.dnn1(latent)
    x = self.dnn2(x)
    x = self.dnn3(x)

    bias = self.bias_net(latent)
    x = 0.2 * x + 0.8 * bias
    return x


def get_dataset(batch_size=1):
  dataset = tfds.load('movielens/1m-ratings', split='train')
  features = dataset.map(
      lambda x: {
          "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
          "user_id": tf.strings.to_number(x["user_id"], tf.int64),
      })
  ratings = dataset.map(
      lambda x: tf.one_hot(tf.cast(x['user_rating'] - 1, dtype=tf.int64), 5))
  dataset = dataset.zip((features, ratings))
  dataset = dataset.shuffle(4096, reshuffle_each_iteration=False)
  if batch_size > 1:
    dataset = dataset.batch(batch_size)

  return dataset


def train():
  dataset = get_dataset(batch_size=32)
  model = DualChannelsDeepModel(FLAGS.embedding_size, FLAGS.embedding_size,
                                tf.keras.initializers.RandomNormal(0.0, 0.5))
  optimizer = tf.keras.optimizers.Adam(1E-3)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer)

  auc = tf.keras.metrics.AUC(num_thresholds=1000)
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[
                    auc,
                ])

  if os.path.exists(FLAGS.model_dir):
    model.load_weights(FLAGS.model_dir)

  model.fit(dataset, epochs=FLAGS.epochs, steps_per_epoch=FLAGS.steps_per_epoch)

  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  model.save(FLAGS.model_dir, options=save_options)


def export():
  model = DualChannelsDeepModel(FLAGS.embedding_size, FLAGS.embedding_size,
                                tf.keras.initializers.Zeros(), False)
  model.load_weights(FLAGS.model_dir)

  # Build input spec with dummy data. If the model is built with explicit
  # input specs, then no need of dummy data.
  dummy_data = {
      'user_id': tf.zeros((16,), dtype=tf.int64),
      'movie_id': tf.zeros([
          16,
      ], dtype=tf.int64)
  }
  model(dummy_data)

  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  tf.keras.models.save_model(
      model,
      FLAGS.export_dir,
      options=save_options,
      include_optimizer=False,
      signatures=model.call.get_concrete_function(input_spec))


def test():
  de.enable_inference_mode()

  dataset = get_dataset(batch_size=FLAGS.test_batch)
  model = tf.keras.models.load_model(FLAGS.export_dir)
  signature = model.signatures['serving_default']

  def get_close_or_equal_cnt(model, features, ratings):
    preds = model(features)
    preds = tf.math.argmax(preds, axis=1)
    ratings = tf.math.argmax(ratings, axis=1)
    close_cnt = tf.reduce_sum(
        tf.cast(tf.math.abs(preds - ratings) <= 1, dtype=tf.int32))
    equal_cnt = tf.reduce_sum(
        tf.cast(tf.math.abs(preds - ratings) == 0, dtype=tf.int32))
    return close_cnt, equal_cnt

  it = iter(dataset)
  for step in range(FLAGS.test_steps):
    features, ratings = it.get_next()
    close_cnt, equal_cnt = get_close_or_equal_cnt(model, features, ratings)
    print(
        f'In batch prediction, step: {step}, {close_cnt}/{FLAGS.test_batch} are closely'
        f' accurate, {equal_cnt}/{FLAGS.test_batch} are absolutely accurate.')


def main(argv):
  del argv
  if FLAGS.mode == 'train':
    train()
  elif FLAGS.mode == 'export':
    export()
  elif FLAGS.mode == 'test':
    test()
  else:
    raise ValueError('running mode only supports `train` or `test`')


if __name__ == '__main__':
  app.run(main)
