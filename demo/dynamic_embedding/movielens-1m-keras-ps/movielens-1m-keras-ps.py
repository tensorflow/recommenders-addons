import os
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import flags
from absl import app
from tensorflow_recommenders_addons import dynamic_embedding as de

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'ps_list', "localhost:2220, localhost:2221",
    'ps_list: to be a comma seperated string, '
    'like "localhost:2220, localhost:2220"')
flags.DEFINE_string(
    'worker_list', "localhost:2231",
    'worker_list: to be a comma seperated string, '
    'like "localhost:2231, localhost:2232"')
flags.DEFINE_string('chief', "localhost:2230", 'chief: like "localhost:2230"')
flags.DEFINE_string('task_mode', "worker",
                    'runninig_mode: ps or worker or chief.')
flags.DEFINE_integer('task_id', 0, 'task_id: used for allocating samples.')

input_spec = {
    'user_id': tf.TensorSpec(shape=[
        None,
    ], dtype=tf.int64, name='user_id'),
    'movie_id': tf.TensorSpec(shape=[
        None,
    ], dtype=tf.int64, name='movie_id')
}


class DualChannelsDeepModel(tf.keras.Model):

  def __init__(self,
               devices=[],
               user_embedding_size=1,
               movie_embedding_size=1,
               embedding_initializer=None,
               is_training=True):

    if not is_training:
      de.enable_inference_mode()

    super(DualChannelsDeepModel, self).__init__()
    self.user_embedding_size = user_embedding_size
    self.movie_embedding_size = movie_embedding_size
    self.devices = devices

    if embedding_initializer is None:
      embedding_initializer = tf.keras.initializers.Zeros()

    self.user_embedding = de.keras.layers.SquashedEmbedding(
        user_embedding_size,
        initializer=embedding_initializer,
        devices=self.devices,
        name='user_embedding')
    self.movie_embedding = de.keras.layers.SquashedEmbedding(
        movie_embedding_size,
        initializer=embedding_initializer,
        devices=self.devices,
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


class Runner():

  def __init__(self, strategy, train_bs, test_bs, epochs, steps_per_epoch,
               model_dir, export_dir):
    self.strategy = strategy
    self.num_worker = strategy._num_workers
    self.num_ps = strategy._num_ps
    self.ps_devices = [
        "/job:ps/replica:0/task:{}/device:CPU:0".format(idx)
        for idx in range(self.num_ps)
    ]
    self.embedding_size = 32
    self.train_bs = train_bs
    self.test_bs = test_bs
    self.epochs = epochs
    self.steps_per_epoch = steps_per_epoch
    self.model_dir = model_dir
    self.export_dir = export_dir

  def get_dataset(self, batch_size=1):
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

  def train(self):
    dataset = self.get_dataset(batch_size=self.train_bs)
    dataset = self.strategy.experimental_distribute_dataset(dataset)
    with self.strategy.scope():
      model = DualChannelsDeepModel(
          self.ps_devices, self.embedding_size, self.embedding_size,
          tf.keras.initializers.RandomNormal(0.0, 0.5))
      optimizer = tf.keras.optimizers.Adam(1E-3)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

      auc = tf.keras.metrics.AUC(num_thresholds=1000)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[
                      auc,
                  ])

    if self.model_dir:
      if os.path.exists(self.model_dir):
        model.load_weights(self.model_dir)

    model.fit(dataset, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch)

    if self.model_dir:
      save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
      model.save(self.model_dir, options=save_options)

  def export(self):
    with self.strategy.scope():
      model = DualChannelsDeepModel(
          self.ps_devices, self.embedding_size, self.embedding_size,
          tf.keras.initializers.RandomNormal(0.0, 0.5))

    def save_spec():
      if hasattr(model, 'save_spec'):
        return model.save_spec()
      else:
        arg_specs = list()
        kwarg_specs = dict()
        for i in model.inputs:
          arg_specs.append(i.type_spec)
        return [arg_specs], kwarg_specs

    @tf.function
    def serve(*args, **kwargs):
      return model(*args, **kwargs)

    save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

    # Only save the calculation graph
    from tensorflow.python.saved_model import save as tf_save
    K.clear_session()
    de.enable_inference_mode()
    # Overwrite saved_model.pb file with save_and_return_nodes function to rewrite the calculation graph
    tf_save.save_and_return_nodes(obj=model,
                                  export_dir=self.export_dir,
                                  signatures={
                                      'serving_default':
                                          serve.get_concrete_function(
                                              *arg_specs, **kwarg_specs)
                                  },
                                  options=save_options,
                                  experimental_skip_checkpoint=True)

  def test(self):
    de.enable_inference_mode()

    dataset = self.get_dataset(batch_size=self.test_bs)
    dataset = self.strategy.experimental_distribute_dataset(dataset)
    with self.strategy.scope():
      model = tf.keras.models.load_model(self.export_dir)
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
    for step in range(self.test_steps):
      features, ratings = it.get_next()
      close_cnt, equal_cnt = get_close_or_equal_cnt(model, features, ratings)
      print(
          f'In batch prediction, step: {step}, {close_cnt}/{self.test_bs} are closely'
          f' accurate, {equal_cnt}/{self.test_bs} are absolutely accurate.')


def start_chief(config):
  print("chief config", config)

  cluster_spec = tf.train.ClusterSpec(config["cluster"])
  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, task_type="chief", task_id=0)
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver)
  runner = Runner(strategy=strategy,
                  train_bs=64,
                  test_bs=1,
                  epochs=2,
                  steps_per_epoch=10,
                  model_dir=None,
                  export_dir=None)
  runner.train()


def start_worker(task_id, config):
  print("worker config", config)
  cluster_spec = tf.train.ClusterSpec(config["cluster"])

  sess_config = tf.compat.v1.ConfigProto()
  sess_config.intra_op_parallelism_threads = 4
  sess_config.inter_op_parallelism_threads = 4
  server = tf.distribute.Server(cluster_spec,
                                config=sess_config,
                                protocol='grpc',
                                job_name="worker",
                                task_index=task_id)
  server.join()


def start_ps(task_id, config):
  print("ps config", config)
  cluster_spec = tf.train.ClusterSpec(config["cluster"])

  sess_config = tf.compat.v1.ConfigProto()
  sess_config.intra_op_parallelism_threads = 4
  sess_config.inter_op_parallelism_threads = 4
  server = tf.distribute.Server(cluster_spec,
                                config=sess_config,
                                protocol='grpc',
                                job_name="ps",
                                task_index=task_id)
  server.join()


def main(argv):
  ps_list = FLAGS.ps_list.replace(' ', '').split(',')
  worker_list = FLAGS.worker_list.replace(' ', '').split(',')
  task_mode = FLAGS.task_mode
  task_id = FLAGS.task_id

  print('ps_list: ', ps_list)
  print('worker_list: ', worker_list)

  cluster_config = {
      "cluster": {
          "chief": [FLAGS.chief],
          "ps": ps_list,
          "worker": worker_list
      }
  }
  if task_mode == 'chief':
    start_chief(cluster_config)
  elif task_mode == 'ps':
    start_ps(task_id, cluster_config)
  elif task_mode == 'worker':
    start_worker(task_id, cluster_config)
  else:
    print('invalid task_mode. Options include "ps" and "worker".')
    sys.exit(1)


if __name__ == "__main__":
  tf.compat.v1.app.run()
