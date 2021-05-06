import os, sys
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds
import tensorflow_recommenders_addons as tfra

tf.compat.v1.disable_v2_behavior()

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'ps_list', "localhost:2220, localhost:2221",
    'ps_list: to be a comma seperated string, '
    'like "localhost:2220, localhost:2220"')
flags.DEFINE_string(
    'worker_list', "localhost:2230",
    'worker_list: to be a comma seperated string, '
    'like "localhost:2230, localhost:2231"')
flags.DEFINE_string('task_mode', "worker", 'runninig_mode: ps or worker.')
flags.DEFINE_integer('task_id', 0, 'task_id: used for allocating samples.')
flags.DEFINE_bool('is_chief', False, ''
                  ': If true, will run init_op and save/restore.')


class Trainer():

  def __init__(self, worker_id, worker_num, ps_num, batch_size, ckpt_dir=None):
    self.embedding_size = 32
    self.worker_id = worker_id
    self.worker_num = worker_num
    self.batch_size = batch_size
    self.devices = [
        "/job:ps/replica:0/task:{}".format(idx) for idx in range(ps_num)
    ]
    self.ckpt_dir = ckpt_dir
    if self.ckpt_dir:
      os.makedirs(os.path.split(self.ckpt_dir)[0], exist_ok=True)

  def read_batch(self):
    split_size = int(100 / self.worker_num)
    split_start = split_size * self.worker_id
    split = 'train[{}%:{}%]'.format(split_start, split_start + split_size - 1)
    print("dataset split, worker{}: {}".format(self.worker_id, split))
    ratings = tfds.load("movielens/1m-ratings", split=split)
    ratings = ratings.map(
        lambda x: {
            "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
            "user_id": tf.strings.to_number(x["user_id"], tf.int64),
            "user_rating": x["user_rating"]
        })
    shuffled = ratings.shuffle(1_000_000,
                               seed=2021,
                               reshuffle_each_iteration=False)
    dataset_train = shuffled.batch(self.batch_size)
    train_iter = tf.compat.v1.data.make_initializable_iterator(dataset_train)
    return train_iter

  def build_graph(self, batch):
    movie_id = batch["movie_id"]
    user_id = batch["user_id"]
    rating = batch["user_rating"]

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
    user_embeddings = tfra.dynamic_embedding.get_variable(
        name="user_dynamic_embeddings",
        dim=self.embedding_size,
        devices=self.devices,
        initializer=tf.keras.initializers.RandomNormal(-1, 1))
    movie_embeddings = tfra.dynamic_embedding.get_variable(
        name="moive_dynamic_embeddings",
        dim=self.embedding_size,
        devices=self.devices,
        initializer=tf.keras.initializers.RandomNormal(-1, 1))

    user_id_val, user_id_idx = tf.unique(user_id)
    user_id_weights, user_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup(
        params=user_embeddings,
        ids=user_id_val,
        name="user-id-weights",
        return_trainable=True)
    user_id_weights = tf.gather(user_id_weights, user_id_idx)

    movie_id_val, movie_id_idx = tf.unique(movie_id)
    movie_id_weights, movie_id_trainable_wrapper = tfra.dynamic_embedding.embedding_lookup(
        params=movie_embeddings,
        ids=movie_id_val,
        name="movie-id-weights",
        return_trainable=True)
    movie_id_weights = tf.gather(movie_id_weights, movie_id_idx)

    embeddings = tf.concat([user_id_weights, movie_id_weights], axis=1)
    dnn = d0(embeddings)
    dnn = d1(dnn)
    dnn = d2(dnn)
    predict = tf.reshape(dnn, shape=[-1])
    loss = tf.keras.losses.MeanSquaredError()(rating, predict)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)
    update = optimizer.minimize(
        loss, global_step=tf.compat.v1.train.get_or_create_global_step())
    return {
        "update": update,
        "predict": predict,
        "loss": loss,
        "size": user_embeddings.size(),
    }


def start_worker(worker_id, config):
  print("worker config", config)
  ps_list = config['cluster']['ps']
  worker_list = config['cluster']['worker']

  num_ps_tasks = len(ps_list)
  num_worker_tasks = len(worker_list)
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.intra_op_parallelism_threads = 1
  sess_config.inter_op_parallelism_threads = 1
  cluster = tf.train.ClusterSpec(config['cluster'])
  server = tf.distribute.Server(cluster,
                                protocol="grpc",
                                job_name="worker",
                                task_index=worker_id,
                                config=sess_config)
  with tf.compat.v1.device("/job:worker/replica:0/task:{}".format(worker_id)):
    trainer = Trainer(worker_id=worker_id,
                      worker_num=num_worker_tasks,
                      ps_num=num_ps_tasks,
                      batch_size=64,
                      ckpt_dir=None)
    train_iter = trainer.read_batch()
    train_data = train_iter.get_next()

  device_setter = tf.compat.v1.train.replica_device_setter(
      ps_tasks=num_ps_tasks,
      worker_device="/job:worker/replica:0/task:{}".format(worker_id),
      ps_device="/job:ps")

  with tf.compat.v1.device(device_setter):
    outputs = trainer.build_graph(train_data)

  with tf.compat.v1.train.MonitoredTrainingSession(
      master=server.target,
      is_chief=FLAGS.is_chief,
      checkpoint_dir=trainer.ckpt_dir if FLAGS.is_chief else None,
      config=sess_config,
  ) as sess:
    sess.run([train_iter.initializer])

    step = 0
    while True:
      step += 1
      try:
        _, _loss, _pred = sess.run(
            [outputs["update"], outputs["loss"], outputs["predict"]])

        _size = sess.run(outputs["size"])
        if step % 100 == 0:
          print("[worker{}]step{}:\tloss={:.4f}\t size={}".format(
              worker_id, step, float(_loss), _size))
      except tf.errors.OutOfRangeError:
        print("[worker{}]no more data!".format(worker_id))
        break


def start_ps(task_id, config):
  print("ps config", config)
  cluster = tf.train.ClusterSpec(config["cluster"])

  sess_config = tf.compat.v1.ConfigProto()
  sess_config.intra_op_parallelism_threads = 1
  sess_config.inter_op_parallelism_threads = 1
  server = tf.distribute.Server(cluster,
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

  cluster_config = {"cluster": {"ps": ps_list, "worker": worker_list}}
  if task_mode == 'ps':
    start_ps(task_id, cluster_config)
  elif task_mode == 'worker':
    start_worker(task_id, cluster_config)
  else:
    print('invalid task_mode. Options include "ps" and "worker".')
    sys.exit(1)


if __name__ == "__main__":
  tf.compat.v1.app.run()
