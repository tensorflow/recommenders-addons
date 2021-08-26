import tensorflow.compat.v1 as tf
import time
import random
from tensorflow.python.framework import ops
import tensorflow_recommenders_addons as tfra

tf.disable_v2_behavior()

tf.app.flags.DEFINE_string("ps_hosts", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_string("worker_hosts", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("buckets", None, "oss buckets")
tf.app.flags.DEFINE_integer("interval", 60, "interval")
tf.app.flags.DEFINE_string("protocol", "grpc", "interval")

FLAGS = tf.app.flags.FLAGS


#tf.logging.set_verbosity(tf.logging.DEBUG)
def model_fn(ids, labels):
  #embedding = tf.get_embedding_variable("var_dist", embedding_dim=24, steps_to_live = 4000, #steps_to_live_l2reg=6000, l2reg_theta=0.01, #'''(10 * 1024 * 1024) * 2''',  steps_to_live_l2reg = 1024*1024, l2reg_theta=0.01, l2reg_lambda=0.01, initializer=tf.ones_initializer, partitioner=tf.fixed_size_partitioner(num_shards=4))

  embedding = tfra.embedding_variable.EmbeddingVariable(
      name="var_dist",
      embedding_dim=24,
      ktype=tf.int64,
      initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))

  values = tf.nn.embedding_lookup(embedding, ids)

  features = tf.reshape(values, shape=[1024, 24])

  # W = tf.Variable(tf.zeros([24, 10]), initializer=tf.initializers.random_uniform)
  # b = tf.Variable(tf.zeros([10]), initializer=tf.initializers.random_uniform)
  W = tf.Variable(tf.zeros([24, 10]))
  b = tf.Variable(tf.zeros([10]))

  pred = tf.matmul(features, W) + b

  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))

  global_step = tf.train.get_or_create_global_step()
  optimizer = tfra.embedding_variable.AdagradOptimizer(
      learning_rate=0.001).minimize(loss, global_step)

  tf.summary.scalar("loss", loss)
  return loss, optimizer, features


def train(task_index, cluster, is_chief, target, buckets):
  x = tf.placeholder(tf.int64, [1024])
  y_ = tf.placeholder(tf.float32, [None, 10])
  one_hot_label = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  labels = []
  for i in range(1024):
    labels.append(one_hot_label)

  available_worker_device = "/job:worker/task:%d" % (task_index)
  with tf.device(
      tf.train.replica_device_setter(worker_device=available_worker_device,
                                     cluster=cluster)):
    loss, opt, emb = model_fn(x, y_)
    local_step = 1
    config = tf.ConfigProto(inter_op_parallelism_threads=7)
    with tf.train.MonitoredTrainingSession(master=target,
                                           is_chief=is_chief,
                                           hooks=[],
                                           save_checkpoint_secs=60,
                                           checkpoint_dir=buckets,
                                           config=config) as mon_sess:
      #print(ops.get_default_graph().as_graph_def())
      while True:
        fe = []
        for i in range(1024):
          fe.append(random.randint(1, 100000000))
        l, m, emb2 = mon_sess.run([loss, opt, emb],
                                  feed_dict={
                                      x: fe,
                                      y_: labels
                                  })
        local_step += 1
        if local_step % 100 == 0:
          print(l)
        #time.sleep(FLAGS.interval)


def main(argv):
  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)
  is_chief = FLAGS.task_index == 0

  # construct the servers
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

  config = tf.ConfigProto(inter_op_parallelism_threads=8)
  server = tf.distribute.Server(cluster,
                                job_name=FLAGS.job_name,
                                protocol=FLAGS.protocol,
                                task_index=FLAGS.task_index,
                                config=config)

  # join the ps server
  if FLAGS.job_name == "ps":
    server.join()

  # start the training
  print("start trainig")
  print(FLAGS.buckets)
  train(task_index=FLAGS.task_index,
        cluster=cluster,
        is_chief=is_chief,
        target=server.target,
        buckets=FLAGS.buckets)


if __name__ == "__main__":
  tf.app.run()
