import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import flags
from absl import app
from tensorflow_recommenders_addons import dynamic_embedding as de
try:
  from tensorflow.keras.legacy.optimizers import Adam
except:
  from tensorflow.keras.optimizers import Adam

import horovod.tensorflow as hvd

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
    'user_id':
        tf.TensorSpec(shape=[
            None,
            1,
        ], dtype=tf.int64, name='user_id'),
    'user_gender':
        tf.TensorSpec(shape=[
            None,
            1,
        ], dtype=tf.int32, name='user_gender'),
    'user_occupation_label':
        tf.TensorSpec(shape=[
            None,
            1,
        ],
                      dtype=tf.int32,
                      name='user_occupation_label'),
    'bucketized_user_age':
        tf.TensorSpec(shape=[
            None,
            1,
        ],
                      dtype=tf.int32,
                      name='bucketized_user_age'),
    'movie_id':
        tf.TensorSpec(shape=[
            None,
            1,
        ], dtype=tf.int64, name='movie_id'),
    'movie_genres':
        tf.TensorSpec(shape=[
            None,
            1,
        ], dtype=tf.int32, name='movie_genres'),
    'timestamp':
        tf.TensorSpec(shape=[
            None,
            1,
        ], dtype=tf.int32, name='timestamp')
}

feature_info_spec = {
    'movie_id': {
        'code': 101,
        'dtype': tf.int64,
        'dim': 1,
        'ptype': 'sparse_cpu',
        'input_tensor': None,
        'pretreated_tensor': None
    },
    'movie_genres': {
        'code': 102,
        'dtype': tf.int32,
        'dim': 1,
        'ptype': 'normal_gpu',
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'user_id': {
        'code': 103,
        'dtype': tf.int64,
        'dim': 1,
        'ptype': 'sparse_cpu',
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'user_gender': {
        'code': 104,
        'dtype': tf.int32,
        'dim': 1,
        'ptype': 'normal_gpu',
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'user_occupation_label': {
        'code': 105,
        'dtype': tf.int32,
        'dim': 1,
        'ptype': 'normal_gpu',
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'bucketized_user_age': {
        'code': 106,
        'dtype': tf.int32,
        'dim': 1,
        'ptype': 'normal_gpu',
        'input_tensor': None,
        'pretreated_tensor': None,
        'boundaries': [i for i in range(0, 100, 10)],
    },
    'timestamp': {
        'code': 107,
        'dtype': tf.int32,
        'dim': 1,
        'ptype': 'normal_gpu',
        'input_tensor': None,
        'pretreated_tensor': None,
    }
}


# Auxiliary function of GPU hash table combined query, recording which input is a vector feature embedding to be marked as a special treatment (usually an average) after embedding layer.
def embedding_inputs_concat(input_tensors, input_dims):
  tmp_sum = 0
  input_split_dims = []
  input_is_sequence_feature = []
  for tensors, dim in zip(input_tensors, input_dims):
    if tensors.get_shape().ndims != 2:
      raise ("Please make sure dimension size of all input tensors is 2!")
    if dim == 1:
      tmp_sum = tmp_sum + 1
    elif dim > 1:
      if tmp_sum > 0:
        input_split_dims.append(tmp_sum)
        input_is_sequence_feature.append(False)
      input_split_dims.append(dim)
      input_is_sequence_feature.append(True)
      tmp_sum = 0
    else:
      raise ("dim must >= 1, which is {}".format(dim))
  if tmp_sum > 0:
    input_split_dims.append(tmp_sum)
    input_is_sequence_feature.append(False)
  input_tensors_concat = tf.keras.layers.Concatenate(axis=1)(input_tensors)
  return (input_tensors_concat, input_split_dims, input_is_sequence_feature)


# After get the results of table combined query, we need to extract the vector features separately by split operator for a special treatment (usually an average).
def embedding_out_split(embedding_out_concat, input_split_dims):
  embedding_out = list()
  embedding_out.extend(
      tf.split(embedding_out_concat, input_split_dims,
               axis=1))  # (feature_combin_num, (batch, dim, emb_size))
  assert (len(input_split_dims) == len(embedding_out))
  return embedding_out


class Bucketize(tf.keras.layers.Layer):

  def __init__(self, boundaries, **kwargs):
    self.boundaries = boundaries
    super(Bucketize, self).__init__(**kwargs)

  def build(self, input_shape):
    # Be sure to call this somewhere!
    super(Bucketize, self).build(input_shape)

  def call(self, x, **kwargs):
    return tf.raw_ops.Bucketize(input=x, boundaries=self.boundaries)

  def get_config(self,):
    config = {'boundaries': self.boundaries}
    base_config = super(Bucketize, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ChannelEmbeddingLayers(tf.keras.layers.Layer):

  def __init__(self,
               name='',
               dense_embedding_size=1,
               sparse_embedding_size=1,
               embedding_initializer=tf.keras.initializers.Zeros(),
               mpi_size=1,
               mpi_rank=0):

    super(ChannelEmbeddingLayers, self).__init__()

    self.gpu_device = ["GPU:0"]
    self.cpu_device = ["CPU:0"]

    # The saver parameter of kv_creator saves the K-V in the hash table into a separate KV file.
    self.kv_creator = de.CuckooHashTableCreator(
        saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank))

    self.dense_embedding_layer = de.keras.layers.HvdAllToAllEmbedding(
        mpi_size=mpi_size,
        embedding_size=dense_embedding_size,
        key_dtype=tf.int32,
        value_dtype=tf.float32,
        initializer=embedding_initializer,
        devices=self.gpu_device,
        name=name + '_DenseUnifiedEmbeddingLayer',
        bp_v2=True,
        init_capacity=4096000,
        kv_creator=self.kv_creator)

    self.sparse_embedding_layer = de.keras.layers.HvdAllToAllEmbedding(
        mpi_size=mpi_size,
        embedding_size=sparse_embedding_size,
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        initializer=embedding_initializer,
        devices=self.cpu_device,
        name=name + '_SparseUnifiedEmbeddingLayer',
        init_capacity=4096000,
        kv_creator=self.kv_creator)

    self.dnn = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))

  def build(self, input_shape):
    super(ChannelEmbeddingLayers, self).build(input_shape)

  def __call__(self, features_info):
    dense_inputs = []
    dense_input_dims = []
    sparse_inputs = []
    sparse_input_dims = []
    for fea_name, fea_info in features_info.items():
      # The features of GPU table and CPU table to be combined and queried are processed separately.
      if fea_info['ptype'] == 'normal_gpu':
        dense_inputs.append(fea_info['pretreated_tensor'])
        dense_input_dims.append(fea_info['dim'])
      elif fea_info['ptype'] == 'sparse_cpu':
        sparse_inputs.append(fea_info['pretreated_tensor'])
        sparse_input_dims.append(fea_info['dim'])
      else:
        ptype = fea_info['ptype']
        raise NotImplementedError(f'Not support ptype {ptype}.')
    # The GPU table combined query starts
    dense_input_tensors_concat, dense_input_split_dims, dense_input_is_sequence_feature = \
        embedding_inputs_concat(dense_inputs, dense_input_dims)
    dense_emb_concat = self.dense_embedding_layer(dense_input_tensors_concat)
    # The CPU table combined query starts
    sparse_input_tensors_concat, sparse_input_split_dims, sparse_input_is_sequence_feature = \
        embedding_inputs_concat(sparse_inputs, sparse_input_dims)
    sparse_emb_concat = self.sparse_embedding_layer(sparse_input_tensors_concat)
    # Slice the combined query result
    dense_emb_outs = embedding_out_split(dense_emb_concat,
                                         dense_input_split_dims)
    sparse_emb_outs = embedding_out_split(sparse_emb_concat,
                                          sparse_input_split_dims)
    # Process the results of the combined query after slicing.
    embedding_outs = []
    input_is_sequence_feature = dense_input_is_sequence_feature + sparse_input_is_sequence_feature
    for i, embedding in enumerate(dense_emb_outs + sparse_emb_outs):
      if input_is_sequence_feature[i] == True:
        # Deal with the embedding from vector features.
        embedding_vec = tf.math.reduce_mean(
            embedding, axis=1,
            keepdims=True)  # (feature_combin_num, (batch, x, emb_size))
      else:
        embedding_vec = embedding
      embedding_vec = tf.keras.layers.Flatten()(embedding_vec)
      embedding_outs.append(embedding_vec)
    # Final embedding result.
    embeddings_concat = tf.keras.layers.Concatenate(axis=1)(embedding_outs)

    return self.dnn(embeddings_concat)


class DualChannelsDeepModel(tf.keras.Model):

  def __init__(self,
               user_embedding_size=1,
               movie_embedding_size=1,
               embedding_initializer=None,
               is_training=True,
               mpi_size=1,
               mpi_rank=0):

    if is_training:
      de.enable_train_mode()
      if embedding_initializer is None:
        embedding_initializer = tf.keras.initializers.VarianceScaling()
    else:
      de.enable_inference_mode()
      if embedding_initializer is None:
        embedding_initializer = tf.keras.initializers.Zeros()

    super(DualChannelsDeepModel, self).__init__()
    self.user_embedding_size = user_embedding_size
    self.movie_embedding_size = movie_embedding_size

    self.user_embedding = ChannelEmbeddingLayers(
        name='user',
        dense_embedding_size=user_embedding_size,
        sparse_embedding_size=user_embedding_size * 2,
        embedding_initializer=embedding_initializer,
        mpi_size=mpi_size,
        mpi_rank=mpi_rank)
    self.movie_embedding = ChannelEmbeddingLayers(
        name='movie',
        dense_embedding_size=movie_embedding_size,
        sparse_embedding_size=movie_embedding_size * 2,
        embedding_initializer=embedding_initializer,
        mpi_size=mpi_size,
        mpi_rank=mpi_rank)

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
    # Construct input layers
    for fea_name in features.keys():
      fea_info = feature_info_spec[fea_name]
      input_tensor = features[fea_name]
      input_tensor = tf.keras.layers.Lambda(lambda x: x,
                                            name=fea_name)(input_tensor)
      input_tensor = tf.reshape(input_tensor, (-1, fea_info['dim']))
      fea_info['input_tensor'] = input_tensor
      if fea_info.__contains__('boundaries'):
        input_tensor = Bucketize(
            boundaries=fea_info['boundaries'])(input_tensor)
      # To prepare for GPU table combined queries, use a prefix to distinguish different features in a table.
      if fea_info['ptype'] == 'normal_gpu':
        if fea_info['dtype'] == tf.int64:
          input_tensor_prefix_code = int(fea_info['code']) << 17
        elif fea_info['dtype'] == tf.int32:
          input_tensor_prefix_code = int(fea_info['code']) << 14
        else:
          input_tensor_prefix_code = None
        if input_tensor_prefix_code is not None:
          # input_tensor = tf.bitwise.bitwise_xor(input_tensor, input_tensor_prefix_code)
          # xor operation can be replaced with addition operation to facilitate subsequent optimization of TRT and OpenVino.
          input_tensor = tf.add(input_tensor, input_tensor_prefix_code)
      fea_info['pretreated_tensor'] = input_tensor

    user_fea = ['user_id', 'user_gender', 'user_occupation_label']
    user_fea = [i for i in features.keys() if i in user_fea]
    user_fea_info = {
        key: value
        for key, value in feature_info_spec.items()
        if key in user_fea
    }
    user_latent = self.user_embedding(user_fea_info)
    movie_fea = ['movie_id', 'movie_genres', 'user_occupation_label']
    movie_fea = [i for i in features.keys() if i in movie_fea]
    movie_fea_info = {
        key: value
        for key, value in feature_info_spec.items()
        if key in movie_fea
    }
    movie_latent = self.movie_embedding(movie_fea_info)
    latent = tf.concat([user_latent, movie_latent], axis=1)

    x = self.dnn1(latent)
    x = self.dnn2(x)
    x = self.dnn3(x)

    bias = self.bias_net(latent)
    x = 0.2 * x + 0.8 * bias
    user_rating = tf.keras.layers.Lambda(lambda x: x, name='user_rating')(x)
    return {'user_rating': user_rating}


def get_dataset(batch_size=1):
  ds = tfds.load("movielens/1m-ratings",
                 split="train",
                 data_dir="~/dataset",
                 download=True)
  features = ds.map(
      lambda x: {
          "movie_id":
              tf.strings.to_number(x["movie_id"], tf.int64),
          "movie_genres":
              tf.cast(x["movie_genres"][0], tf.int32),
          "user_id":
              tf.strings.to_number(x["user_id"], tf.int64),
          "user_gender":
              tf.cast(x["user_gender"], tf.int32),
          "user_occupation_label":
              tf.cast(x["user_occupation_label"], tf.int32),
          "bucketized_user_age":
              tf.cast(x["bucketized_user_age"], tf.int32),
          "timestamp":
              tf.cast(x["timestamp"] - 880000000, tf.int32),
      })

  ratings = ds.map(lambda x: {
      "user_rating":
          tf.one_hot(tf.cast(x["user_rating"] - 1, dtype=tf.int64), 5)
  })
  dataset = tf.data.Dataset.zip((features, ratings))
  shuffled = dataset.shuffle(1_000_000,
                             seed=2021,
                             reshuffle_each_iteration=False)
  dataset = shuffled.repeat(1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  # Only GPU:0 since TF is set to be visible to GPU:X
  dataset = dataset.apply(
      tf.data.experimental.prefetch_to_device('GPU:0', buffer_size=2))
  return dataset


def export_to_savedmodel(model, savedmodel_dir):
  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  if not os.path.exists(savedmodel_dir):
    os.mkdir(savedmodel_dir)

  ########################## What really happened ##########################
  # # Calling the TF save API for all ranks causes file conflicts, so KV files other than rank0 need to be saved by calling the underlying API separately.
  # if hvd.rank() == 0:
  #   tf.keras.models.save_model(model,
  #                              savedmodel_dir,
  #                              overwrite=True,
  #                              include_optimizer=True,
  #                              save_traces=True,
  #                              options=save_options)
  # else:
  #   de_dir = os.path.join(savedmodel_dir, "variables", "TFRADynamicEmbedding")
  #   for layer in model.layers:
  #     if hasattr(layer, "params"):
  #       # Save embedding parameters
  #       layer.params.save_to_file_system(dirpath=de_dir,
  #                                        proc_size=hvd.size(),
  #                                        proc_rank=hvd.rank())
  #       # Save the optimizer parameters
  #       opt_de_vars = layer.optimizer_vars.as_list() if hasattr(
  #           layer.optimizer_vars, "as_list") else layer.optimizer_vars
  #       for opt_de_var in opt_de_vars:
  #         opt_de_var.save_to_file_system(dirpath=de_dir,
  #                                        proc_size=hvd.size(),
  #                                        proc_rank=hvd.rank())

  # TFRA modify the Keras save function with a patch.
  # !!!! Run save_model function in all rank !!!!
  de.keras.models.save_model(model,
                             savedmodel_dir,
                             overwrite=True,
                             include_optimizer=True,
                             save_traces=True,
                             options=save_options)


def export_for_serving(model, export_dir):
  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  if not os.path.exists(export_dir):
    os.mkdir(export_dir)

  def save_spec():
    if hasattr(model, 'save_spec'):
      # tf version >= 2.6
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

  arg_specs, kwarg_specs = save_spec()

  ########################## What really happened ##########################
  # if hvd.rank() == 0:
  #   # Remember to remove optimizer parameters when ready to serve.
  #   tf.keras.models.save_model(
  #       model,
  #       export_dir,
  #       overwrite=True,
  #       include_optimizer=False,
  #       options=save_options,
  #       signatures={
  #           'serving_default':
  #               serve.get_concrete_function(*arg_specs, **kwarg_specs)
  #       },
  #   )
  # else:
  #   de_dir = os.path.join(export_dir, "variables", "TFRADynamicEmbedding")
  #   for layer in model.layers:
  #     if hasattr(layer, "params"):
  #       layer.params.save_to_file_system(dirpath=de_dir,
  #                                        proc_size=hvd.size(),
  #                                        proc_rank=hvd.rank())

  # TFRA modify the Keras save function with a patch.
  # !!!! Run save_model function in all rank !!!!
  de.keras.models.save_model(
      model,
      export_dir,
      overwrite=True,
      include_optimizer=False,
      options=save_options,
      signatures={
          'serving_default':
              serve.get_concrete_function(*arg_specs, **kwarg_specs)
      },
  )

  if hvd.rank() == 0:
    # Modify the inference graph to a stand-alone version
    from tensorflow.python.saved_model import save as tf_save
    tf.keras.backend.clear_session()
    de.enable_inference_mode()
    export_model = DualChannelsDeepModel(FLAGS.embedding_size,
                                         FLAGS.embedding_size,
                                         tf.keras.initializers.Zeros(),
                                         hvd.size(), hvd.rank())
    # The save_and_return_nodes function is used to overwrite the saved_model.pb file generated by the save_model function and rewrite the inference graph.
    tf_save.save_and_return_nodes(obj=export_model,
                                  export_dir=export_dir,
                                  options=save_options,
                                  experimental_skip_checkpoint=True)


def train():
  dataset = get_dataset(batch_size=32)
  model = DualChannelsDeepModel(FLAGS.embedding_size, FLAGS.embedding_size,
                                tf.keras.initializers.RandomNormal(0.0, 0.5),
                                hvd.size(), hvd.rank())
  optimizer = Adam(1E-3)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer)

  auc = tf.keras.metrics.AUC(num_thresholds=1000)
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[
                    auc,
                ])

  if os.path.exists(FLAGS.model_dir + '/variables'):
    model.load_weights(FLAGS.model_dir + '/variables/variables')

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.model_dir)
  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  # horovod callback is used to broadcast the value generated by initializer of rank0.
  hvd_opt_init_callback = de.keras.callbacks.DEHvdBroadcastGlobalVariablesCallback(
      root_rank=0)
  ckpt_callback = de.keras.callbacks.ModelCheckpoint(
      filepath=FLAGS.model_dir + '/weights_epoch{epoch:03d}_loss{loss:.4f}',
      options=save_options)
  callbacks_list = [hvd_opt_init_callback, ckpt_callback]
  # The log class callback only takes effect in rank0 for convenience
  if hvd.rank() == 0:
    callbacks_list.extend([tensorboard_callback])
  # If there are callbacks such as evaluation metrics that call model calculations, take effect on all ranks.
  # callbacks_list.extend([my_auc_callback])

  model.fit(dataset,
            callbacks=callbacks_list,
            epochs=FLAGS.epochs,
            steps_per_epoch=FLAGS.steps_per_epoch,
            verbose=1 if hvd.rank() == 0 else 0)

  export_to_savedmodel(model, FLAGS.model_dir)
  export_for_serving(model, FLAGS.export_dir)


def export():
  de.enable_inference_mode()
  if not os.path.exists(FLAGS.export_dir):
    shutil.copytree(FLAGS.model_dir, FLAGS.export_dir)
  export_model = DualChannelsDeepModel(FLAGS.embedding_size,
                                       FLAGS.embedding_size,
                                       tf.keras.initializers.RandomNormal(
                                           0.0, 0.5),
                                       mpi_size=1,
                                       mpi_rank=0)
  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  # Modify the inference graph to a stand-alone version
  from tensorflow.python.saved_model import save as tf_save
  # The save_and_return_nodes function is used to overwrite the saved_model.pb file generated by the save_model function and rewrite the inference graph.
  tf_save.save_and_return_nodes(obj=export_model,
                                export_dir=FLAGS.export_dir,
                                options=save_options,
                                experimental_skip_checkpoint=True)


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
