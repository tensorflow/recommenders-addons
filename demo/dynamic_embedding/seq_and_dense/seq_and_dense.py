import math
import os
import shutil

import keras.layers
from absl import flags
from absl import app

from tensorflow_recommenders_addons import dynamic_embedding as de

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  #VERY IMPORTANT!
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
# Because of the two environment variables above no non-standard library imports should happen before this.
import tensorflow as tf
from tensorflow_recommenders_addons import dynamic_embedding as de
try:
  from tensorflow.keras.legacy.optimizers import Adam
except:
  from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import horovod.tensorflow as hvd
# optimal performance
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

tf.config.optimizer.set_jit(True)

def has_horovod() -> bool:
  return 'OMPI_COMM_WORLD_RANK' in os.environ or 'PMI_RANK' in os.environ


def config():
  # callback calls hvd.rank() so we need to initialize horovod here
  hvd.init()
  if has_horovod():
    print("Horovod is enabled.")
    if hvd.rank() > 0:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config_gpu(hvd.local_rank())
  else:
    config_gpu()


def config_gpu(rank=0):
  physical_devices = tf.config.list_physical_devices('GPU')
  if physical_devices:
    tf.config.set_visible_devices(physical_devices[rank], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[rank], True)
  else:
    print("No GPU found, using CPU instead.")


def get_cluster_size() -> int:
  return hvd.size() if has_horovod() else 1


def get_rank() -> int:
  return hvd.rank() if has_horovod() else 0


flags.DEFINE_string('mode', 'train', 'Select the running mode: train or test.')
flags.DEFINE_string('model_dir', 'model_dir',
                    'Directory where checkpoint stores.')
flags.DEFINE_string('export_dir', 'export_dir',
                    'Directory where model stores for inference.')
flags.DEFINE_integer('steps_per_epoch', 20000, 'Number of training steps.')
flags.DEFINE_integer('epochs', 1, 'Number of training epochs.')
flags.DEFINE_integer('embedding_size', 32,
                     'Embedding size for users and movies')
flags.DEFINE_integer('test_steps', 128, 'test steps.')
flags.DEFINE_integer('test_batch', 1024, 'test batch size.')
flags.DEFINE_bool('shuffle', True, 'shuffle dataset.')
FLAGS = flags.FLAGS

feature_info_spec = {
    'movie_id': {
        'code': 101,
        'dtype': tf.int64,
        'dim': 1,
        'vocab': 0,
        'input_tensor': None,
        'pretreated_tensor': None
    },
    'movie_genres': {
        'code': 102,
        'dtype': tf.int64,
        'dim': 0,  # means variable length
        'vocab': 20,
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'user_id': {
        'code': 103,
        'dtype': tf.int64,
        'dim': 1,
        'vocab': 0,
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'user_gender': {
        'code': 104,
        'dtype': tf.int64,
        'dim': 1,
        'vocab': 2,
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'user_occupation_label': {
        'code': 105,
        'dtype': tf.int64,
        'dim': 1,
        'vocab': 22,
        'input_tensor': None,
        'pretreated_tensor': None,
    },
    'bucketized_user_age': {
        'code': 106,
        'dtype': tf.int64,
        'dim': 1,
        'vocab': 10,
        'input_tensor': None,
        'pretreated_tensor': None,
        'boundaries': [i for i in range(0, 100, 10)],
    },
    'timestamp': {
        'code': 107,
        'dtype': tf.int64,
        'dim': 1,
        'vocab': 0,
        'input_tensor': None,
        'pretreated_tensor': None,
    }
}
# use one vocab size for both user and movie tower for simplicity
# encoding: user tower: user_occupation_label then user_gender
# movie tower: user_occupation_label then movie_genres
vocab_size = feature_info_spec['user_occupation_label'][
    'vocab'] + feature_info_spec['user_gender']['vocab'] + feature_info_spec[
        'movie_genres']['vocab']
prefix_size = feature_info_spec['user_occupation_label']['vocab']


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
    elif dim > 1:  # fixed seq features
      if tmp_sum > 0:
        input_split_dims.append(tmp_sum)
        input_is_sequence_feature.append(False)
      input_split_dims.append(dim)
      input_is_sequence_feature.append(True)
      tmp_sum = 0
    else:
      raise ("dim must >= 0, which is {}".format(dim))
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

  def call(self, x, **kwargs):
    return tf.raw_ops.Bucketize(input=x, boundaries=self.boundaries)

  def get_config(self,):
    config = {'boundaries': self.boundaries}
    base_config = super(Bucketize, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def concat_tensors(tensors):
  flat_values = []
  row_lengths = []  # This will now be a list of lists

  for tensor in tensors:
    if isinstance(tensor, tf.RaggedTensor):
      flat_values.append(tensor.flat_values)
      row_lengths.append(tensor.row_lengths())  # Append as a sublist
    else:
      flat_values.append(tf.reshape(tensor, [-1]))
      tensor_row_lengths = tf.ones(tf.shape(tensor)[0], dtype=tf.int32)
      row_lengths.append(tensor_row_lengths)
  concatenated = tf.concat(flat_values, axis=0)
  return concatenated, row_lengths

def concat_embedding(tensors, embeddings, row_lengths):
  results = []
  indices = [0]
  for length in row_lengths:  # Compute start indices for each segment
    sum =  tf.reduce_sum(length).numpy()
    indices.append(indices[-1] + sum)
  emb_shape = embeddings.shape[-1]
  for i, tensor in enumerate(tensors):
    # Calculate the start and end indices for the current tensor
    start_index = indices[i]
    end_index = indices[i+1]
    begin = tf.constant([start_index, 0])
    end = tf.constant([end_index, emb_shape])
    emb = tf.strided_slice(
      embeddings, begin, end)
    if isinstance(tensor, tf.RaggedTensor):
      orignal = tf.RaggedTensor.from_row_lengths(emb, row_lengths[i])
      emb = tf.reduce_mean(orignal, axis=1)
    results.append(emb)
  return tf.concat(results, axis=0)

def concat_embedding_slow(tensors, embeddings, row_lengths):
  results = []
  start_indices = [0]
  for length in row_lengths[:-1]:  # Compute start indices for each segment
    start_indices.append(start_indices[-1] + length)

  for i, tensor in enumerate(tensors):
    if isinstance(tensor, tf.RaggedTensor):
      # Calculate the start and end indices for the current tensor
      start_index = start_indices[i]
      end_index = start_index + row_lengths[i]

      # Extract embeddings using tf.strided_slice
      sliced_embeddings = tf.strided_slice(
        embeddings, [start_index, 0], [end_index, embeddings.shape[-1]])

      # Pool the embeddings
      pooled = tf.reduce_mean(sliced_embeddings, axis=0)
      results.append(pooled)
    else:
      count = tf.shape(tensor)[0]
      pooled_embeddings = tf.reshape(embeddings[start_indices[i]:start_indices[i] + count],
                                     [-1, embeddings.shape[-1]])
      results.append(pooled_embeddings)

  return tf.concat(results, axis=0)


def get_kv_creator(mpi_size: int,
                   mpi_rank: int,
                   vocab_size: int = 1,
                   value_size: int = 4,
                   dim: int = 16):
  gpus = tf.config.list_physical_devices('GPU')
  # The saver parameter of kv_creator saves the K-V in the hash table into a separate KV file.
  saver = de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank)
  if gpus:
    max_capacity = 2 * vocab_size
    # HKV use 128 slots per bucket, the key may lost if same bucket has more than 128 keys
    # so set the factor larger than max_capacity to avoid this case
    factor = mpi_size * 0.7
    config = de.HkvHashTableConfig(
        init_capacity=math.ceil(vocab_size / factor),
        max_capacity=math.ceil(max_capacity / factor),
        max_hbm_for_values=math.ceil(max_capacity * value_size * dim / factor))
    return de.HkvHashTableCreator(config=config, saver=saver)
  else:
    # for CuckooHashTable case the init_capacity passed in by Embedding layer
    # it handles one node multiple gpu but not multi-nodes case
    return de.CuckooHashTableCreator(saver=saver)


class ChannelEmbeddingLayers(tf.keras.layers.Layer):

  def __init__(self,
               name='',
               dense_embedding_size=1,
               sparse_embedding_size=1,
               dense_dim=1,
               embedding_initializer=tf.keras.initializers.Zeros(),
               mpi_size=1,
               mpi_rank=0):

    super(ChannelEmbeddingLayers, self).__init__()
    self.dense_embedding_layer = keras.layers.Embedding(
        input_dim=dense_dim,
        output_dim=dense_embedding_size,
        embeddings_initializer=embedding_initializer,
        name=name + '_DenseUnifiedEmbeddingLayer',
        dtype=tf.float32)

    init_capacity = 4096000
    kv_creator_sparse = get_kv_creator(mpi_size, mpi_rank, init_capacity,
                                       tf.dtypes.float32.size,
                                       sparse_embedding_size)
    self.sparse_embedding_layer = de.keras.layers.HvdAllToAllEmbedding(
        mpi_size=mpi_size,
        embedding_size=sparse_embedding_size,
        key_dtype=tf.int64,
        value_dtype=tf.float32,
        initializer=embedding_initializer,
        name=name + '_SparseUnifiedEmbeddingLayer',
        init_capacity=init_capacity,
        kv_creator=kv_creator_sparse,
        short_file_name=True,
    )

    self.dnn = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
        bias_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))

  def __call__(self, features_info):
    dense_inputs = []
    sparse_inputs = []
    sparse_input_dims = []
    for fea_name, fea_info in features_info.items():
      # The features of GPU table and CPU table to be combined and queried are processed separately.
      if fea_info['vocab'] > 0:
        dense_inputs.append(fea_info['pretreated_tensor'])
      else:
        sparse_inputs.append(fea_info['pretreated_tensor'])
    assert len(sparse_inputs) == 1, "Only one sparse input is supported."
    dense_input_tensors_concat, row_length = concat_tensors(dense_inputs)
    dense_emb_concat = self.dense_embedding_layer(dense_input_tensors_concat)
    dense_emb_outs = concat_embedding(dense_inputs, dense_emb_concat,
                                      row_length)
    sparse_emb_concat = self.sparse_embedding_layer(sparse_inputs[0])
    # Process the results of the combined query after slicing.
    embedding_outs = []

    embedding_vec = tf.keras.layers.Flatten()(sparse_emb_concat)
    embedding_outs.append(embedding_vec)
    # Final embedding result.
    dense_emb_outs = tf.keras.layers.Flatten()(dense_emb_outs)
    embedding_outs.append(dense_emb_outs)
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
    print(f"mpi_size {mpi_size}, mpi_rank {mpi_rank}")
    self.user_embedding = ChannelEmbeddingLayers(
        name='user',
        dense_embedding_size=user_embedding_size,
        sparse_embedding_size=user_embedding_size * 2,
        dense_dim=vocab_size,
        embedding_initializer=embedding_initializer,
        mpi_size=mpi_size,
        mpi_rank=mpi_rank)
    self.movie_embedding = ChannelEmbeddingLayers(
        name='movie',
        dense_embedding_size=movie_embedding_size,
        sparse_embedding_size=movie_embedding_size * 2,
        dense_dim=vocab_size,
        embedding_initializer=embedding_initializer,
        mpi_size=mpi_size,
        mpi_rank=mpi_rank)
    self.dynamic_layer_norm = de.keras.layers.LayerNormalization()
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
      if fea_info['dim'] > 0:
        input_tensor = tf.reshape(input_tensor, (-1, fea_info['dim']))
      fea_info['input_tensor'] = input_tensor
      if fea_info.__contains__('boundaries'):
        input_tensor = Bucketize(
            boundaries=fea_info['boundaries'])(input_tensor)
      if fea_name == 'user_gender' or fea_name == 'movie_genres':
        input_tensor = tf.add(input_tensor, prefix_size)
      fea_info['pretreated_tensor'] = input_tensor

    user_fea = ['user_id', 'user_gender', 'user_occupation_label']
    user_fea = [i for i in features.keys() if i in user_fea]
    user_fea_info = {
        key: value
        for key, value in feature_info_spec.items()
        if key in user_fea
    }
    movie_fea = ['movie_id', 'movie_genres', 'user_occupation_label']
    movie_fea = [i for i in features.keys() if i in movie_fea]
    movie_fea_info = {
        key: value
        for key, value in feature_info_spec.items()
        if key in movie_fea
    }
    user_latent = self.user_embedding(user_fea_info)
    movie_latent = self.movie_embedding(movie_fea_info)
    latent = tf.concat([user_latent, movie_latent], axis=1)

    normalized_emb = self.dynamic_layer_norm(latent)
    x = self.dnn1(normalized_emb)
    x = self.dnn2(x)
    x = self.dnn3(x)

    bias = self.bias_net(normalized_emb)
    x = 0.2 * x + 0.8 * bias
    user_rating = tf.keras.layers.Lambda(lambda x: x, name='user_rating')(x)
    return {'user_rating': user_rating}


def get_dataset(batch_size=1):
  ds = tfds.load("movielens/1m-ratings",
                 split="train",
                 data_dir="~/dataset",
                 download=True)

  def process_features(x):
    # read seq as ragged tensor
    movie_genres_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(
        x['movie_genres'], axis=-1),
                                                      lengths=None)
    return {
        "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
        "movie_genres": movie_genres_ragged,
        "user_id": tf.strings.to_number(x["user_id"], tf.int64),
        "user_gender": tf.cast(x["user_gender"], tf.int64),
        "user_occupation_label": tf.cast(x["user_occupation_label"], tf.int64),
        "bucketized_user_age": tf.cast(x["bucketized_user_age"], tf.int64),
        "timestamp": tf.cast(x["timestamp"] - 880000000, tf.int64),
    }

  features = ds.map(process_features)

  ratings = ds.map(lambda x: {
      "user_rating":
          tf.one_hot(tf.cast(x["user_rating"] - 1, dtype=tf.int64), 5)
  })
  dataset = tf.data.Dataset.zip((features, ratings))
  if FLAGS.shuffle:
    dataset = dataset.shuffle(1_000_000,
                              seed=2021,
                              reshuffle_each_iteration=False)
  dataset = dataset.repeat(1).batch(batch_size).prefetch(tf.data.AUTOTUNE)
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


def save_spec(save_model):
  if hasattr(save_model, 'save_spec'):
    # tf version >= 2.6
    return save_model.save_spec()
  else:
    arg_specs = list()
    kwarg_specs = dict()
    for i in save_model.inputs:
      arg_specs.append(i.type_spec)
    return [arg_specs], kwarg_specs


@tf.function
def serve(save_model, *args, **kwargs):
  return save_model(*args, **kwargs)


def export_for_serving(model, export_dir):
  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  if not os.path.exists(export_dir):
    os.mkdir(export_dir)

  arg_specs, kwarg_specs = save_spec(model)

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
              serve.get_concrete_function(model, *arg_specs, **kwarg_specs)
      },
  )

  if get_rank() == 0:
    # Modify the inference graph to a stand-alone version
    tf.keras.backend.clear_session()
    from tensorflow.python.saved_model import save as tf_save
    de.enable_inference_mode()
    export_model = DualChannelsDeepModel(FLAGS.embedding_size,
                                         FLAGS.embedding_size,
                                         tf.keras.initializers.Zeros(), False,
                                         1, 0)
    # The save_and_return_nodes function is used to overwrite the saved_model.pb file generated by the save_model function and rewrite the inference graph.
    tf_save.save_and_return_nodes(obj=export_model,
                                  export_dir=export_dir,
                                  options=save_options,
                                  experimental_skip_checkpoint=True,
                                  signatures={
                                      'serving_default':
                                          serve.get_concrete_function(
                                              export_model, *arg_specs,
                                              **kwarg_specs)
                                  })


def train():
  dataset = get_dataset(batch_size=32)
  model = DualChannelsDeepModel(FLAGS.embedding_size, FLAGS.embedding_size,
                                tf.keras.initializers.RandomNormal(0.0, 0.5),
                                True, get_cluster_size(), get_rank())
  optimizer = Adam(1E-3)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer, synchronous=True)

  auc = tf.keras.metrics.AUC(num_thresholds=1000)
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[
                    auc,
                ])

  if os.path.exists(FLAGS.model_dir + '/variables'):
    model.load_weights(FLAGS.model_dir)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.model_dir)
  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  ckpt_callback = de.keras.callbacks.ModelCheckpoint(
      filepath=FLAGS.model_dir + '/weights_epoch{epoch:03d}_loss{loss:.4f}',
      options=save_options)
  if has_horovod():
    # horovod callback is used to broadcast the value generated by initializer of rank0.
    hvd_opt_init_callback = de.keras.callbacks.DEHvdBroadcastGlobalVariablesCallback(
        root_rank=0)
    callbacks_list = [hvd_opt_init_callback, ckpt_callback]
  else:
    callbacks_list = [ckpt_callback]

  # The log class callback only takes effect in rank0 for convenience
  if get_rank() == 0:
    callbacks_list.extend([tensorboard_callback])
  # If there are callbacks such as evaluation metrics that call model calculations, take effect on all ranks.
  # callbacks_list.extend([my_auc_callback])

  model.fit(dataset,
            callbacks=callbacks_list,
            epochs=FLAGS.epochs,
            steps_per_epoch=FLAGS.steps_per_epoch,
            verbose=1 if get_rank() == 0 else 0)

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
                                       False,
                                       mpi_size=1,
                                       mpi_rank=0)
  save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  dummy_features = {
      'movie_id': tf.constant([0], dtype=tf.int64),
      'movie_genres': tf.constant([0], dtype=tf.int64),
      'user_id': tf.constant([0], dtype=tf.int64),
      'user_gender': tf.constant([0], dtype=tf.int64),
      'user_occupation_label': tf.constant([0], dtype=tf.int64),
      'bucketized_user_age': tf.constant([0], dtype=tf.int64),
      'timestamp': tf.constant([0], dtype=tf.int64)
  }
  export_model(dummy_features)
  arg_specs, kwarg_specs = save_spec(export_model)
  # Modify the inference graph to a stand-alone version
  from tensorflow.python.saved_model import save as tf_save
  # The save_and_return_nodes function is used to overwrite the saved_model.pb file generated by the save_model function and rewrite the inference graph.
  tf_save.save_and_return_nodes(obj=export_model,
                                export_dir=FLAGS.export_dir,
                                options=save_options,
                                experimental_skip_checkpoint=True,
                                signatures={
                                    'serving_default':
                                        serve.get_concrete_function(
                                            export_model, *arg_specs,
                                            **kwarg_specs)
                                })


def test():
  de.enable_inference_mode()

  dataset = get_dataset(batch_size=FLAGS.test_batch)
  model = tf.keras.models.load_model(FLAGS.export_dir)

  def get_close_or_equal_cnt(model, features, ratings):
    preds = model(features)
    preds = tf.math.argmax(preds['user_rating'], axis=1)
    ratings = tf.math.argmax(ratings['user_rating'], axis=1)
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


def inference():
  de.enable_inference_mode()
  # model = keras.models.load_model(
  model = tf.keras.models.load_model(FLAGS.export_dir)
  print(f"model signature keys: {model.signatures.keys()} {model.signatures}")
  inference_func = model.signatures['serving_default']

  dataset = get_dataset(batch_size=FLAGS.test_batch)
  it = iter(dataset)

  def get_close_or_equal_cnt(preds, ratings):
    preds = tf.math.argmax(preds, axis=1)
    ratings = tf.math.argmax(ratings, axis=1)
    close_cnt = tf.reduce_sum(
        tf.cast(tf.math.abs(preds - ratings) <= 1, dtype=tf.int32))
    equal_cnt = tf.reduce_sum(
        tf.cast(tf.math.abs(preds - ratings) == 0, dtype=tf.int32))
    return close_cnt, equal_cnt

  for step in range(FLAGS.test_steps):
    features, ratings = next(it)
    ratings = ratings['user_rating']
    outputs = inference_func(**features)
    preds = outputs['user_rating']

    close_cnt, equal_cnt = get_close_or_equal_cnt(preds, ratings)

    print(
        f'In batch prediction, step: {step}, {close_cnt}/{FLAGS.test_batch} are closely'
        f' accurate, {equal_cnt}/{FLAGS.test_batch} are absolutely accurate.')


def main(argv):
  del argv
  config()
  if FLAGS.mode == 'train':
    train()
  elif FLAGS.mode == 'export':
    export()
  elif FLAGS.mode == 'test':
    test()
  elif FLAGS.mode == 'inference':
    inference()
  else:
    raise ValueError('running mode only supports `train` or `test`')


if __name__ == '__main__':
  app.run(main)
