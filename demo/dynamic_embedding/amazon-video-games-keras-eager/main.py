import feature
import video_game_model
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de
try:
  from tensorflow.keras.optimizers.legacy import Adam
except:
  from tensorflow.keras.optimizers import Adam

from absl import flags
from absl import app

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_steps', 500, 'Number of training steps.')
flags.DEFINE_integer('embedding_size', 4, 'Embedding size.')
flags.DEFINE_integer('shuffle_size', 3000,
                     'Shuffle pool size for input examples.')
flags.DEFINE_integer('max_size', 100000,
                     'Number of reserved features in embedding.')
flags.DEFINE_string('export_dir', './export_dir', 'Directory to export model.')
flags.DEFINE_string('mode', 'train', 'Select the running mode: train or test.')
flags.DEFINE_string('save_format', 'keras', 'options: keras, tf')

FLAGS = flags.FLAGS


def train(num_steps):
  """
  Do trainnig and produce model.
  """

  # Create a model
  model = video_game_model.VideoGameDnn(batch_size=FLAGS.batch_size,
                                        embedding_size=FLAGS.embedding_size)
  optimizer = Adam(1E-3, clipnorm=None)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer)
  auc = tf.keras.metrics.AUC(num_thresholds=1000)
  accuracy = tf.keras.metrics.BinaryAccuracy(dtype=tf.float32)
  model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[accuracy, auc])

  # Get data iterator
  iterator = feature.initialize_dataset(batch_size=FLAGS.batch_size,
                                        split='train',
                                        shuffle_size=FLAGS.shuffle_size,
                                        skips=0,
                                        balanced=True)

  # Run training.
  try:
    for step in range(num_steps):
      features, labels = feature.input_fn(iterator)

      if step % 10 == 0:
        verbose = 1
      else:
        verbose = 0

      model.fit(features, labels, steps_per_epoch=1, epochs=1, verbose=verbose)

      if verbose > 0:
        print('step: {}, size of sparse domain: {}'.format(
            step, model.embedding_store.size()))
      model.embedding_store.restrict(int(FLAGS.max_size * 0.8),
                                     trigger=FLAGS.max_size)

  except tf.errors.OutOfRangeError:
    print('Run out the training data.')

  # Save the model for inference.
  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  if FLAGS.save_format == 'tf':
    model.save(FLAGS.export_dir, options=options)
  elif FLAGS.save_format == 'keras':
    tf.keras.models.save_model(model, FLAGS.export_dir, options=options)
  else:
    raise NotImplemented


def test(num_steps):
  """
  Use some sampels to test the accuracy of model prediction.
  """

  # Load model.
  options = tf.saved_model.LoadOptions()
  if FLAGS.save_format == 'tf':
    model = tf.saved_model.load(FLAGS.export_dir, tags='serve')

    def model_fn(x):
      return model.signatures['serving_default'](x)['output_1']

  elif FLAGS.save_format == 'keras':
    model = tf.keras.models.load_model(FLAGS.export_dir)
    model_fn = model.__call__

  else:
    raise NotImplemented

  # Get data iterator
  iterator = feature.initialize_dataset(batch_size=FLAGS.batch_size,
                                        split='train',
                                        shuffle_size=0,
                                        skips=100000)

  # Test click-ratio
  ctr = tf.metrics.Accuracy()
  for step in range(num_steps):
    features, labels = feature.input_fn(iterator)
    probabilities = model_fn(features)
    probabilities = tf.reshape(probabilities, (-1))
    preds = tf.cast(tf.round(probabilities), dtype=tf.int32)
    labels = tf.cast(labels, dtype=tf.int32)
    ctr.update_state(labels, preds)
    print("step: {}, ctr: {}".format(step, ctr.result()))


def main(argv):
  del argv
  if FLAGS.mode == 'train':
    train(FLAGS.num_steps)
  elif FLAGS.mode == 'test':
    test(FLAGS.num_steps)
  else:
    raise ValueError('running mode only supports `train` or `test`')


if __name__ == '__main__':
  app.run(main)
