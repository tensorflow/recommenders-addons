import feature
import video_game_model
import tensorflow as tf
from tensorflow_recommenders_addons import dynamic_embedding as de

from absl import flags
from absl import app

flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_steps', 500, 'Number of training steps.')
flags.DEFINE_integer('embedding_size', 4, 'Embedding size.')
flags.DEFINE_integer('shuffle_size', 3000,
                     'Shuffle pool size for input examples.')
flags.DEFINE_integer('reserved_features', 30000,
                     'Number of reserved features in embedding.')
flags.DEFINE_string('export_dir', './export_dir', 'Directory to export model.')
flags.DEFINE_string('mode', 'train', 'Select the running mode: train or test.')

FLAGS = flags.FLAGS


def train(num_steps):
  """
  Do trainnig and produce model.
  """

  # Create a model
  model = video_game_model.VideoGameDnn(batch_size=FLAGS.batch_size,
                                        embedding_size=FLAGS.embedding_size)

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
      loss, auc = model.train(features, labels)

      # To avoid too many features burst the memory, we restrict
      # the model embedding layer to `reserved_features` features.
      # And the restriction behavior will be triggered when it gets
      # over `reserved_features * 1.2`.
      model.embedding_store.restrict(FLAGS.reserved_features,
                                     trigger=int(FLAGS.reserved_features * 1.2))

      if step % 10 == 0:
        print('step: {}, loss: {}, var_size: {}, auc: {}'.format(
            step, loss, model.embedding_store.size(), auc))

  except tf.errors.OutOfRangeError:
    print('Run out the training data.')

  # Set TFRA ops become legit.
  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  # Save the model for inference.
  inference_model = video_game_model.VideoGameDnnInference(model)
  inference_model(feature.input_fn(iterator)[0])
  inference_model.save('export', signatures=None, options=options)


def test(num_steps):
  """
  Use some sampels to test the accuracy of model prediction.
  """

  # Load model.
  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  model = tf.saved_model.load('export', tags='serve', options=options)
  sig = model.signatures['serving_default']

  # Get data iterator
  iterator = feature.initialize_dataset(batch_size=FLAGS.batch_size,
                                        split='train',
                                        shuffle_size=0,
                                        skips=100000)

  # Do tests.
  for step in range(num_steps):
    features, labels = feature.input_fn(iterator)
    probabilities = sig(features)['output_1']
    probabilities = tf.reshape(probabilities, (-1))
    preds = tf.cast(tf.round(probabilities), dtype=tf.int32)
    labels = tf.cast(labels, dtype=tf.int32)
    ctr = tf.metrics.Accuracy()(labels, preds)
    print("step: {}, ctr: {}".format(step, ctr))


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
