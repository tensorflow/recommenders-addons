import tensorflow as tf
import tensorflow_datasets as tfds
import sys

ENCODDING_SEGMENT_LENGTH = 1000000
NON_LETTER_OR_NUMBER_PATTERN = r'[^a-zA-Z0-9]'

FAETURES = [
    'customer_id', 'helpful_votes', 'marketplace', 'product_category',
    'product_id', 'product_parent', 'product_title', 'review_body',
    'review_date', 'review_headline', 'review_id', 'star_rating', 'total_votes'
]
LABEL = 'verified_purchase'


class _RawFeature(object):
  """
  Base class to mark a feature and encode.
  """

  def __init__(self, dtype, category):
    if not isinstance(category, int):
      raise TypeError('category must be an integer.')
    self.category = category

  def encode(self, tensor):
    raise NotImplementedError

  def match_category(self, tensor):
    min_code = self.category * ENCODDING_SEGMENT_LENGTH
    max_code = (self.category + 1) * ENCODDING_SEGMENT_LENGTH
    mask = tf.math.logical_and(tf.greater_equal(tensor, min_code),
                               tf.less(tensor, max_code))
    return mask


class _StringFeature(_RawFeature):

  def __init__(self, dtype, category):
    super(_StringFeature, self).__init__(dtype, category)

  def encode(self, tensor):
    tensor = tf.strings.to_hash_bucket_fast(tensor, ENCODDING_SEGMENT_LENGTH)
    tensor += ENCODDING_SEGMENT_LENGTH * self.category
    return tensor


class _TextFeature(_RawFeature):

  def __init__(self, dtype, category):
    super(_TextFeature, self).__init__(dtype, category)

  def encode(self, tensor):
    tensor = tf.strings.regex_replace(tensor, NON_LETTER_OR_NUMBER_PATTERN, ' ')
    tensor = tf.strings.split(tensor, sep=' ').to_tensor('')
    tensor = tf.strings.to_hash_bucket_fast(tensor, ENCODDING_SEGMENT_LENGTH)
    tensor += ENCODDING_SEGMENT_LENGTH * self.category
    return tensor


class _IntegerFeature(_RawFeature):

  def __init__(self, dtype, category):
    super(_IntegerFeature, self).__init__(dtype, category)

  def encode(self, tensor):
    tensor = tf.as_string(tensor)
    tensor = tf.strings.to_hash_bucket_fast(tensor, ENCODDING_SEGMENT_LENGTH)
    tensor += ENCODDING_SEGMENT_LENGTH * self.category
    return tensor


FEATURE_AND_ENCODER = {
    'customer_id': _StringFeature(tf.string, 1),
    'helpful_votes': _IntegerFeature(tf.int32, 2),
    'product_category': _StringFeature(tf.string, 3),
    'product_id': _StringFeature(tf.string, 4),
    'product_parent': _StringFeature(tf.string, 5),
    'product_title': _TextFeature(tf.string, 6),
    #'review_body': _TextFeature(tf.string, 7),  # bad feature
    'review_headline': _TextFeature(tf.string, 8),
    'review_id': _StringFeature(tf.string, 9),
    'star_rating': _IntegerFeature(tf.int32, 10),
    'total_votes': _IntegerFeature(tf.int32, 11),
}


def encode_feature(data):
  """
  Encode a single example to tensor.
  """
  collected_features = []
  for ft, encoder in FEATURE_AND_ENCODER.items():
    feature = encoder.encode(data[ft])
    batch_size = tf.shape(feature)[0]
    feature = tf.reshape(feature, (batch_size, -1))
    collected_features.append(feature)
  collected_features = tf.concat(collected_features, 1)
  return collected_features


def get_labels(data):
  return data['verified_purchase']


def initialize_dataset(batch_size=1,
                       split='train',
                       skips=0,
                       shuffle_size=0,
                       balanced=False):
  """
  Create a dataset and return a data iterator.
  """
  video_games_data = tfds.load('amazon_us_reviews/Digital_Video_Games_v1_00',
                               split=split,
                               as_supervised=False)

  if balanced:
    choice = tf.data.Dataset.range(2).repeat(None).shuffle(300)
    positive = video_games_data.filter(
        lambda x: tf.math.equal(get_labels(x['data']), 1))
    negative = video_games_data.filter(
        lambda x: tf.math.equal(get_labels(x['data']), 0))
    video_games_data = tf.data.experimental.choose_from_datasets(
        [positive, negative], choice)

  if shuffle_size > 0:
    video_games_data.shuffle(shuffle_size)
  if skips > 0:
    video_games_data.skip(skips)
  video_games_data = video_games_data.batch(batch_size)
  iterator = video_games_data.__iter__()
  return iterator


def input_fn(iterator):
  nested_input = iterator.get_next()
  data = nested_input['data']
  collected_features = encode_feature(data)
  labels = get_labels(data)
  return collected_features, labels
