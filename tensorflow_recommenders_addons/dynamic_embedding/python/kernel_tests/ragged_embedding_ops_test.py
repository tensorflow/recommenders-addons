import tensorflow as tf
import unittest

from tensorflow_recommenders_addons.dynamic_embedding.python.ops.ragged_embedding_ops import _fill_empty_rows


class TestFillEmptyRows(unittest.TestCase):

  def test_fill_empty_rows(self):
    test_ragged_tensor = tf.ragged.constant([[1, 2, 3], [], [4], [], [5, 6]],
                                            dtype=tf.int32)
    default_id = 0

    filled_ragged_tensor, is_row_empty = _fill_empty_rows(
        test_ragged_tensor, default_id)

    expected_filled = tf.ragged.constant([[1, 2, 3], [0], [4], [0], [5, 6]],
                                         dtype=tf.int32)
    expected_empty = tf.constant([False, True, False, True, False])

    self.assertTrue(
        tf.reduce_all(filled_ragged_tensor.to_tensor() ==
                      expected_filled.to_tensor()).numpy(),
        "Filled tensors do not match")
    self.assertTrue(
        tf.reduce_all(is_row_empty == expected_empty).numpy(),
        "Empty row flags do not match")
