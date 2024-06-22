import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape


class TensorShapeTest(tf.test.TestCase):

  def test_empty_tensor_shape(self):
    """Test empty tensor shape."""
    assert TensorShape(None).as_list() == []
