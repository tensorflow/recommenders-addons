import numpy as np

import tensorflow as tf
from tensorflow_recommenders_addons import dynamic_embedding as de


class DynamicLayerNormalizationTest(tf.test.TestCase):

  def test_dynamic_shape_support(self):
    input_data = tf.keras.Input(shape=(None, 10), dtype=tf.float32)
    layer = de.keras.layers.LayerNormalization()
    output = layer(input_data)

    model = tf.keras.models.Model(inputs=input_data, outputs=output)

    np.random.seed(0)
    test_data = np.random.randn(2, 5, 10).astype(np.float32)
    output_data = model.predict(test_data)
    self.assertAllEqual(output_data.shape, (2, 5, 10))

    expected_mean = np.mean(test_data, axis=-1, keepdims=True)
    expected_std = np.std(test_data, axis=-1, keepdims=True)
    expected_normalized = (test_data - expected_mean) / (expected_std +
                                                         layer.epsilon)

    # Calculate expected output considering gamma and beta are default (i.e., gamma=1, beta=0)
    # 1e-3 is the default value for epsilon in LayerNormalization
    self.assertAllClose(output_data, expected_normalized, rtol=1e-3, atol=1e-3)

  def test_training_with_layer_normalization(self):
    input_dim = 10
    num_samples = 100
    output_dim = 1

    np.random.seed(0)
    features = np.random.randn(num_samples, input_dim).astype(np.float32)
    labels = (np.sum(features, axis=1) +
              np.random.randn(num_samples) * 0.5).astype(np.float32).reshape(
                  -1, 1)

    input_data = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    normalized = de.keras.layers.LayerNormalization()(input_data)
    output = tf.keras.layers.Dense(output_dim)(normalized)
    model = tf.keras.models.Model(inputs=input_data, outputs=output)

    model.compile(optimizer='adam', loss='mean_squared_error')
    initial_weights = [layer.get_weights() for layer in model.layers]

    model.fit(features, labels, epochs=5, batch_size=10, verbose=0)

    updated_weights = [layer.get_weights() for layer in model.layers]

    for initial, updated in zip(initial_weights, updated_weights):
      for ini_w, upd_w in zip(initial, updated):
        self.assertGreater(np.sum(np.abs(ini_w - upd_w)), 0)

    predictions = model.predict(features)
    self.assertAllEqual(predictions.shape, (num_samples, output_dim))
    self.assertGreater(np.std(predictions), 0.1)
