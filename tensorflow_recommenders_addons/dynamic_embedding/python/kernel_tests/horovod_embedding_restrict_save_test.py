"""
unit tests of save model that uses HvdAllToAllEmbedding
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from time import sleep

import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

try:
  from tf_keras import layers, Sequential, models, backend
  from tf_keras.initializers import Zeros
  from tf_keras.optimizers import Adam
except:
  from tensorflow.keras import layers, Sequential, models, backend
  from tensorflow.keras.initializers import Zeros
  try:
    from tensorflow.keras.optimizers import Adam
  except:
    from tensorflow.keras.legacy.optimizers import Adam


def get_all_to_all_emb_model(emb_t, opt, *args, **kwargs):
  l0 = layers.InputLayer(input_shape=(None,), dtype=dtypes.int64)
  l1 = emb_t(*args, **kwargs)
  l2 = layers.Dense(8, 'relu', kernel_initializer='zeros')
  l3 = layers.Dense(1, 'sigmoid', kernel_initializer='zeros')
  if emb_t == de.keras.layers.HvdAllToAllEmbedding:
    model = Sequential([l0, l1, l2, l3])
  else:
    raise TypeError('Unsupported embedding layer {}'.format(emb_t))

  model.compile(optimizer=opt, loss='mean_absolute_error')
  return model


class HorovodAllToAllRestrictPolicyTest(test.TestCase):

  def test_all_to_all_embedding_restrict_policy_save(self):
    try:
      import horovod.tensorflow as hvd
    except (NotFoundError):
      self.skipTest(
          "Skip the test for horovod import error with Tensorflow-2.7.0 on MacOS-12."
      )

    hvd.init()

    name = "all2all_emb"
    keras_base_opt = Adam(1.0)
    base_opt = de.DynamicEmbeddingOptimizer(keras_base_opt, synchronous=True)

    init = Zeros()
    kv_creator = de.CuckooHashTableCreator(
        saver=de.FileSystemSaver(proc_size=hvd.size(), proc_rank=hvd.rank()))
    batch_size = 8
    start = 0
    dim = 10
    run_step = 10

    save_dir = "/tmp/hvd_distributed_restrict_policy_save" + str(
        hvd.size()) + str(dim)  # All ranks should share same save directory

    base_model = get_all_to_all_emb_model(
        de.keras.layers.HvdAllToAllEmbedding,
        base_opt,
        embedding_size=dim,
        initializer=init,
        bp_v2=False,
        kv_creator=kv_creator,
        restrict_policy=de.
        TimestampRestrictPolicy,  # Embedding table with restrict policy
        name='all2all_emb')

    for i in range(1, run_step):
      x = math_ops.range(start, start + batch_size, dtype=dtypes.int64)
      x = tf.reshape(x, (batch_size, -1))
      start += batch_size
      y = tf.zeros((batch_size, 1), dtype=dtypes.float32)
      base_model.fit(x, y, verbose=0)

    save_options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
    if hvd.rank() == 0:
      if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    hvd.join()  # Sync for avoiding files conflict
    base_model.save(save_dir, options=save_options)
    de.keras.models.save_model(base_model, save_dir, options=save_options)

    sleep(4)  # Wait for filesystem operation
    hvd_size = hvd.size()
    if hvd_size <= 1:
      hvd_size = 1
    base_dir = os.path.join(save_dir, "variables", "TFRADynamicEmbedding")
    for tag in ['keys', 'values']:
      for rank in range(hvd_size):
        self.assertTrue(
            os.path.exists(
                base_dir +
                f'/{name}-parameter_mht_1of1_rank{rank}_size{hvd_size}-{tag}'))
        self.assertTrue(
            os.path.exists(
                base_dir +
                f'/{name}-parameter_DynamicEmbedding_{name}-shadow_m_mht_1of1_rank{rank}_size{hvd_size}-{tag}'
            ))
        self.assertTrue(
            os.path.exists(
                base_dir +
                f'/{name}-parameter_DynamicEmbedding_{name}-shadow_v_mht_1of1_rank{rank}_size{hvd_size}-{tag}'
            ))
        # Restrict policy var saved for all ranks
        self.assertTrue(
            os.path.exists(
                base_dir +
                f'/{name}-parameter_timestamp_mht_1of1_rank{rank}_size{hvd_size}-{tag}'
            ))


if __name__ == "__main__":
  test.main()
