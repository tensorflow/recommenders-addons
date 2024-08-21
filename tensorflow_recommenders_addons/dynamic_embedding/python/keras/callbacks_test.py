import pytest
from tensorflow.python.framework import test_util

from tensorflow_recommenders_addons.dynamic_embedding.python.keras.callbacks import \
  DEHvdBroadcastGlobalVariablesCallback


def test_on_batch_end_subsequent_calls():
  if test_util.is_gpu_available():
    broadcast_callback = DEHvdBroadcastGlobalVariablesCallback(root_rank=0,
                                                               device='/gpu:0')
    broadcast_callback.broadcast_done = True
    broadcast_callback.on_batch_end(1)
