import pytest
from tensorflow_recommenders_addons.dynamic_embedding.python.keras.callbacks import \
  DEHvdBroadcastGlobalVariablesCallback, DEHvdModelCheckpoint


@pytest.fixture
def broadcast_callback(root_rank=0, device='/gpu:0'):
  # Instantiate with the corrected parameters
  return DEHvdBroadcastGlobalVariablesCallback(root_rank=root_rank,
                                               device=device)


def test_on_batch_end_subsequent_calls(broadcast_callback):
  broadcast_callback.broadcast_done = True
  broadcast_callback.on_batch_end(1)
