# Copyright 2022 The TensorFlow Recommenders-Addons Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# lint-as: python3

import os

from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging

from tensorflow_recommenders_addons.dynamic_embedding.python.keras.layers import HvdAllToAllEmbedding
from tensorflow_recommenders_addons import dynamic_embedding as de

try:
  import horovod.tensorflow as hvd
except Exception as e:
  logging.warning("An exception occurred when import horovod.tensorflow: " +
                  str(e))
  hvd = None


class DEHvdBroadcastGlobalVariablesCallbackImpl(object):

  def __init__(self, backend, root_rank, device='', *args):
    if hvd is None:
      raise ValueError(
          "Please install Horovod properly first if you want to use distributed synchronous training based on Horovod"
      )
    super(DEHvdBroadcastGlobalVariablesCallbackImpl, self).__init__(*args)
    self.backend = backend
    self.root_rank = root_rank
    self.device = device
    self.broadcast_done = False
    self._local_vars = set()

  def register_local_var(self, var):
    """
    Registers a variable as worker local. Horovod will not perform broadcasting
        operation on this variable.
    """
    if int(versions.__version__.split('.')[0]) < 2:
      raise Exception(
          'Registering local variables for '
          'BroadcastGlobalVariablesCallback is not supported in TF 1.*')

    self._local_vars.add(var.ref())

  def on_batch_end(self, batch, logs=None):
    if self.broadcast_done:
      return

    with ops.device(self.device):
      if hvd._executing_eagerly() and hasattr(self.model, 'variables'):
        # TensorFlow 2.0 or TensorFlow eager
        filter_lambda = lambda x: (x.ref() not in self._local_vars) and (
            not isinstance(x, de.TrainableWrapper)) and (not isinstance(
                x, de.DEResourceVariable))
        broadcast_vars = [
            var for var in self.model.variables if filter_lambda(var)
        ]
        hvd.broadcast_variables(broadcast_vars, root_rank=self.root_rank)
        broadcast_optimizer_vars = [
            var for var in self.model.optimizer.variables()
            if filter_lambda(var)
        ]
        hvd.broadcast_variables(broadcast_optimizer_vars,
                                root_rank=self.root_rank)
      else:
        bcast_op = hvd.broadcast_global_variables(self.root_rank)
        self.backend.get_session().run(bcast_op)

    self.broadcast_done = True


class DEHvdBroadcastGlobalVariablesCallback(
    DEHvdBroadcastGlobalVariablesCallbackImpl, callbacks.Callback):

  def __init__(self, root_rank=0, device='', local_variables=None):
    super(DEHvdBroadcastGlobalVariablesCallback,
          self).__init__(K, root_rank, device)
    is_var_fn = lambda x: (isinstance(x, variables.Variable) or isinstance(
        x, variables.VariableV1))
    if local_variables is None:
      local_variables = []
    elif is_var_fn(local_variables):
      local_variables = [local_variables]
    elif not all(is_var_fn(var) for var in local_variables):
      raise ValueError("All local variables must be of tf.Variable type.")
    for var in local_variables:
      self.register_local_var(var)


class DEHvdModelCheckpoint(callbacks.ModelCheckpoint):

  def __init__(self, *args, **kwargs):
    self._signatures = kwargs.get('signatures', None)
    super(DEHvdModelCheckpoint, self).__init__(*args, **kwargs)

  def _save_de_model(self, filepath):
    if hvd.rank() == 0:
      if self.save_weights_only:
        self.model.save_weights(filepath, overwrite=True, options=self._options)
      else:
        self.model.save(filepath,
                        overwrite=True,
                        signatures=self._signatures,
                        options=self._options)
    else:
      de_dir = os.path.join(filepath, "variables", "TFRADynamicEmbedding")
      for var in self.model.variables:
        if not hasattr(var, "params") or not isinstance(var,
                                                        de.TrainableWrapper):
          continue
        if not hasattr(var.params, "_created_in_class"):
          continue
        de_var = var.params
        a2a_emb = de_var._created_in_class
        if issubclass(a2a_emb.__class__, HvdAllToAllEmbedding):
          # save Dynamic Embedding Parameters
          de_var.save_to_file_system(dirpath=de_dir,
                                     proc_size=hvd.size(),
                                     proc_rank=hvd.rank())
          # save optimizer parameters of Dynamic Embedding
          de_opt_vars = a2a_emb.optimizer_vars.as_list() if hasattr(
              a2a_emb.optimizer_vars, "as_list") else a2a_emb.optimizer_vars
          for de_opt_var in de_opt_vars:
            de_opt_var.save_to_file_system(dirpath=de_dir,
                                           proc_size=hvd.size(),
                                           proc_rank=hvd.rank())
    hvd.join()  # Sync for avoiding data conflict or missing rank

  def _save_model(self, epoch, logs):
    """Saves the model.

    Args:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      # Block only when saving interval is reached.
      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, logs)

      try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if current is None:
            logging.warning(
                'Can save best model only with %s available, '
                'skipping.', self.monitor)
          else:
            if self.monitor_op(current, self.best):
              if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' %
                      (epoch + 1, self.monitor, self.best, current, filepath))
              self.best = current
              self._save_de_model(filepath)
            else:
              if self.verbose > 0:
                print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, self.monitor, self.best))
        else:
          if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
          self._save_de_model(filepath)

        self._maybe_remove_file()
      except IOError as e:
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in str(e.args[0]).lower():
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))
        # Re-throw the error for any other causes.
        raise e
