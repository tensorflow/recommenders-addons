# Copyright 2023 The TensorFlow Recommenders-Addons Authors.
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

import functools
import os.path

from tensorflow_recommenders_addons import dynamic_embedding as de

try:
  from keras.saving.saved_model import save as keras_saved_model_save
except:
  keras_saved_model_save = None
from tensorflow.python.keras.saving.saved_model import save as tf_saved_model_save
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model.save_options import SaveOptions

tf_original_save_func = tf_saved_model_save.save
if keras_saved_model_save is not None:
  keras_original_save_func = keras_saved_model_save.save


def _de_keras_save_func(original_save_func,
                        model,
                        filepath,
                        overwrite,
                        include_optimizer,
                        signatures=None,
                        options=None,
                        save_traces=True,
                        *args,
                        **kwargs):
  """Overwrite TF Keras save function
    Calling the TF save API for all ranks causes file conflicts, 
    so KV files other than rank0 need to be saved by calling the underlying API separately.
    This is a convenience function for saving HvdAllToAllEmbedding to KV files in different rank.
  """
  try:
    import horovod.tensorflow as hvd
    try:
      hvd.rank()
    except:
      hvd = None
  except:
    hvd = None

  if hvd is not None:
    filepath = hvd.broadcast_object(filepath,
                                    root_rank=0,
                                    name='de_hvd_broadcast_filepath')

  call_original_save_func = functools.partial(
      original_save_func,
      model=model,
      filepath=filepath,
      overwrite=overwrite,
      include_optimizer=include_optimizer,
      signatures=signatures,
      options=options,
      save_traces=save_traces,
      *args,
      **kwargs)

  de_dir = os.path.join(filepath, "variables", "TFRADynamicEmbedding")

  def _check_saveable_and_redirect_new_de_dir(hvd_rank=0):
    for var in model.variables:
      if not hasattr(var, "params"):
        continue
      if not hasattr(var.params, "_created_in_class"):
        continue
      de_var = var.params
      a2a_emb = de_var._created_in_class
      if issubclass(a2a_emb.__class__, de.keras.layers.HvdAllToAllEmbedding):
        if de_var._saveable_object_creator is None:
          if hvd_rank == 0:
            tf_logging.warning(
                "Please use FileSystemSaver when use HvdAllToAllEmbedding. "
                "It will allow TFRA load KV files when Embedding tensor parallel. "
                f"The embedding shards at each horovod rank are now temporarily stored in {de_dir}"
            )
      if not isinstance(de_var.kv_creator.saver, de.FileSystemSaver):
        # This function only serves FileSystemSaver.
        continue
      # Redirect new de_dir
      if hasattr(de_var, 'saveable'):
        de_var.saveable._saver_config.save_path = de_dir

  def _traverse_emb_layers_and_save(proc_size=1, proc_rank=0):
    for var in model.variables:
      if not hasattr(var, "params"):
        continue
      if not hasattr(var.params, "_created_in_class"):
        continue
      de_var = var.params
      a2a_emb = de_var._created_in_class
      if de_var._saveable_object_creator is not None:
        if not isinstance(de_var.kv_creator.saver, de.FileSystemSaver):
          # This function only serves FileSystemSaver.
          continue
        # save optimizer parameters of Dynamic Embedding
        if include_optimizer is True:
          de_opt_vars = a2a_emb.optimizer_vars.as_list() if hasattr(
              a2a_emb.optimizer_vars, "as_list") else a2a_emb.optimizer_vars
          for de_opt_var in de_opt_vars:
            de_opt_var.save_to_file_system(dirpath=de_dir,
                                           proc_size=proc_size,
                                           proc_rank=proc_rank)
        if proc_rank == 0:
          # FileSystemSaver works well at rank 0.
          continue
        # save Dynamic Embedding Parameters
        de_var.save_to_file_system(dirpath=de_dir,
                                   proc_size=proc_size,
                                   proc_rank=proc_rank)

  if hvd is None:
    call_original_save_func()
    _traverse_emb_layers_and_save()
  else:
    _check_saveable_and_redirect_new_de_dir(hvd.rank())
    if hvd.rank() == 0:
      call_original_save_func()
    _traverse_emb_layers_and_save(hvd.size(), hvd.rank())
    hvd.join()  # Sync for avoiding rank conflict


def de_hvd_save_model(model,
                      filepath,
                      overwrite=True,
                      include_optimizer=True,
                      signatures=None,
                      options=None,
                      save_traces=True,
                      *args,
                      **kwargs):
  return de_save_model(model=model,
                       filepath=filepath,
                       overwrite=True,
                       include_optimizer=True,
                       signatures=None,
                       options=None,
                       save_traces=True,
                       *args,
                       **kwargs)


def de_save_model(model,
                  filepath,
                  overwrite=True,
                  include_optimizer=True,
                  signatures=None,
                  options=None,
                  save_traces=True,
                  *args,
                  **kwargs):
  if keras_saved_model_save is not None:
    _save_handle = functools.partial(_de_keras_save_func,
                                     keras_original_save_func)
  else:
    _save_handle = functools.partial(_de_keras_save_func, tf_original_save_func)
  if options is None:
    options = SaveOptions(namespace_whitelist=['TFRA'])
  elif isinstance(options, SaveOptions) and hasattr(options,
                                                    'namespace_whitelist'):
    options.namespace_whitelist.append('TFRA')

  return _save_handle(model,
                      filepath,
                      overwrite,
                      include_optimizer,
                      signatures=signatures,
                      options=options,
                      save_traces=save_traces,
                      *args,
                      **kwargs)
