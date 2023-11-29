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

import os.path
import re

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.keras.layers import HvdAllToAllEmbedding
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_ops import TrainableWrapper, DEResourceVariable

from tensorflow.python.framework import constant_op
try:
  from tensorflow.python.checkpoint.checkpoint import Checkpoint
except:
  from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging


class DEHvdCheckpoint(Checkpoint):
  """Overwrite tf.train.Saver class
    Calling the TF save API for all ranks causes file conflicts, 
    so KV files other than rank0 need to be saved by calling the underlying API separately.
    This is a convenience function for saving HvdAllToAllEmbedding to KV files in different rank.
  """

  def __init__(self, root=None, **kwargs):
    """Creates a training checkpoint for a single or group of objects.

    Args:
      root: The root object to checkpoint. `root` may be a trackable object or
        `WeakRef` of a trackable object.
      **kwargs: Keyword arguments are set as attributes of this object, and are
        saved with the checkpoint. All `kwargs` must be trackable objects, or a
        nested structure of trackable objects (`list`, `dict`, or `tuple`).

    Raises:
      ValueError: If `root` or the objects in `kwargs` are not trackable. A
        `ValueError` is also raised if the `root` object tracks different
        objects from the ones listed in attributes in kwargs (e.g.
        `root.child = A` and `tf.train.Checkpoint(root, child=B)` are
        incompatible).

    """
    try:
      import horovod.tensorflow as hvd
      try:
        hvd.rank()
        self._hvd = hvd
      except:
        self._hvd = None
    except:
      self._hvd = None

    self._tmp_var_key_set = set({})
    for k, _ in sorted(kwargs.items(), key=lambda item: item[0]):
      self._tmp_var_key_set.add(k)
    super(DEHvdCheckpoint, self).__init__(root, **kwargs)

  def _get_de_variable_folder_dir(self,
                                  save_path: str,
                                  global_step: str = None):
    save_path_parent = os.path.dirname(save_path)
    if global_step is not None:
      de_variable_folder_dir = os.path.join(
          save_path_parent, "TFRADynamicEmbedding-{}".format(global_step))
    else:
      de_variable_folder_dir = os.path.join(save_path_parent,
                                            "TFRADynamicEmbedding")
    return de_variable_folder_dir

  def _delete_redundant_de_dir(self, ckpt_index_list: list):
    if not len(ckpt_index_list) > 0:
      return
    save_path_parent = os.path.dirname(ckpt_index_list[0])
    de_dir_pattern = os.path.join(save_path_parent, "TFRADynamicEmbedding-*")
    found_de_dir_set = set(file_io.get_matching_files(de_dir_pattern))
    keep_de_dir_set = set([])
    for file_path in ckpt_index_list:
      global_step = file_path.split('.index')[-2].split('-')[-1]
      de_dir = os.path.join(save_path_parent,
                            "TFRADynamicEmbedding-{}".format(global_step))
      keep_de_dir_set.add(de_dir)
    delete_de_dir_set = found_de_dir_set - keep_de_dir_set
    for de_dir in delete_de_dir_set:
      if file_io.is_directory(de_dir):
        file_io.delete_recursively(de_dir)

  def _de_var_fs_save_funtion(self, de_var, de_dir: str):
    a2a_emb = de_var._created_in_class
    hvd_size = 1 if self._hvd is None else self._hvd.size()
    hvd_rank = 0 if self._hvd is None else self._hvd.rank()
    if issubclass(a2a_emb.__class__, HvdAllToAllEmbedding):
      if de_var._saveable_object_creator is None:
        tf_logging.warning(
            "Please use FileSystemSaver when use HvdAllToAllEmbedding. "
            "It will allow TFRA load KV files when Embedding tensor parallel. "
            f"The embedding shards at each horovod rank are now temporarily stored in {de_dir}"
        )
      else:
        # save Dynamic Embedding Parameters
        de_var.save_to_file_system(dirpath=de_dir,
                                   proc_size=hvd_size,
                                   proc_rank=hvd_rank)
        # save optimizer parameters of Dynamic Embedding
        de_opt_vars = a2a_emb.optimizer_vars.as_list() if hasattr(
            a2a_emb.optimizer_vars, "as_list") else a2a_emb.optimizer_vars
        for de_opt_var in de_opt_vars:
          de_opt_var.save_to_file_system(dirpath=de_dir,
                                         proc_size=hvd_size,
                                         proc_rank=hvd_rank)

  def _de_var_fs_restore_funtion(self, de_var, de_dir: str):
    a2a_emb = de_var._created_in_class
    hvd_size = 1 if self._hvd is None else self._hvd.size()
    hvd_rank = 0 if self._hvd is None else self._hvd.rank()
    if issubclass(a2a_emb.__class__, HvdAllToAllEmbedding):
      if de_var._saveable_object_creator is None:
        tf_logging.warning(
            "Please use FileSystemSaver when use HvdAllToAllEmbedding. "
            "It will allow TFRA load KV files when Embedding tensor parallel. "
            f"The embedding shards at each horovod rank are now temporarily stored in {de_dir}"
        )
      else:
        # restore Dynamic Embedding Parameters
        de_var.load_from_file_system_with_restore_function(dirpath=de_dir,
                                                           proc_size=hvd_size,
                                                           proc_rank=hvd_rank)
        # restore optimizer parameters of Dynamic Embedding
        de_opt_vars = a2a_emb.optimizer_vars.as_list() if hasattr(
            a2a_emb.optimizer_vars, "as_list") else a2a_emb.optimizer_vars
        for de_opt_var in de_opt_vars:
          de_opt_var.load_from_file_system_with_restore_function(
              dirpath=de_dir, proc_size=hvd_size, proc_rank=hvd_rank)

  def _de_handle_root_and_var_with_func(self, de_dir: str, func):

    def _filter_de_hvd_a2a_tw(var):
      if not hasattr(var, "params") or not isinstance(var, TrainableWrapper):
        return False
      if not hasattr(var.params, "_created_in_class"):
        return False
      return True

    def _handle_model_or_variable(obj):
      if _filter_de_hvd_a2a_tw(obj):
        func(var.params, de_dir)
      if hasattr(obj, 'variables'):
        _iter = obj.variables() if callable(obj.variables) else obj.variables
        for var in _iter:
          if _filter_de_hvd_a2a_tw(var):
            func(var.params, de_dir)

    if hasattr(self, 'root'):
      _handle_model_or_variable(self.root)
    if len(self._tmp_var_key_set):
      for obj_key in self._tmp_var_key_set:
        obj_var = getattr(self, obj_key)
        _handle_model_or_variable(obj_var)

  def _de_hvd_write_fs_func(self, file_prefix, tf_write_func):

    def _get_de_dir_from_file_path(file_path):
      file_prefix_split = file_path.split('-')
      file_prefix_pattern = ''.join(file_prefix_split[0:-1])
      global_step = file_prefix_split[-1]
      if not global_step.isdigit():
        global_step = None
      de_dir = self._get_de_variable_folder_dir(file_path, global_step)
      return file_prefix_pattern, global_step, de_dir

    def _rank0_delete_files_and_return_de_dir(file_path):
      file_prefix_pattern, global_step, de_dir = _get_de_dir_from_file_path(
          file_path)
      if global_step is not None:
        ckpt_index_list = file_io.get_matching_files(file_prefix_pattern +
                                                     '-*.index')
        self._delete_redundant_de_dir(
            ckpt_index_list
        )  # Compatible with automatic sweep function of checkpointmanager
      return de_dir

    if self._hvd is None:
      file_path = tf_write_func()
      de_dir = _rank0_delete_files_and_return_de_dir(file_path)
      self._de_handle_root_and_var_with_func(de_dir=de_dir,
                                             func=self._de_var_fs_save_funtion)
    else:
      file_path = ''
      if self._hvd.rank() == 0:
        file_path = tf_write_func()
        self._hvd.broadcast_object(file_path,
                                   root_rank=0,
                                   name='de_hvd_broadcast_file_path')
        de_dir = _rank0_delete_files_and_return_de_dir(file_path)
        self._hvd.join()  # Sync for avoiding files conflict
        self._de_handle_root_and_var_with_func(
            de_dir=de_dir, func=self._de_var_fs_save_funtion)
        self._hvd.join(
        )  # Sync for avoiding files conflict and rank finish early
      else:
        file_path = self._hvd.broadcast_object(
            None, root_rank=0, name='de_hvd_broadcast_file_path')
        _, _, de_dir = _get_de_dir_from_file_path(file_path)
        self._hvd.join()  # Sync for avoiding files conflict
        self._de_handle_root_and_var_with_func(
            de_dir=de_dir, func=self._de_var_fs_save_funtion)
        self._hvd.join(
        )  # Sync for avoiding files conflict and rank finish early
    return file_path

  def _write(self, file_prefix, options=None, *args, **kwargs):
    """Internal method that implements Checkpoint.write().

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      options: Optional `tf.train.CheckpointOptions` object.
      write_done_callback: Optional callback function to be executed once
        the underlying checkpoint saving is finished. Example usage includes
        updating the checkpoint internal state.

    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """

    def tf_write_func_impl():
      return super(DEHvdCheckpoint, self)._write(file_prefix=file_prefix,
                                                 options=options,
                                                 *args,
                                                 **kwargs)

    return self._de_hvd_write_fs_func(file_prefix=file_prefix,
                                      tf_write_func=tf_write_func_impl)

  def write(self, file_prefix, options=None, *args, **kwargs):
    """
    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).
      options: Optional `tf.train.CheckpointOptions` object.

    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    """

    def tf_write_func_impl():
      if hasattr(super(DEHvdCheckpoint, self), '_write'):
        return super(DEHvdCheckpoint, self)._write(file_prefix=file_prefix,
                                                   options=options,
                                                   *args,
                                                   **kwargs)
      else:
        return super(DEHvdCheckpoint, self).write(file_prefix=file_prefix,
                                                  options=options,
                                                  *args,
                                                  **kwargs)

    return self._de_hvd_write_fs_func(file_prefix=file_prefix,
                                      tf_write_func=tf_write_func_impl)

  def restore(self, save_path, options=None, *args, **kwargs):
    """
    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency graph.
        If the checkpoint was written by the name-based
        `tf.compat.v1.train.Saver`, names are used to match variables.
      options: Optional `tf.train.CheckpointOptions` object.

    Returns:
      A load status object, which can be used to make assertions about the
      status of checkpoint restoration and run initialization/restore ops
      (of type `CheckpointLoadStatus`, or `InitializationOnlyStatus` if
      `save_path` is `None`).

      If `save_path` points to a name-based checkpoint, a `NameBasedSaverStatus`
      object is returned which runs restore ops from a name-based saver.

    Raises:
      RuntimeError: When a checkpoint file saved by async checkpoint is not
        available upon restore().
    """
    save_path_split = save_path.split('-')
    save_path_pattern = ''.join(save_path_split[0:-1])
    global_step = save_path_split[-1]
    if not global_step.isdigit():
      global_step = None
    de_dir = self._get_de_variable_folder_dir(save_path, global_step)

    impl_save_path = save_path
    if 'TFRADynamicEmbedding' in save_path:
      tf_logging.warning(
          f'''Arg save_path is {save_path}. Please do not name checkpoint with \'TFRADynamicEmbedding\', it is a special term. 
          If you are sure that this is not the name of checkpoint, 
          it is an unfixed bug related to tf.train.latest_checkpoint. 
          Please call restore function directly with the name of checkpoint.''')
      if global_step is not None:
        corresponding_ckpt_index = file_io.get_matching_files(
            os.path.join(os.path.dirname(save_path), f'*-{global_step}.index'))
      else:
        corresponding_ckpt_index = file_io.get_matching_files(
            os.path.join(os.path.dirname(save_path), '*.index'))
        de_dir = self._get_de_variable_folder_dir(
            save_path,
            (corresponding_ckpt_index[0].split('-')[-1].split('.index')[0]))
      if len(corresponding_ckpt_index) > 0:
        impl_save_path = corresponding_ckpt_index[0].split('.index')[0]
        if global_step is None:
          tf_logging.warning(
              f'Arg save_path {save_path} is illegal or not existing. Now using index {impl_save_path}'
          )

    result = super(DEHvdCheckpoint, self).restore(save_path=impl_save_path,
                                                  options=options,
                                                  *args,
                                                  **kwargs)
    if os.path.exists(de_dir):
      self._de_handle_root_and_var_with_func(
          de_dir=de_dir, func=self._de_var_fs_restore_funtion)
    else:
      tf_logging.warning(
          f'TFRADynamicEmbedding directory {de_dir} is not existing.')
    if self._hvd is not None:
      self._hvd.join()  # Sync for avoiding files conflict
    return result
