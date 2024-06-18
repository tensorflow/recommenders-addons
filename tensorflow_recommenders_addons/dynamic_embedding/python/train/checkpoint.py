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
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.tf_save_restore_patch import de_fs_saveable_class_names, de_fs_sub_saveable_class_names
from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
try:  # tf version >= 2.10.0
  from tensorflow.python.checkpoint.checkpoint import Checkpoint as TFCheckpoint
  from tensorflow.python.checkpoint import restore as ckpt_base
except:
  from tensorflow.python.training.tracking.util import Checkpoint as TFCheckpoint
  from tensorflow.python.training.tracking import base as ckpt_base
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging
from tensorflow.python.framework import dtypes


class DECheckpoint(TFCheckpoint):
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

    self._de_need_opt = False
    self._tmp_var_key_set = set({})
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      v_strname = str(v).lower()
      if "optimizer" in v_strname:
        self._de_need_opt = True
      self._tmp_var_key_set.add(k)
    patch_tf_checkpoint()
    super(DECheckpoint, self).__init__(root, **kwargs)

  def _de_var_fs_save_funtion(self, de_var, de_dir: str):
    hvd_size = 1 if self._hvd is None else self._hvd.size()
    hvd_rank = 0 if self._hvd is None else self._hvd.rank()
    de_var.save_to_file_system(dirpath=de_dir,
                               proc_size=hvd_size,
                               proc_rank=hvd_rank)

  def _de_handle_root_and_var_with_func(self, de_dir: str, func):

    def _filter_de_tw(var):
      if not hasattr(var, "params") or not isinstance(var, de.TrainableWrapper):
        return False
      if not hasattr(var.params, "saveable"):
        return False
      if type(var.params.saveable).__name__ not in de_fs_saveable_class_names:
        return False
      return True

    def _handle_model_or_variable(obj):
      if _filter_de_tw(obj):
        func(var.params, de_dir)
      if hasattr(obj, 'variables'):
        _iter = obj.variables() if callable(obj.variables) else obj.variables
        for var in _iter:
          if _filter_de_tw(var):
            func(var.params, de_dir)

    if hasattr(self, 'root'):
      _handle_model_or_variable(self.root)
    if len(self._tmp_var_key_set):
      for obj_key in self._tmp_var_key_set:
        obj_var = getattr(self, obj_key)
        _handle_model_or_variable(obj_var)

  def _redirect_new_de_dir(self, de_dir):
    use_session = (not context.executing_eagerly()
                   and not ops.inside_function())
    if use_session:
      if self._object_graph_feed_tensor is None:
        with ops.device("/cpu:0"):
          self._object_graph_feed_tensor = constant_op.constant(
              "", dtype=dtypes.string)
      object_graph_tensor = self._object_graph_feed_tensor
    else:
      object_graph_tensor = None
    try:
      if hasattr(self._saver, "_gather_saveables"):
        #TODO: _gather_saveables return nothing when restore
        named_saveable_objects, _, _, _ = self._saver._gather_saveables(
            object_graph_tensor=object_graph_tensor)
      elif hasattr(self._saver._graph_view, "serialize_object_graph"):
        named_saveable_objects, _, _ = self._saver._graph_view.serialize_object_graph(
        )
    except:
      raise (
          "Can't find _gather_saveables or _graph_view.serialize_object_graph function at self._saver! "
          "Unsupport TrackableSaver version!")
    for saveable in named_saveable_objects:
      if type(saveable).__name__ in de_fs_sub_saveable_class_names:
        if hasattr(saveable, '_saver_config'):
          saveable._saver_config.save_path = de_dir

  def _de_hvd_write_fs_func(self, file_prefix, tf_write_func):
    _, _, de_dir = _get_de_dir_from_file_path(file_prefix)
    self._redirect_new_de_dir(de_dir)
    hvd_only_worker = False
    if self._hvd is None:
      hvd_only_worker = True
    else:
      if self._hvd.size() <= 1:
        hvd_only_worker = True
    if hvd_only_worker is True:
      file_path = tf_write_func()
      de_dir = _rank0_delete_files_and_return_de_dir(file_path)
    else:
      file_path = ''
      if self._hvd.rank() == 0:
        file_path = tf_write_func()
        self._hvd.broadcast_object(file_path,
                                   root_rank=0,
                                   name='de_hvd_broadcast_file_path')
        _, _, tf_return_de_dir = _get_de_dir_from_file_path(file_path)
        if tf_return_de_dir != de_dir:
          self._de_handle_root_and_var_with_func(
              de_dir=tf_return_de_dir, func=self._de_var_fs_save_funtion)
        self._hvd.join(
        )  # Sync for avoiding files conflict and rank finish early
        de_dir = _rank0_delete_files_and_return_de_dir(file_path)
        self._hvd.join()  # Sync for avoiding files conflict
      else:
        file_path = self._hvd.broadcast_object(
            None, root_rank=0, name='de_hvd_broadcast_file_path')
        _, _, de_dir = _get_de_dir_from_file_path(file_path)
        self._de_handle_root_and_var_with_func(
            de_dir=de_dir, func=self._de_var_fs_save_funtion)
        self._hvd.join(
        )  # Sync for avoiding files conflict and rank finish early
        self._hvd.join()  # Sync for avoiding files conflict
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
      return super(DECheckpoint, self)._write(file_prefix=file_prefix,
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
      if hasattr(super(DECheckpoint, self), '_write'):
        return super(DECheckpoint, self)._write(file_prefix=file_prefix,
                                                options=options,
                                                *args,
                                                **kwargs)
      else:
        return super(DECheckpoint, self).write(file_prefix=file_prefix,
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
    de_dir = _get_de_variable_folder_dir(save_path, global_step)

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
        de_dir = _get_de_variable_folder_dir(
            save_path,
            (corresponding_ckpt_index[0].split('-')[-1].split('.index')[0]))
      if len(corresponding_ckpt_index) > 0:
        impl_save_path = corresponding_ckpt_index[0].split('.index')[0]
        if global_step is None:
          tf_logging.warning(
              f'Arg save_path {save_path} is illegal or not existing. Now using index {impl_save_path}'
          )

    self._redirect_new_de_dir(de_dir)
    result = super(DECheckpoint, self).restore(save_path=impl_save_path,
                                               options=options,
                                               *args,
                                               **kwargs)

    if not os.path.exists(de_dir):
      tf_logging.warning(
          f'TFRADynamicEmbedding directory {de_dir} is not existing.')
    if self._hvd is not None:
      if self._hvd.size() > 1:
        self._hvd.join()  # Sync for avoiding files conflict
    return result

  def read(self, save_path, options=None, *args, **kwargs):
    save_path_split = save_path.split('-')
    save_path_pattern = ''.join(save_path_split[0:-1])
    global_step = save_path_split[-1]
    if not global_step.isdigit():
      global_step = None
    de_dir = _get_de_variable_folder_dir(save_path, global_step)

    self._redirect_new_de_dir(de_dir)
    result = super(DECheckpoint, self).read(save_path=save_path,
                                            options=options,
                                            *args,
                                            **kwargs)

    if not os.path.exists(de_dir):
      tf_logging.warning(
          f'TFRADynamicEmbedding directory {de_dir} is not existing.')
    if self._hvd is not None:
      if self._hvd.size() > 1:
        self._hvd.join()  # Sync for avoiding files conflict
    return result


def _get_de_variable_folder_dir(save_path: str, global_step: str = None):
  save_path_parent = os.path.dirname(save_path)
  if global_step is not None:
    de_variable_folder_dir = os.path.join(
        save_path_parent, "TFRADynamicEmbedding-{}".format(global_step))
  else:
    de_variable_folder_dir = os.path.join(save_path_parent,
                                          "TFRADynamicEmbedding")
  return de_variable_folder_dir


def _delete_redundant_de_dir(ckpt_index_list: list):
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


def _get_de_dir_from_file_path(file_path):
  file_prefix_split = file_path.split('-')
  file_prefix_pattern = ''.join(file_prefix_split[0:-1])
  global_step = file_prefix_split[-1]
  if not global_step.isdigit():
    global_step = None
  de_dir = _get_de_variable_folder_dir(file_path, global_step)
  return file_prefix_pattern, global_step, de_dir


def _rank0_delete_files_and_return_de_dir(file_path):
  file_prefix_pattern, global_step, de_dir = _get_de_dir_from_file_path(
      file_path)
  if global_step is not None:
    ckpt_index_list = file_io.get_matching_files(file_prefix_pattern +
                                                 '-*.index')
    _delete_redundant_de_dir(
        ckpt_index_list
    )  # Compatible with automatic sweep function of checkpointmanager
  return de_dir


def patch_tf_checkpoint():
  ckpt_base.CheckpointPosition = DECheckpointPosition


class DECheckpointPosition(ckpt_base.CheckpointPosition):

  def _redirect_new_de_dir(self, named_saveables, de_dir):
    for saveable in named_saveables.values():
      if type(saveable).__name__ in de_fs_sub_saveable_class_names:
        if hasattr(saveable, '_saver_config'):
          saveable._saver_config.save_path = de_dir

  def _single_restoration_from_checkpoint_position(self, checkpoint_position,
                                                   visit_queue):
    restore_ops, tensor_saveables, python_saveables = \
      super(DECheckpointPosition, self)._single_restoration_from_checkpoint_position(
        checkpoint_position, visit_queue
      )
    _, _, de_dir = _get_de_dir_from_file_path(self._checkpoint.save_path_string)
    self._redirect_new_de_dir(named_saveables, de_dir)
    return restore_ops, tensor_saveables, python_saveables

  def gather_ops_or_named_saveables(self):
    result_tuple = super(DECheckpointPosition,
                         self).gather_ops_or_named_saveables()
    named_saveables = result_tuple[1]
    registered_savers = None
    if len(result_tuple) == 4:
      registered_savers = result_tuple[3]
    _, _, de_dir = _get_de_dir_from_file_path(self._checkpoint.save_path_string)
    self._redirect_new_de_dir(named_saveables, de_dir)
    return result_tuple
