# Copyright 2020 The TensorFlow Recommenders-Addons Authors.
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
"""patch on tensorflow"""

import inspect
import functools
import os.path
import re

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.dynamic_embedding_variable \
  import load_de_variable_from_file_system

try:
  from keras.saving.saved_model import save as keras_saved_model_save
except:
  keras_saved_model_save = None
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.keras.saving.saved_model import save as tf_saved_model_save
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import saver
from tensorflow.python.training import training_util
try:  # tf version >= 2.10.0
  from tensorflow.python.checkpoint import checkpoint_management
  from tensorflow.python.checkpoint import checkpoint_options
  from tensorflow.python.checkpoint import functional_saver
except:
  from tensorflow.python.training import checkpoint_management
  from tensorflow.python.training.saving import checkpoint_options
  from tensorflow.python.training.saving import functional_saver
from tensorflow.python.util import compat
from tensorflow.python.util import nest

tf_original_save_func = tf_saved_model_save.save
if keras_saved_model_save is not None:
  keras_original_save_func = keras_saved_model_save.save

de_fs_saveable_class_names = [
    '_DynamicEmbeddingVariabelFileSystemSaveable',
]

de_fs_sub_saveable_class_names = [
    '_DynamicEmbeddingShardFileSystemSaveable',
]


def _de_var_fs_save_fn(trackables, file_prefix):
  variables_folder_dir = string_ops.regex_replace(file_prefix,
                                                  pattern='/([^/]*)$',
                                                  rewrite='')
  for obj_prefix, obj in trackables.items():
    if not hasattr(obj, 'saveable'):
      continue
    saveable = obj.saveable
    if type(saveable).__name__ == '_DynamicEmbeddingVariabelFileSystemSaveable':
      if saveable._saver_config.save_path:
        de_variable_folder_dir = saveable._saver_config.save_path
      else:
        de_variable_folder_dir = string_ops.string_join(
            [variables_folder_dir, 'TFRADynamicEmbedding'], separator='/')

      # Rewrite saved file name by user specified node information when use multi process distributed training such as horovod.
      # Because table shards in different process couldn't touch each other, all origin shards name would be '_mht_1of1'.
      obj.save_to_file_system(de_variable_folder_dir,
                              proc_size=saveable.proc_size,
                              proc_rank=saveable.proc_rank)
  return []


def _de_var_fs_restore_fn(trackables, merged_prefix):
  variables_folder_dir = string_ops.regex_replace(merged_prefix,
                                                  pattern='/([^/]*)$',
                                                  rewrite='')
  load_ops = tf_utils.ListWrapper([])
  for obj_prefix, obj in trackables.items():
    if not hasattr(obj, 'saveable'):
      continue
    saveable = obj.saveable
    if type(saveable).__name__ == '_DynamicEmbeddingVariabelFileSystemSaveable':
      with ops.name_scope(saveable._restore_name, "dynamic_embedding_restore"):
        if saveable._saver_config.save_path:
          de_variable_folder_dir = saveable._saver_config.save_path
        else:
          de_variable_folder_dir = string_ops.string_join(
              [variables_folder_dir, 'TFRADynamicEmbedding'], separator='/')
        load_ops.as_list().append(
            load_de_variable_from_file_system(
                saveable.op, de_variable_folder_dir, saveable.proc_size,
                saveable.proc_rank, saveable._saver_config.buffer_size))
  return load_ops.as_list()


try:  # tf version <= 2.15

  class _DynamicEmbeddingSingleDeviceSaver(functional_saver._SingleDeviceSaver):

    def save(self, file_prefix, options=None):
      """Save the saveable objects to a checkpoint with `file_prefix`.

      Args:
        file_prefix: A string or scalar string Tensor containing the prefix to
          save under.
        options: Optional `CheckpointOptions` object.
      Returns:
        An `Operation`, or None when executing eagerly.
      """
      options = options or checkpoint_options.CheckpointOptions()
      tensor_names = []
      tensors = []
      tensor_slices = []
      save_ops = tf_utils.ListWrapper([])
      variables_folder_dir = string_ops.regex_replace(
          file_prefix, pattern='/([^/]*)/([^/]*)$', rewrite='')
      for saveable in self._saveable_objects:
        if type(saveable).__name__ in de_fs_sub_saveable_class_names:
          if saveable._saver_config.save_path:
            de_variable_folder_dir = saveable._saver_config.save_path
          else:
            de_variable_folder_dir = string_ops.string_join(
                [variables_folder_dir, 'TFRADynamicEmbedding'], separator='/')

          # Rewrite saved file name by user specified node information when use multi process distributed training such as horovod.
          # Because table shards in different process couldn't touch each other, all origin shards name would be '_mht_1of1'.
          save_file_name = re.sub(
              r'_mht_([^/]*)of([^/]*)',
              '_mht_' + str(saveable.local_shard_idx + 1) + 'of' +
              str(saveable.local_shard_num) + '_rank' +
              str(saveable.proc_rank) + '_size' + str(saveable.proc_size),
              saveable.op._name)
          _DynamicEmbeddingShardSaveable_save_op = saveable.op.save_to_file_system(
              de_variable_folder_dir,
              file_name=save_file_name,
              buffer_size=saveable._saver_config.buffer_size)
          save_ops.as_list().append(_DynamicEmbeddingShardSaveable_save_op)
        for spec in saveable.specs:
          tensor = spec.tensor
          # A tensor value of `None` indicates that this SaveableObject gets
          # recorded in the object graph, but that no value is saved in the
          # checkpoint.
          if tensor is not None:
            tensor_names.append(spec.name)
            tensors.append(tensor)
            tensor_slices.append(spec.slice_spec)
      save_device = options.experimental_io_device or "cpu:0"
      with ops.device(save_device):
        tf_save_op = io_ops.save_v2(file_prefix, tensor_names, tensor_slices,
                                    tensors)
      save_ops.as_list().append(tf_save_op)
      return control_flow_ops.group(save_ops.as_list())

    def restore(self, file_prefix, options=None):
      """Restore the saveable objects from a checkpoint with `file_prefix`.

      Args:
        file_prefix: A string or scalar string Tensor containing the prefix for
          files to read from.
        options: Optional `CheckpointOptions` object.

      Returns:
        A dictionary mapping from SaveableObject names to restore operations.
      """
      options = options or checkpoint_options.CheckpointOptions()
      restore_specs = []
      tensor_structure = []
      restore_ops = {}
      variables_folder_dir = string_ops.regex_replace(file_prefix,
                                                      pattern='/([^/]*)$',
                                                      rewrite='')

      for saveable in self._saveable_objects:
        saveable_class_name = type(saveable).__name__
        if saveable_class_name == '_DynamicEmbeddingVariabelFileSystemSaveable':
          with ops.name_scope(saveable._restore_name,
                              "dynamic_embedding_restore"):
            if saveable._saver_config.save_path:
              de_variable_folder_dir = saveable._saver_config.save_path
            else:
              de_variable_folder_dir = string_ops.string_join(
                  [variables_folder_dir, 'TFRADynamicEmbedding'], separator='/')
            restore_ops[saveable.name] = load_de_variable_from_file_system(
                saveable.op, de_variable_folder_dir, saveable.proc_size,
                saveable.proc_rank, saveable._saver_config.buffer_size)

      _unified_restore_saveable_objects = []
      for saveable in self._saveable_objects:
        _unified_restore_saveable_objects.append(saveable)
        saveable_tensor_structure = []
        tensor_structure.append(saveable_tensor_structure)
        for spec in saveable.specs:
          saveable_tensor_structure.append(spec.name)
          restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
      tensor_names, tensor_slices, tensor_dtypes = zip(*restore_specs)
      restore_device = options.experimental_io_device or "cpu:0"
      with ops.device(restore_device):
        restored_tensors = io_ops.restore_v2(file_prefix, tensor_names,
                                             tensor_slices, tensor_dtypes)
      structured_restored_tensors = nest.pack_sequence_as(
          tensor_structure, restored_tensors)
      for saveable, restored_tensors in zip(_unified_restore_saveable_objects,
                                            structured_restored_tensors):
        saveable_class_name = type(saveable).__name__
        if (saveable_class_name not in de_fs_saveable_class_names) and (
            saveable_class_name not in de_fs_sub_saveable_class_names):
          restore_ops[saveable.name] = saveable.restore(restored_tensors,
                                                        restored_shapes=None)
        elif (saveable_class_name in de_fs_saveable_class_names):
          restore_ops[saveable.name] = control_flow_ops.group([
              saveable.restore(restored_tensors, restored_shapes=None),
              restore_ops[saveable.name]
          ])
      return restore_ops
except:
  print(" _SingleDeviceSaver removed after tf version 2.15")


class _DynamicEmbeddingSaver(saver.Saver):

  def _get_dynamic_embedding_save_ops(self):
    save_ops = tf_utils.ListWrapper([])
    if not self._var_list:
      return control_flow_ops.group(save_ops.as_list())

    for var in self._var_list:
      de_var = None
      if isinstance(
          var,
          (de.FileSystemSaver._DynamicEmbeddingShardFileSystemSaveable,
           de.FileSystemSaver._DynamicEmbeddingVariabelFileSystemSaveable)):
        de_var = var._de_variable
      elif isinstance(var, de.Variable) and var._saveable_object_creator:
        de_var = var

      if de_var and isinstance(de_var._saveable_object_creator,
                               de.FileSystemSaver):
        if de_var._saveable_object_creator.config.save_path:
          de_variable_folder_dir = de_var._saveable_object_creator.config.save_path
        else:
          de_variable_folder_dir = self._de_var_fs_save_dir

        save_op = de_var.save_to_file_system(
            dirpath=de_variable_folder_dir,
            proc_size=de_var._saveable_object_creator.config.proc_size,
            proc_rank=de_var._saveable_object_creator.config.proc_rank,
            buffer_size=de_var._saveable_object_creator.config.buffer_size)
        save_ops.as_list().append(save_op)
    return control_flow_ops.group(save_ops.as_list())

  def _get_dynamic_embedding_restore_ops(self):
    restore_ops = tf_utils.ListWrapper([])
    if not self._var_list:
      return control_flow_ops.group(restore_ops.as_list())

    for var in self._var_list:
      de_var = None
      if isinstance(
          var,
          (de.FileSystemSaver._DynamicEmbeddingShardFileSystemSaveable,
           de.FileSystemSaver._DynamicEmbeddingVariabelFileSystemSaveable)):
        de_var = var._de_variable
      elif isinstance(var, de.Variable) and var._saveable_object_creator:
        de_var = var

      if de_var and isinstance(de_var._saveable_object_creator,
                               de.FileSystemSaver):
        if de_var._saveable_object_creator.config.save_path:
          de_variable_folder_dir = de_var._saveable_object_creator.config.save_path
        else:
          de_variable_folder_dir = self._de_var_fs_save_dir

        restore_op = de_var.load_from_file_system_with_restore_function(
            dirpath=de_variable_folder_dir,
            proc_size=de_var._saveable_object_creator.config.proc_size,
            proc_rank=de_var._saveable_object_creator.config.proc_rank,
            buffer_size=de_var._saveable_object_creator.config.buffer_size)
        restore_ops.as_list().append(restore_op)
    return control_flow_ops.group(restore_ops.as_list())

  def _build(self, checkpoint_path, build_save, build_restore):
    # TrainableWrapper and DEResourceVariable should not be save or restore parameter.
    from tensorflow_recommenders_addons.dynamic_embedding.python.ops.shadow_embedding_ops import DEResourceVariable
    filter_lambda = lambda x: (isinstance(x, de.TrainableWrapper)) or (
        isinstance(x, DEResourceVariable))
    if isinstance(self._var_list, dict):
      for key, value in self._var_list.items():
        if filter_lambda(value):
          self._var_list.pop(key)
    elif isinstance(self._var_list, list):
      _tmp_var_list = []
      for value in self._var_list:
        if not filter_lambda(value):
          _tmp_var_list.append(value)
      self._var_list = _tmp_var_list

    super(_DynamicEmbeddingSaver, self)._build(checkpoint_path, build_save,
                                               build_restore)

    with ops.name_scope("FileSystemSaver", "save_to_file_system", []) as name:
      self._de_var_fs_save_dir = array_ops.placeholder(
          dtype=dtypes.string, shape=(), name="de_var_file_system_save_dir")
      self._de_save_ops = self._get_dynamic_embedding_save_ops()
      self._de_restore_ops = self._get_dynamic_embedding_restore_ops()

  def save(self,
           sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True,
           write_state=True,
           strip_default_attrs=False,
           save_debug_info=False):
    # pylint: disable=line-too-long
    """Saves variables.

    This method runs the ops added by the constructor for saving variables.
    It requires a session in which the graph was launched.  The variables to
    save must also have been initialized.

    The method returns the path prefix of the newly created checkpoint files.
    This string can be passed directly to a call to `restore()`.

    Args:
      sess: A Session to use to save the variables.
      save_path: String.  Prefix of filenames created for the checkpoint.
      global_step: If provided the global step number is appended to `save_path`
        to create the checkpoint filenames. The optional argument can be a
        `Tensor`, a `Tensor` name or an integer.
      latest_filename: Optional name for the protocol buffer file that will
        contains the list of most recent checkpoints.  That file, kept in the
        same directory as the checkpoint files, is automatically managed by the
        saver to keep track of recent checkpoints.  Defaults to 'checkpoint'.
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
      write_meta_graph: `Boolean` indicating whether or not to write the meta
        graph file.
      write_state: `Boolean` indicating whether or not to write the
        `CheckpointStateProto`.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued
          Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of save_path and with `_debug` added before
        the file extension. This is only enabled when `write_meta_graph` is
        `True`

    Returns:
      A string: path prefix used for the checkpoint files.  If the saver is
        sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'
        is the number of shards created.
      If the saver is empty, returns None.

    Raises:
      TypeError: If `sess` is not a `Session`.
      ValueError: If `latest_filename` contains path components, or if it
        collides with `save_path`.
      RuntimeError: If save and restore ops weren't built.
    """
    # pylint: enable=line-too-long
    if not self._is_built and not context.executing_eagerly():
      raise RuntimeError(
          "`build()` should be called before save if defer_build==True")
    if latest_filename is None:
      latest_filename = "checkpoint"
    if self._write_version != saver_pb2.SaverDef.V2:
      tf_logging.warning(
          "*******************************************************")
      tf_logging.warning(
          "TensorFlow's V1 checkpoint format has been deprecated.")
      tf_logging.warning("Consider switching to the more efficient V2 format:")
      tf_logging.warning(
          "   `tf.train.Saver(write_version=tf.train.SaverDef.V2)`")
      tf_logging.warning("now on by default.")
      tf_logging.warning(
          "*******************************************************")

    if os.path.split(latest_filename)[0]:
      raise ValueError("'latest_filename' must not contain path components")

    save_path = compat.as_str(save_path)
    if global_step is not None:
      if not isinstance(global_step, compat.integral_types):
        global_step = training_util.global_step(sess, global_step)
      checkpoint_file = "%s-%d" % (save_path, global_step)
      if self._pad_step_number:
        # Zero-pads the step numbers, so that they are sorted when listed.
        checkpoint_file = "%s-%s" % (save_path, "{:08d}".format(global_step))
    else:
      checkpoint_file = save_path
      if os.path.basename(save_path) == latest_filename and not self._sharded:
        # Guard against collision between data file and checkpoint state file.
        raise ValueError(
            "'latest_filename' collides with 'save_path': '%s' and '%s'" %
            (latest_filename, save_path))

    if (not context.executing_eagerly()
        and not isinstance(sess, session.SessionInterface)):
      raise TypeError("'sess' must be a Session; %s" % sess)

    save_path_parent = os.path.dirname(save_path)

    if global_step is not None:
      de_variable_folder_dir = os.path.join(
          save_path_parent, "TFRADynamicEmbedding-{}".format(global_step))
      if self._pad_step_number:
        # Zero-pads the step numbers, so that they are sorted when listed.
        de_variable_folder_dir = os.path.join(
            save_path_parent, "TFRADynamicEmbedding-{:08d}".format(global_step))
    else:
      de_variable_folder_dir = os.path.join(save_path_parent,
                                            "TFRADynamicEmbedding")

    if not self._is_empty:
      try:
        if context.executing_eagerly():
          self._build_eager(checkpoint_file,
                            build_save=True,
                            build_restore=False)
          model_checkpoint_path = self.saver_def.save_tensor_name
        else:
          model_checkpoint_path = sess.run(
              self.saver_def.save_tensor_name,
              {self.saver_def.filename_tensor_name: checkpoint_file})
          sess.run(self._de_save_ops,
                   {self._de_var_fs_save_dir: de_variable_folder_dir})

        model_checkpoint_path = compat.as_str(model_checkpoint_path)
        if write_state:
          self._RecordLastCheckpoint(model_checkpoint_path)
          checkpoint_management.update_checkpoint_state_internal(
              save_dir=save_path_parent,
              model_checkpoint_path=model_checkpoint_path,
              all_model_checkpoint_paths=self.last_checkpoints,
              latest_filename=latest_filename,
              save_relative_paths=self._save_relative_paths)
          self._MaybeDeleteOldCheckpoints(meta_graph_suffix=meta_graph_suffix)
      except (errors.FailedPreconditionError, errors.NotFoundError) as exc:
        if not gfile.IsDirectory(save_path_parent):
          exc = ValueError(
              "Parent directory of {} doesn't exist, can't save.".format(
                  save_path))
        raise exc

    if write_meta_graph:
      meta_graph_filename = checkpoint_management.meta_graph_filename(
          checkpoint_file, meta_graph_suffix=meta_graph_suffix)
      if not context.executing_eagerly():
        with sess.graph.as_default():
          self.export_meta_graph(meta_graph_filename,
                                 strip_default_attrs=strip_default_attrs,
                                 save_debug_info=save_debug_info)

    if self._is_empty:
      return None
    else:
      return model_checkpoint_path

  def restore(self, sess, save_path):
    """Restores previously saved variables.

    This method runs the ops added by the constructor for restoring variables.
    It requires a session in which the graph was launched.  The variables to
    restore do not have to have been initialized, as restoring is itself a way
    to initialize variables.

    The `save_path` argument is typically a value previously returned from a
    `save()` call, or a call to `latest_checkpoint()`.

    Args:
      sess: A `Session` to use to restore the parameters. None in eager mode.
      save_path: Path where parameters were previously saved.

    Raises:
      ValueError: If save_path is None or not a valid checkpoint.
    """
    if self._is_empty:
      return
    if save_path is None:
      raise ValueError("Can't load save_path when it is None.")

    checkpoint_prefix = compat.as_text(save_path)
    if not checkpoint_management.checkpoint_exists_internal(checkpoint_prefix):
      raise ValueError("The passed save_path is not a valid checkpoint: " +
                       checkpoint_prefix)

    tf_logging.info("Restoring parameters from %s", checkpoint_prefix)
    save_path_parent = os.path.dirname(save_path)

    maybe_global_step = os.path.basename(save_path).split('-')[-1]
    matched_de_dir = os.path.join(save_path_parent,
                                  'TFRADynamicEmbedding-' + maybe_global_step)
    if os.path.exists(matched_de_dir):
      de_variable_folder_dir = matched_de_dir
    else:
      de_variable_folder_dir = os.path.join(save_path_parent,
                                            'TFRADynamicEmbedding')

    try:
      if context.executing_eagerly():
        self._build_eager(save_path, build_save=False, build_restore=True)
      else:
        sess.run(self.saver_def.restore_op_name,
                 {self.saver_def.filename_tensor_name: save_path})
        sess.run(self._de_restore_ops,
                 {self._de_var_fs_save_dir: de_variable_folder_dir})
    except errors.NotFoundError as err:
      # There are three common conditions that might cause this error:
      # 0. The file is missing. We ignore here, as this is checked above.
      # 1. This is an object-based checkpoint trying name-based loading.
      # 2. The graph has been altered and a variable or other name is missing.

      # 1. The checkpoint would not be loaded successfully as is. Try to parse
      # it as an object-based checkpoint.
      try:
        names_to_keys = saver.object_graph_key_mapping(save_path)
      except errors.NotFoundError:
        # 2. This is not an object-based checkpoint, which likely means there
        # is a graph mismatch. Re-raise the original error with
        # a helpful message (b/110263146)
        raise saver._wrap_restore_error_with_msg(
            err, "a Variable name or other graph key that is missing")

      # This is an object-based checkpoint. We'll print a warning and then do
      # the restore.
      tf_logging.warning(
          "Restoring an object-based checkpoint using a name-based saver. This "
          "may be somewhat fragile, and will re-build the Saver. Instead, "
          "consider loading object-based checkpoints using "
          "tf.train.Checkpoint().")
      self._object_restore_saver = saver.saver_from_object_based_checkpoint(
          checkpoint_path=save_path,
          var_list=self._var_list,
          builder=self._builder,
          names_to_keys=names_to_keys,
          cached_saver=self._object_restore_saver)
      self._object_restore_saver.restore(sess=sess, save_path=save_path)
    except errors.InvalidArgumentError as err:
      # There is a mismatch between the graph and the checkpoint being loaded.
      # We add a more reasonable error message here to help users (b/110263146)
      raise saver._wrap_restore_error_with_msg(
          err, "a mismatch between the current graph and the graph")


def patch_on_tf_save_restore():
  try:
    from tensorflow.python.saved_model.registration.registration import register_checkpoint_saver
    class_obj = de.Variable
    predicate = lambda x: isinstance(x, class_obj)
    prekwargs = {
        "package": "DECustomSaver",
        "name": class_obj.__name__,
        "predicate": predicate,
        "save_fn": _de_var_fs_save_fn,
        "restore_fn": _de_var_fs_restore_fn,
        "strict_predicate_restore": False
    }
    rcs_sig = inspect.signature(register_checkpoint_saver)
    kwargs = {}
    for param in rcs_sig.parameters.values():
      k_name = param.name
      kwargs[k_name] = prekwargs[k_name]
    register_checkpoint_saver(**kwargs)
  except:
    functional_saver._SingleDeviceSaver = _DynamicEmbeddingSingleDeviceSaver
  saver.Saver = _DynamicEmbeddingSaver
  # # Replace origin saving function is too dangerous.
  # tf_saved_model_save.save = functools.partial(de.keras.models._de_keras_save_func,
  #                                              tf_original_save_func)
  # if keras_saved_model_save is not None:
  #   keras_saved_model_save.save = functools.partial(de.keras.models._de_keras_save_func,
  #                                                   keras_original_save_func)
