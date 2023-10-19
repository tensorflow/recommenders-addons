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

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow_recommenders_addons.dynamic_embedding.python.ops.tf_save_restore_patch import _DynamicEmbeddingSaver

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import training_util
from tensorflow.python.util import compat


class DEHvdSaver(_DynamicEmbeddingSaver):

  def save(self,
           sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True,
           write_state=True,
           strip_default_attrs=False,
           save_debug_info=False,
           *args,
           **kwargs):
    """Overwrite tf.train.Saver class
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

    def _saver_save():
      return super(DEHvdSaver,
                   self).save(sess=sess,
                              save_path=save_path,
                              global_step=global_step,
                              latest_filename=latest_filename,
                              meta_graph_suffix=meta_graph_suffix,
                              write_meta_graph=write_meta_graph,
                              write_state=write_state,
                              strip_default_attrs=strip_default_attrs,
                              save_debug_info=save_debug_info,
                              *args,
                              **kwargs)

    if hvd is None:
      return _saver_save()
    else:
      if hvd.rank() == 0:
        return _saver_save()
      else:
        save_path = compat.as_str(save_path)
        if global_step is not None:
          if not isinstance(global_step, compat.integral_types):
            global_step = training_util.global_step(sess, global_step)
        else:
          if os.path.basename(
              save_path) == latest_filename and not self._sharded:
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
                save_path_parent,
                "TFRADynamicEmbedding-{:08d}".format(global_step))
        else:
          de_variable_folder_dir = os.path.join(save_path_parent,
                                                "TFRADynamicEmbedding")
        if not self._is_empty:
          try:
            if context.executing_eagerly():
              with ops.name_scope("FileSystemSaver", "save_to_file_system",
                                  []) as name:
                self._de_var_fs_save_dir = array_ops.placeholder(
                    dtype=dtypes.string,
                    shape=(),
                    name="de_var_file_system_save_dir")
                self._de_save_ops = self._get_dynamic_embedding_save_ops()
            else:
              sess.run(self._de_save_ops,
                       {self._de_var_fs_save_dir: de_variable_folder_dir})
          except (errors.FailedPreconditionError, errors.NotFoundError) as exc:
            if not gfile.IsDirectory(save_path_parent):
              exc = ValueError(
                  "Parent directory of {} doesn't exist, can't save.".format(
                      save_path))
            raise exc
