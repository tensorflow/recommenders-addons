import horovod.tensorflow as hvd
import tensorflow as tf


class CustomSaveHook(tf.compat.v1.train.SessionRunHook):

  def __init__(self, ckpt_dir, global_step, worker_id):
    self._ckpt_dir = ckpt_dir
    self._global_step = global_step
    self._saver = tf.compat.v1.train.Saver(sharded=True,
                                           allow_empty=True,
                                           max_to_keep=1)
    self._worker_id = worker_id
    super(CustomSaveHook, self).__init__()

  def end(self, session):
    global_step = session.run(self._global_step)
    if self._worker_id == 0 and self._ckpt_dir:
      # only save checkpoint once when the train is finished.
      self._saver.save(session, self._ckpt_dir, global_step)


class HorovodSyncHook(tf.compat.v1.train.SessionRunHook):

  def __init__(self, device=''):
    hvd.init()
    with tf.device(device):
      self._bcast_op = hvd.broadcast_global_variables(0)
      self._exit_op = hvd.join()

    self._broadcast_done = False
    super(HorovodSyncHook, self).__init__()

  def after_run(self, run_context, run_values):
    if self._broadcast_done:
      return
    run_context.session.run(self._bcast_op)

    self._broadcast_done = True

  def end(self, session):
    session.run(self._exit_op)
