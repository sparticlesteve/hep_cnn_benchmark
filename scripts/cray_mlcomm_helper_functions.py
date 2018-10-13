from __future__ import print_function
import tensorflow as tf
import numpy as np
import ml_comm as mc
from functools import reduce


class BcastTensors(tf.train.SessionRunHook):
    """
    A TensorFlow hook which synchronizes all model weights after the training
    session is created.
    """

    def __init__(self):
        self.bcast = None

    def begin(self):
        if not self.bcast:
            new_vars   = mc.broadcast(tf.trainable_variables(),0)
            self.bcast = tf.group(*[tf.assign(v,new_vars[k])
                                  for k,v in enumerate(tf.trainable_variables())])

    def validate(self, session):
        py_all_vars = [session.run(v) for v in tf.trainable_variables()]
        if (mc.check_buffers_match(py_all_vars,1) != 0):
            raise ValueError("ERROR: not all processes have the same initial model!")
        else:
            print("Initial model is consistent on all ranks")

    def after_create_session(self, session, coord, validate_init=True):
        session.run(self.bcast)
        if validate_init:
            self.validate(session)

class InitConfigBcastTensors(tf.train.SessionRunHook):
    """
    A more complicated broadcast hook which also takes care of initializing
    the Cray PE ML Plugin, using the actual model weights to size the
    communication buffers.
    """

    def __init__(self, max_step_global, sync_frac, thread_team, trans_alg,
                 perf_freq=200):
        self.bcast = None
        self.max_steps = max_step_global
        self.sync_frac = sync_frac
        self.team = thread_team  
        self.alg = trans_alg
        self.perf_freq = perf_freq

    def begin(self):
        if not self.bcast:
            # initialize the Cray PE ML Plugin
            buffer_size = sum([reduce(lambda x, y: x*y, v.get_shape().as_list())
                               for v in tf.trainable_variables()])
            mc.init(1, 1, buffer_size, "tensorflow")

            # config the thread team (correcting the number of epochs for the effective batch size)
            max_steps_local = int(self.max_steps/mc.get_nranks())
            thread_team = self.team 
            transition_algorithm = self.alg 
            print_performance_freq = self.perf_freq 
            sync_steps = max(int(self.sync_frac * self.max_steps / mc.get_nranks()), 1)
            mc.config_team(thread_team, transition_algorithm,
                           ksteps=sync_steps, max_steps=max_steps_local,
                           verbosity=2, perf_freq=print_performance_freq)

            # prepare the bcast op
            new_vars   = mc.broadcast(tf.trainable_variables(), 0)
            self.bcast = tf.group(*[tf.assign(v,new_vars[k])
                                    for k,v in enumerate(tf.trainable_variables())])

    def validate(self, session):
        py_all_vars = [session.run(v) for v in tf.trainable_variables()]
        if (mc.check_buffers_match(py_all_vars,1) != 0):
            raise ValueError("ERROR: not all processes have the same initial model!")
        else:
            print("Initial model is consistent on all ranks")

    def after_create_session(self, session, coord, validate_init=True):
        session.run(self.bcast)
        if validate_init:
            self.validate(session)

class AverageTrainMetrics(tf.train.SessionRunHook):

  def __init__(self,loss, metrics, log_freq, batch_size, lr, epoch):
      self.log_freq = log_freq
      self.lr = lr
      self.epoch_true = 0
      self.epoch = epoch
      self.batch_size = batch_size * mc.get_nranks()
      self.both = [loss,metrics['accuracy'][0],lr,epoch]
      self.samps = 0.
      self.perf = 0.
      self.step = 0
      self.sums = 0
      self.start_time = None

  def begin(self):
      self.step = 0
      self.epoch_true += 1
      self.start_time = time.time()

  def before_run(self, run_context):
      self.step += 1
      return tf.train.SessionRunArgs(self.both)  # Asks for loss value.

  def after_run(self, run_context, run_values):
      if self.step % self.log_freq == 0:

          current_time = time.time()
          duration = current_time - self.start_time
          self.start_time = current_time
          loss_value = np.asarray([run_values.results[0]],dtype='float32')
          acc_value = np.asarray([run_values.results[1]],dtype='float32')

          examples_per_sec = self.log_freq * self.batch_size / duration
          sec_per_batch = float(duration / self.log_freq)
          mc.average(loss_value)
          mc.average(acc_value)

          format_str = ('%s: step %d, loss = %.3f, acc = %.3f, (%.1f examples/sec; %.3f '
                        'sec/batch)')
          if (mc.get_rank() == 0):
              print ("available values = ", run_values)
              print (format_str % (datetime.now(), self.step, loss_value, acc_value,
                                   examples_per_sec, sec_per_batch ))

          self.samps = self.samps + examples_per_sec
          self.perf  = self.perf + sec_per_batch
          self.sums  = self.sums + 1

  def end(self, session):

      lr = session.run(self.lr)

      format_str = ('TRAIN Session ENDED at %s: step %d (%.1f examples/sec; %.3f '
                    'sec/batch), learning rate: %.5f')
      self.epoch_true = tf.train.global_step(session, self.epoch)/(self.step+1)

      if (mc.get_rank() == 0):
          print('Epoch: ', self.epoch_true)
          print('global_step: %s' % tf.train.global_step(session, self.epoch))
          print (format_str % (datetime.now(), self.step, self.samps/self.sums,
                               self.perf/self.sums, lr))

class AverageEvalMetrics(tf.train.SessionRunHook):

    def __init__(self,loss,metrics,batch_size):
        self.batch_size = batch_size * mc.get_nranks()
        self.local_batch_size = batch_size
        self.both = [loss,metrics['accuracy'][0]]
        self.samps = 0
        self.perf = 0
        self.step = 0
        self.loss = np.asarray([0.],dtype='float32')
        self.acc = np.asarray([0.],dtype='float32')
        self.start_time = None

    def begin(self):
        self.step = 0
        self.start_time = time.time()

    def before_run(self, run_context):
        self.step += 1
        return tf.train.SessionRunArgs(self.both)  # Asks for loss value.

    def after_run(self, run_context, run_values):

        current_time = time.time()
        duration = current_time - self.start_time
        self.start_time = current_time

        loss_value = np.asarray([run_values.results[0]],dtype='float32')
        acc_value = np.asarray([run_values.results[1]],dtype='float32')
        examples_per_sec = self.batch_size / duration
        sec_per_batch = duration

        self.samps = self.samps + examples_per_sec
        self.perf  = self.perf + sec_per_batch
        self.loss  = self.loss + loss_value
        self.acc   = self.acc + acc_value

        if (mc.get_rank() == 0):
            print("Eval step {:9d}".format(self.step))

    def end(self,session):

        self.loss = self.loss / self.step
        self.acc  = self.acc / self.step

        mc.average(self.loss)
        mc.average(self.acc)

        format_str = ('EVAL Session ENDED at %s: step %d, loss = %.3f, accuracy = %.3f '
                      '(%.1f examples/sec; %.3f sec/batch)')

        if (mc.get_rank() == 0):
            print (format_str % (datetime.now(), self.step, self.loss, self.acc,
                                 self.samps/self.step, self.perf/self.step))
