#!/usr/bin/env python
# ==============================================================================
# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================


from datetime import datetime
import os
import time

import numpy as np
import tensorflow as tf

import model
import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("dir_gray", "../data/grayscales",
                           "directory contains grayscale images. folder structure is defined in input.py")
tf.app.flags.DEFINE_string("dir_color", "../data/colors",
                           "directory contains color images. folder structure is defined in input.py")
tf.app.flags.DEFINE_string("list_train", "../data/list_train.txt",
                           "directory list of train set")

tf.app.flags.DEFINE_string('train_dir', '../events',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


########  optimization parameters #############
tf.app.flags.DEFINE_float('lr', 0.1,
                          "initial learning rate")

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 8770
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 789

tf.app.flags.DEFINE_integer('decay_steps', 20000,
                        "learning rate decay steps")
tf.app.flags.DEFINE_float('decay', 0.1,
                        "learning rate decay factor")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def make_data_directory_list():
    """
    gets data directory
    """
    dirs_gray = []
    dirs_color = []
    with open(FLAGS.list_train, "r") as f:
        for l in f.readlines():
            l = l[:-1]
            dirs_gray.append(os.path.join(FLAGS.dir_gray, l))
            dirs_color.append(os.path.join(FLAGS.dir_color, l))

    return dirs_gray, dirs_color


def get_train_op(total_loss, global_step):
    """
    gets train operator
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    :param Tensor total_loss: Total loss from loss().
    :param Tensor global_step: Integer Variable counting the number of training steps
    :return: op for training updates variables
    """

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.lr,
                                    global_step,
                                    FLAGS.decay_steps,
                                    FLAGS.decay,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Compute gradients.
    with tf.control_dependencies([total_loss]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def train():
    with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False)

      dirs_gray, dirs_color = make_data_directory_list()
      gray_images, color_images = input.read_dirs(dirs_gray, dirs_color, is_train=True)

      inferenced = model.inference(gray_images)
      raw_loss, total_loss = model.loss(inferenced, color_images)

      train_op = get_train_op(total_loss, global_step)

      summary_op = tf.merge_all_summaries()
      saver = tf.train.Saver(tf.all_variables())

      init = tf.initialize_all_variables()

      sess = tf.Session(config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement))
      sess.run(init)

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

      for step in xrange(FLAGS.max_steps):
          start_time = time.time()
          _,value_raw_loss, value_total_loss = sess.run([train_op,raw_loss, total_loss])
          duration = time.time() - start_time


          if step % 10 == 0:
              num_examples_per_step = FLAGS.batch_size
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = float(duration)
              format_str = ('%s: step %d, raw_loss = %.2f, total_loss = %.2f (%.1f examples/sec; %.3f '
              'sec/batch)')
              print (format_str % (datetime.now(), step, value_raw_loss, value_total_loss,
              examples_per_sec, sec_per_batch))

          if step % 100 == 0:
              summary_str = sess.run(summary_op)
              summary_writer.add_summary(summary_str, step)


          # Save the model checkpoint periodically.
          if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
              checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
