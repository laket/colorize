#!/usr/bin/env python

from datetime import datetime
import os
import time

import numpy as np
import tensorflow as tf

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

def train():
    with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False)

      dirs_gray, dirs_color = make_data_directory_list()
      gray_images, color_images = input.read_dirs(dirs_gray, dirs_color, is_train=True)


      summary_op = tf.merge_all_summaries()

      init = tf.initialize_all_variables()

      sess = tf.Session(config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement))
      sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        value_gray_images = sess.run([gray_images])
        duration = time.time() - start_time


        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            loss_value = 0.0
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
            'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
            examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
