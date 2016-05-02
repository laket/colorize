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
import cv2

import model
import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("dir_gray", "../data/grayscales",
                           "directory contains grayscale images. folder structure is defined in input.py")
tf.app.flags.DEFINE_string("dir_color", "../data/colors",
                           "directory contains color images. folder structure is defined in input.py")
tf.app.flags.DEFINE_string("dir_out", "../output",
                           "directory of output images")

tf.app.flags.DEFINE_string("list_test", "../data/list_test.txt",
                           "directory list of train set")

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

def make_data_directory_list():
    """
    gets data directory
    """
    dirs_gray = []
    dirs_color = []
    with open(FLAGS.list_test, "r") as f:
        for l in f.readlines():
            l = l[:-1]
            dirs_gray.append(os.path.join(FLAGS.dir_gray, l))
            dirs_color.append(os.path.join(FLAGS.dir_color, l))

    return dirs_gray, dirs_color


def evaluate():
    with tf.Graph().as_default() as g, tf.device("/cpu:0"):
      dirs_gray, dirs_color = make_data_directory_list()

      gray_images = input.read_dirs(dirs_gray, list_color_dir=None, is_train=False)
      
      inferenced = model.inference(gray_images)

      int_images = tf.image.convert_image_dtype(inferenced, dtype=tf.uint8, saturate=True)

      sess = tf.Session(config=tf.ConfigProto())

      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)


      if ckpt and ckpt.model_checkpoint_path:
          saver = tf.train.Saver()
          saver.restore(sess, ckpt.model_checkpoint_path)
          global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
          print('No checkpoint file found')
          raise ValueError("invalid checkpoint")


      tf.train.start_queue_runners(sess=sess)

      count = 0
      #num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      for i in range(10):
          value_images = sess.run([int_images])[0]

          for cur_image in value_images:
              count += 1
              cv_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2BGR)

              cur_path = os.path.join(FLAGS.dir_out, "{}.png".format(count))
              cv2.imwrite(cur_path, cv_image)
        

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
