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

"""
Builds the colorization network.


"""

import os
import re
import sys
import tarfile

import tensorflow as tf

import model_parts as parts
from model_parts import variable

FLAGS = tf.app.flags.FLAGS


def layer(inputs, kernel_size, channel, stride):
    """
    unit layer of this model.

    :parma Tensor inputs: input tensor, which is batch Tensor.
    :param int kernel_size: convolution kernel size
    :param int channel: output channels
    :parma int stride: convolution stride
    :return: output of layer
    :rtype: Tensor
    """
    wd = 1e-4
    input_depth = inputs.get_shape()[-1]

    kernel = variable('weights', shape=[kernel_size, kernel_size, input_depth, channel],
                      init_type="xavier", wd=wd)

    conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding='SAME')
    normalized = parts.batch_norm(conv)

    # pool1
    pool = tf.nn.max_pool(normalized, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool')

    relu = tf.nn.relu(pool, name="relu")
    parts.activation_summary(relu)

    return relu

def inference(images):
    """
    builds the colorization model.

    :param Tenor images: batch tesnor of grayscale images. shape is [image_height, image_width, 1]
    :return: inferenced color image, which has shape [image_height, image_width, 3]
    :rtype: Tensor
    """
    features = [images]
    inputs = images

    # 1/4
    with tf.variable_scope('layer1') as scope:
        outputs = layer(inputs, kernel_size=3, channel=32, stride=2)
        layer_features = tf.image.resize_images(outputs, FLAGS.image_height, FLAGS.image_width)
        features.append(layer_features)

        inputs = outputs


    # 1/2 ** 3
    for idx_layer, channel in zip([2, 3, 4], [64, 64, 64]):
    #for idx_layer, channel in zip([2, 3], [64, 64]):
        layer_name = "layer{}".format(idx_layer)

        with tf.variable_scope(layer_name) as scope:
            outputs = layer(inputs, kernel_size=3, channel=channel, stride=1)
            #outputs = layer(inputs, kernel_size=3, channel=channel, stride=2)
            layer_features = tf.image.resize_images(outputs, FLAGS.image_height, FLAGS.image_width)
            features.append(layer_features)

            inputs = outputs

    # concate nate
    all_features = tf.concat(3, features, "feature_concat")

    with tf.variable_scope("generator") as scope:
        input_depth = all_features.get_shape()[-1]
        kernel = variable('weights', shape=[5, 5, input_depth, 3],
                          init_type="xavier", wd=1e-05)

        conv = tf.nn.conv2d(all_features, kernel, [1, 1, 1, 1], padding='SAME')
        bias = variable("bias", shape=[3], init_type="const", wd=None)
        color_map = tf.nn.bias_add(conv, bias)

    return color_map

def loss(inferenced, original):
    """
    calculates loss value.
    The loss value contains difference between colors and regularization terms.

    :param Tensor inferenced: inferenced colors
    :param Tensor original: original colors (pixel range [0,1])
    :return: the loss value.
    :rtype: Tensor
    """
    loss = tf.nn.l2_loss(original - inferenced) / (FLAGS.image_height*FLAGS.image_width*3)
    tf.add_to_collection('losses', loss)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss, total_loss


def add_loss_summaries(raw_loss, total_loss):
    """
    Add summaries for losses

    Generates moving average for all losses and associated summariefs for
    visualizing the performance of the network.

    :param Tensor raw_loss: tensor of loss value without regularization terms.
    :param Tensor total_loss: tensor of loss value with regularization terms.
    :return: op for generating moving averages of losses.
    """

    loss_averages = tf.train.ExponentialMovingAverage(0.95, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply([raw_loss, total_loss])

    for l in [raw_loss, total_loss]:
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

