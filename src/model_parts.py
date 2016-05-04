"""
This contains fundamental model parts.
"""

import re

import tensorflow as tf

def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def variable(name, shape, init_type="const", init_val=0.01, wd=None):
    """
    Helper to create a Variable.

    :param str name: name of the variable
    :param list[int] shape: shape of the variable
    :param str init_type: initializer type of the variable (const or xavier)
    :return: variable
    :rtype: Variable of Tensor
    """
    if init_type == "const":
        initializer = tf.constant_initializer(init_val, dtype=tf.float32)
    elif init_type == "xavier":
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
    else:
        raise ValueError("unknown init_type {}".format(init_type))

    var = tf.get_variable(name, shape, initializer=initializer)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


#http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
def batch_norm(x):
    mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
    depth = x.get_shape()[-1]
    beta = variable("beta", shape=depth, init_val=0.0)
    gamma = variable("gamma", shape=depth, init_val=1.0)

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-05)
