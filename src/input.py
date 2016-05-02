"""
handle input data.

This module assumes folder tree such as
  movies/
    grayscales/
      01_001/
        img00004.png
        img00018.png
      01_002/
    skip_frames/
      01_001/
        img00004.png
        img00018.png

Image files which have a same name have same images of color and grayscale.


"""

import os
import glob

import tensorflow as tf


_IMAGE_EXT = "png"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_width', 640,
                            """width of image""")
tf.app.flags.DEFINE_integer('image_height', 360,
                            """width of image""")
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """minibatch size""")


def read_dirs(list_gray_dir, list_color_dir=None, is_train=False):
    """
    gets image batch tensor from list of directories.
    Each tensor is preprocessed and tensor is shuffled randomly.
    When you need only gray scale images for test, you set list_color_dir=None.

    :param list list_color_dir: list of input directories. Each directory has color image files.
    :param list list_gray_dir: list of input directories. Each directory has grayscale image files.
    :pram bool is_shuffle: whether you need shuffle tensor or not
    :return: tensor
    :rtype: tuple(Tensor) or Tensor
    """
    if is_train and list_color_dir is None:
        raise ValueError("requires list_color_dir in train")

    def _process_each_list(list_dir, is_color):
        images = []

        for cur_dir in list_dir:
            new_images = glob.glob(cur_dir + "/*." + _IMAGE_EXT)
            if len(new_images) == 0:
                raise ValueError("{} doesn't have image".format(cur_dir))

            images += new_images

        queue_images = tf.train.string_input_producer(images, shuffle=False, name="image_queue")
        image = read_image(queue_images, is_color)
        processed_image = preprocess_image(image)

        return processed_image

    gray_image = _process_each_list(list_gray_dir, is_color=False)
    if list_color_dir is not None:
        color_image = _process_each_list(list_color_dir, is_color=True)

    num_preprocess_threads = 4
    min_queue_examples = 16
    batch_size = FLAGS.batch_size

    if is_train:
        # flipping color and grayscale images simultaneously
        # MEMO: should use conditioning for performance?
        concat_image = tf.concat(2, [gray_image, color_image])
        flipped_image = tf.image.random_flip_left_right(concat_image)
        # shape of flipped_image = (height, width, 1 + 3)
        gray_image = tf.slice(flipped_image, [0, 0, 0], [FLAGS.image_height, FLAGS.image_width, 1])
        color_image = tf.slice(flipped_image, [0, 0, 1], [FLAGS.image_height, FLAGS.image_width, 3])


        gray_images, color_images = tf.train.shuffle_batch(
            [gray_image, color_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

        # Display the training images in the visualizer.
        tf.image_summary('gray_images', gray_images)
        tf.image_summary('color_images', color_images)

        return gray_images, color_images
    else:
        gray_images = tf.train.batch(
            [gray_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

        tf.image_summary('gray_images', gray_images)

        return gray_images



def read_image(filename_queue, is_color=False):
    """
    reads and preprocessed an image file.

    :param Queue filename_queue: string tensor queue containing image file path.
    :return: image tensor
    :rtype: tensor
    """

    reader = tf.WholeFileReader()
    _, content = reader.read(filename_queue)

    if is_color:
        num_channels = 3
    else:
        num_channels = 1

    image = tf.image.decode_png(content, channels=num_channels)

    # add type hint
    image = tf.reshape(image, [FLAGS.image_height, FLAGS.image_width, num_channels])

    return image

def preprocess_image(image):
    """
    preprocesses image tensor by such as scaling and mean substraction.
    This does NOT add noise such as flipping to use for train and test.

    :param Tensor image: tensor of image
    :return: preprocessed image
    :rtype: Tensor
    """
    float_image = tf.cast(image, dtype=tf.float32)
    return float_image / 255


