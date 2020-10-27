from functools import partial
from absl import app
from absl import flags

import gin
import numpy as np
import cv2
import tensorflow as tf

from uflow import uflow_augmentation
from uflow import uflow_data
from uflow import uflow_utils
# pylint:disable=unused-import
from uflow import uflow_flags
from uflow import uflow_plotting
from uflow.uflow_net import UFlow

from datetime import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

FLAGS = flags.FLAGS

# Create a description of the features.
feature_description = {
    'seq_ids': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'el_ids': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


def parse_data(proto, height, width):
  """Parse features from byte-encoding to the correct type and shape.

  Args:
    proto: Encoded data in proto / tf-sequence-example format.
    height: int, desired image height.
    width: int, desired image width.

  Returns:
    A sequence of images as tf.Tensor of shape
    [sequence length, height, width, 3].
  """
  #print('proto', proto)
  #tf.print(proto)
  # Parse context and image sequence from protobuffer.
  unused_context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
      proto,
      context_features={
          'height': tf.io.FixedLenFeature([], tf.int64),
          'width': tf.io.FixedLenFeature([], tf.int64)
      },
      sequence_features={
          'seq_ids': tf.io.FixedLenSequenceFeature([], tf.int64),
          'el_ids': tf.io.FixedLenSequenceFeature([], tf.int64)
          #'images': tf.io.FixedLenSequenceFeature([], tf.string)
      })

  # Deserialize images to float32 tensors.
  def deserialize(image_raw):
    image_uint = tf.image.decode_png(image_raw)
    image_float = tf.image.convert_image_dtype(image_uint, tf.float32)
    return image_float


  #def deserialize(num):
  #
  #    return tf.convert_to_tensor(num, tf.int64)

  #ids = tf.io.parse_single_example(sequence_parsed, feature_description)
  #seq_ids = tf.map_fn(deserialize, sequence_parsed['seq_ids'], dtype=tf.int64)
  #el_ids = sequence_parsed['el_ids']

  #images = tf.map_fn(deserialize, sequence_parsed['images'], dtype=tf.float32)

  # Resize images.
  #images = uflow_utils.resize(images, height, width, is_flow=False)

  return sequence_parsed['seq_ids'] * 21 + sequence_parsed['el_ids']



def main(unused_argv):
    apply_augmentation = True
    seq_len = FLAGS.seq_len
    batch_size = FLAGS.batch_size
    height = FLAGS.height
    width = FLAGS.width
    shuffle_buffer_size = FLAGS.shuffle_buffer_size
    mode = 'train'
    seed = 41
    path = '../datasets/KITTI_flow_multiview/KITTI_flow_multiview_test_384x1280-tfrecords'
    path = '../datasets/KITTI_flow_multiview_test_384x1280_ids_fullseq-tfrecords'

    if ',' in path:
        l = path.split(',')
        d = '/'.join(l[0].split('/')[:-1])
        l[0] = l[0].split('/')[-1]
        paths = [os.path.join(d, x) for x in l]
    else:
        paths = [path]

    # Generate list of filenames.
    # pylint:disable=g-complex-comprehension
    files = [os.path.join(d, f) for d in paths for f in tf.io.gfile.listdir(d)]
    num_files = len(files)
    if 'train' in mode:
        rgen = np.random.RandomState(seed)
        rgen.shuffle(files)
    ds = tf.data.Dataset.from_tensor_slices(files)

    if shuffle_buffer_size:
        ds = ds.shuffle(num_files)
    # Create a nested dataset.
    ds = ds.map(tf.data.TFRecordDataset)
    # Parse each element of the subsequences and unbatch the result.
    # pylint:disable=g-long-lambda
    ds = ds.map(lambda x: x.map(
        lambda y: parse_data(y, height, width),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch())
    # Slide a window over each dataset, combine either by interleaving or by
    # sequencing the result (produces a a nested dataset)
    window_fn = lambda x: x.window(size=seq_len, shift=1, drop_remainder=True)
    # Interleave subsequences (too long cycle length causes memory issues).
    ds = ds.interleave(
        window_fn,
        cycle_length=1 if 'video' in mode else min(10, num_files),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle_buffer_size:
        # Shuffle subsequences.
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Put repeat after shuffle.
    ds = ds.repeat()
    # Flatten the nested dataset into a batched dataset.
    ds = ds.flat_map(lambda x: x.batch(seq_len))
    # Prefetch a number of batches because reading new ones can take much longer
    # when they are from new files.
    ds = ds.prefetch(10)

    #augmentation_fn = partial(
    #    uflow_augmentation.apply_augmentation,
    #    crop_height=height,
    #    crop_width=width)

    #if apply_augmentation:
    #    ds = ds.map(augmentation_fn)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    #ds = ds.map(_ensure_shapes())
    train_it = tf.compat.v1.data.make_one_shot_iterator(ds)

    seq_ids = []
    el_ids = []

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    for epoch in range(10):
        num_steps = 1000*1
        print('len(train_it)')
        for _, batch in zip(range(num_steps), train_it):
            #print(batch)

            seq_ids.append(batch[0][0].numpy())
            seq_ids.append(batch[0][1].numpy())

        plt.hist(seq_ids, bins=4021)
        plt.show()
        cv2.waitKey(0)
    pass

    '''
        images, labels = batch
        ground_truth_flow = labels.get('flow_uv', None)
        ground_truth_valid = labels.get('flow_valid', None)
        ground_truth_occlusions = labels.get('occlusions', None)
        images_without_photo_aug = labels.get('images_without_photo_aug', None)

        img1 = images_without_photo_aug[0][0]
        img2 = images_without_photo_aug[0][1]
        img = tf.concat([img1, img2], 1)
        img_vis = (img[:, :, ::-1].numpy() * 255.).astype(np.uint8)
        cv2.imshow('a', img_vis)
        cv2.waitKey(0)
    '''



if __name__ == '__main__':
  app.run(main)
