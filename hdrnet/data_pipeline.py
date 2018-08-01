# Copyright 2016 Google Inc.
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

"""Data pipeline for HDRnet."""

import abc
import json
import logging
import magic
import numpy as np
import os
import random
import itertools
import pandas as pd

import tensorflow as tf
from tensorflow import FixedLenFeature

log = logging.getLogger("root")
log.setLevel(logging.INFO)

__all__ = [
    'LRNetDataPipeline'
]


def check_dir(dirname):
    fnames = os.listdir(dirname)
    if not os.path.isdir(dirname):
        log.error("Training dir {} does not exist".format(dirname))
        return False
    if not "input" in fnames:
        log.error("Training dir {} does not containt 'input' folder".format(dirname))
    if not "output" in fnames:
        log.error("Training dir {} does not containt 'output' folder".format(dirname))

    return True


class DataPipeline(object, metaclass=abc.ABCMeta):
    """Abstract operator to stream data to the network.

    Attributes:
      path:
      batch_size: number of sampes per batch.
      capacity: max number of samples in the data queue.
      pass
      min_after_dequeue: minimum number of image used for shuffling the queue.
      shuffle: random shuffle the samples in the queue.
      nthreads: number of threads use to fill up the queue.
      samples: a dict of tensors containing a batch of input samples with keys:
        'image_input' (the source image),
        'image_output' (the filtered image to match),
        'guide_image' (the image used to slice in the grid).
    """

    def __init__(self, path, batch_size=32,
                 capacity=16,
                 min_after_dequeue=4,
                 output_resolution=[1080, 1920],
                 shuffle=False,
                 fliplr=False,
                 flipud=False,
                 rotate=False,
                 random_crop=False,
                 params=None,
                 nthreads=1,
                 num_epochs=None,
                 validate_sizes=True):
        self.validate_sizes = validate_sizes
        self.path = path
        self.batch_size = batch_size
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue
        self.shuffle = shuffle
        self.nthreads = nthreads
        self.num_epochs = num_epochs
        self.params = params

        self.output_resolution = output_resolution

        # Data augmentation
        self.fliplr = fliplr
        self.flipud = flipud
        self.rotate = rotate
        self.random_crop = random_crop

        sample = self._produce_one_sample()
        self.samples = self._batch_samples(sample)

    @abc.abstractmethod
    def _produce_one_sample(self):
        pass

    def _batch_samples(self, sample):
        """Batch several samples together."""

        # Batch and shuffle
        if self.shuffle:
            samples = tf.train.shuffle_batch(
                sample,
                batch_size=self.batch_size,
                num_threads=self.nthreads,
                capacity=self.capacity,
                min_after_dequeue=self.min_after_dequeue)
        else:
            samples = tf.train.batch(
                sample,
                batch_size=self.batch_size,
                num_threads=self.nthreads,
                capacity=self.capacity)
        return samples

    def _augment_data(self, inout, nchan=6):
        """Flip, crop and rotate samples randomly."""

        with tf.name_scope('data_augmentation'):
            if self.fliplr:
                inout = tf.image.random_flip_left_right(inout, seed=1234)
            if self.flipud:
                inout = tf.image.random_flip_up_down(inout, seed=3456)
            if self.rotate:
                angle = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32, seed=4567)
                inout = tf.case([(tf.equal(angle, 1), lambda: tf.image.rot90(inout, k=1)),
                                 (tf.equal(angle, 2), lambda: tf.image.rot90(inout, k=2)),
                                 (tf.equal(angle, 3), lambda: tf.image.rot90(inout, k=3))],
                                lambda: inout)

            inout.set_shape([None, None, nchan])

            if self.validate_sizes:
                with tf.name_scope('crop'):
                    shape = tf.shape(inout)
                    new_height = tf.to_int32(self.output_resolution[0])
                    new_width = tf.to_int32(self.output_resolution[1])
                    height_ok = tf.assert_less_equal(new_height, shape[0])
                    width_ok = tf.assert_less_equal(new_width, shape[1])
                    with tf.control_dependencies([height_ok, width_ok]):
                        if self.random_crop:
                            inout = tf.random_crop(
                                inout, tf.stack([new_height, new_width, nchan]))
                        else:
                            height_offset = tf.to_int32((shape[0] - new_height) / 2)
                            width_offset = tf.to_int32((shape[1] - new_width) / 2)
                            inout = tf.image.crop_to_bounding_box(
                                inout, height_offset, width_offset,
                                new_height, new_width)

            inout.set_shape([None, None, nchan])
            inout = tf.image.resize_images(inout, self.output_resolution)
            fullres = inout

            with tf.name_scope('resize'):
                lowres_size = 256
                inout = tf.image.resize_images(
                    inout, [lowres_size, lowres_size],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            return fullres, inout


class LRNetDataPipeline(DataPipeline):

    def _produce_one_sample(self):
        dirname = os.path.dirname(self.path)

        self.nsamples = pd.read_csv(self.path).shape[0]

        if not check_dir(dirname):
            raise ValueError("Invalid data path.")

        data_root = tf.constant(dirname + '/')

        file = tf.train.string_input_producer([self.path])
        reader = tf.TextLineReader(skip_header_lines=1)

        key, value = reader.read(file)

        row = tf.decode_csv(value, [[''], ['']] + [[0.]] * self.params['lr_params'])

        input_file, output_file = row[:2]  # File names are in first two columns
        params = tf.stack(row[2:])  # And parameters are the rest

        dtype = tf.uint16
        wl = 65535.0

        input_file = tf.read_file(tf.string_join([data_root, input_file]))
        output_file = tf.read_file(tf.string_join([data_root, output_file]))

        im_input = tf.image.decode_png(input_file, dtype=dtype, channels=3)
        im_output = tf.image.decode_png(output_file, dtype=dtype, channels=3)

        # normalize input/output
        sample = {}
        with tf.name_scope('normalize_images'):
            im_input = tf.to_float(im_input) / wl
            im_output = tf.to_float(im_output) / wl

        inout = tf.concat([im_input, im_output], 2)
        fullres, inout = self._augment_data(inout, 6)

        sample['lowres_input'] = inout[:, :, :3]
        sample['lowres_output'] = inout[:, :, 3:]
        sample['image_input'] = fullres[:, :, :3]
        sample['image_output'] = fullres[:, :, 3:]
        sample['params'] = params
        return sample


# __all__ = ['RecordWriter', 'RecordReader',]


# Index and range for type serialization
TYPEMAP = {
    np.dtype(np.uint8): 0,
    np.dtype(np.int16): 1,
    np.dtype(np.float32): 2,
    np.dtype(np.int32): 3,
}

REVERSE_TYPEMAP = {
    0: tf.uint8,
    1: tf.int16,
    2: tf.float32,
    3: tf.int32,
}


class RecordWriter(object):
    """Writes input/output pairs to .tfrecords.

    Attributes:
      output_dir: directory where the .tfrecords are saved.
    """

    def __init__(self, output_dir, records_per_file=500, prefix=''):
        self.output_dir = output_dir
        self.records_per_file = records_per_file
        self.written = 0
        self.nfiles = 0
        self.prefix = prefix

        # internal state
        self._writer = None
        self._fname = None

    def _get_new_filename(self):
        self.nfiles += 1
        return os.path.join(self.output_dir, '{}{:06d}.tfrecords'.format(self.prefix, self.nfiles))

    def write(self, data):
        """Write the arrays in data to the currently opened tfrecords.

        Args:
          data: a dict of numpy arrays.

        Returns:
          The filename just written to.
        """

        if self.written % self.records_per_file == 0:
            self.close()
            self._fname = self._get_new_filename()
            self._writer = tf.python_io.TFRecordWriter(self._fname)

        feature = {}
        for k in list(data.keys()):
            feature[k] = self._bytes_feature(data[k].tobytes())
            feature[k + '_sz'] = self._int64_list_feature(data[k].shape)
            feature[k + '_dtype'] = self._int64_feature(TYPEMAP[data[k].dtype])

        example = tf.train.Example(
            features=tf.train.Features(feature=feature))

        self._writer.write(example.SerializeToString())
        self.written += 1
        return self._fname

    def close(self):
        if self._writer is not None:
            self._writer.close()

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class RecordReader(object):
    """Produces a queue of input/output data from .tfrecords.

    Attributes:
      shapes: dict of list of integers. The shape of each feature in the record.
    """

    FEATURES = ['image_input', 'image_output']
    NDIMS = [3, 3]

    def __init__(self, fnames, shuffle=True, num_epochs=None):
        """Init from a list of filenames to enqueue.

        Args:
          fnames: list of .tfrecords filenames to enqueue.
          shuffle: if true, shuffle the list at each epoch
        """
        self._fnames = fnames
        self._fname_queue = tf.train.string_input_producer(
            self._fnames,
            capacity=1000,
            shuffle=shuffle,
            num_epochs=num_epochs)
        self._reader = tf.TFRecordReader()

        # Read first record to initialize the shape parameters
        with tf.Graph().as_default():
            fname_queue = tf.train.string_input_producer(self._fnames)
            reader = tf.TFRecordReader()
            _, serialized = reader.read(fname_queue)
            shapes = self._parse_shape(serialized)
            dtypes = self._parse_dtype(serialized)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                self.shapes = sess.run(shapes)
                self.shapes = {k: self.shapes[k + '_sz'].tolist() for k in self.FEATURES}

                self.dtypes = sess.run(dtypes)
                self.dtypes = {k: REVERSE_TYPEMAP[self.dtypes[k + '_dtype'][0]] for k in self.FEATURES}

                coord.request_stop()
                coord.join(threads)

    def read(self):
        """Read and parse a single sample from the input queue.

        Returns:
          sample: a dict with keys in self.FEATURES. Each entry holds the
          corresponding Tensor.
        """
        _, serialized = self._reader.read(self._fname_queue)
        sample = self._parse_example(serialized)
        return sample

    def _get_dtype_features(self):
        ret = {}
        for i, f in enumerate(self.FEATURES):
            ret[f + '_dtype'] = tf.FixedLenFeature([1, ], tf.int64)
        return ret

    def _get_sz_features(self):
        ret = {}
        for i, f in enumerate(self.FEATURES):
            ret[f + '_sz'] = tf.FixedLenFeature([self.NDIMS[i], ], tf.int64)
        return ret

    def _get_data_features(self):
        ret = {}
        for f in self.FEATURES:
            ret[f] = tf.FixedLenFeature([], tf.string)
        return ret

    def _parse_shape(self, serialized):
        sample = tf.parse_single_example(serialized,
                                         features=self._get_sz_features())
        return sample

    def _parse_dtype(self, serialized):
        sample = tf.parse_single_example(serialized,
                                         features=self._get_dtype_features())
        return sample

    def _parse_example(self, serialized):
        """Unpack a serialized example to Tensor."""
        feats = self._get_data_features()
        sz_feats = self._get_sz_features()
        for s in sz_feats:
            feats[s] = sz_feats[s]
        sample = tf.parse_single_example(serialized, features=feats)

        data = {}
        for i, f in enumerate(self.FEATURES):
            s = tf.to_int32(sample[f + '_sz'])

            data[f] = tf.decode_raw(sample[f], self.dtypes[f], name='decode_{}'.format(f))
            data[f] = tf.reshape(data[f], s)

        return data
