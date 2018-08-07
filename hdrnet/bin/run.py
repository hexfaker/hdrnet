#!/usr/bin/env python3
# encoding: utf-8
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

"""Evaluates a trained network."""

import argparse
import cv2
import logging
import numpy as np
import os
import re
import setproctitle
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import pandas as pd
import tqdm

import hdrnet.models as models
import hdrnet.utils as utils

logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def get_input_list(path):
    regex = re.compile(".*.(png|jpeg|jpg|tif|tiff)")
    if os.path.isdir(path):
        inputs = os.listdir(path)
        inputs = [os.path.join(path, f) for f in inputs if regex.match(f)]
        log.info("Directory input {}, with {} images".format(path, len(inputs)))

    elif os.path.splitext(path)[-1] == ".txt":
        dirname = os.path.dirname(path)
        with open(path, 'r') as fid:
            inputs = [l.strip() for l in fid.readlines()]
        inputs = [os.path.join(dirname, 'input', im) for im in inputs]
        log.info("Filelist input {}, with {} images".format(path, len(inputs)))
    elif regex.match(path):
        inputs = [path]
        log.info("Single input {}".format(path))
    return inputs


def get_input_list_csv(path_to_csv):
    df = pd.read_csv(path_to_csv)
    root = os.path.dirname(path_to_csv)

    return [(os.path.join(root, row[0]), os.path.join(root, row[1]), list(row[2:])) for row in
            df.itertuples(False, None)]


def dec(b):
    return b.decode('utf-8')


def main(args):
    setproctitle.setproctitle('hdrnet_run')

    inputs = get_input_list_csv(args.input)

    # -------- Load params ----------------------------------------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        checkpoint_path = args.checkpoint_path
        if os.path.isdir(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

        if checkpoint_path is None:
            log.error('Could not find a checkpoint in {}'.format(args.checkpoint_path))
            return

        metapath = ".".join([checkpoint_path, "meta"])
        log.info('Loading graph from {}'.format(metapath))
        tf.train.import_meta_graph(metapath)

        model_params = utils.get_model_params(sess)

    model_params['model_name'] = dec(model_params['model_name'])

    # -------- Setup graph ----------------------------------------------------
    print(model_params['model_name'])
    if not hasattr(models, model_params['model_name']):
        log.error("Model {} does not exist".format(model_params.model_name))
        return
    mdl = getattr(models, model_params['model_name'])

    tf.reset_default_graph()
    net_shape = model_params['net_input_size']
    t_fullres_input = tf.placeholder(tf.float32, (1, None, None, 3))
    t_lowres_input = tf.placeholder(tf.float32, (1, net_shape, net_shape, 3))
    t_params_input = tf.placeholder(tf.float32, (1, model_params['lr_params']))

    with tf.variable_scope('inference'):
        prediction = mdl.inference(
            t_lowres_input, t_fullres_input, t_params_input, model_params, is_training=False)
    output = tf.squeeze(tf.clip_by_value(prediction, 0, 1))
    saver = tf.train.Saver()

    if args.debug:
        coeffs = tf.get_collection('bilateral_coefficients')[0]
        if len(coeffs.get_shape().as_list()) == 6:
            bs, gh, gw, gd, no, ni = coeffs.get_shape().as_list()
            coeffs = tf.transpose(coeffs, [0, 3, 1, 4, 5, 2])
            coeffs = tf.reshape(coeffs, [bs, gh * gd, gw * ni * no, 1])
            coeffs = tf.squeeze(coeffs)
            m = tf.reduce_max(tf.abs(coeffs))
            coeffs = tf.clip_by_value((coeffs + m) / (2 * m), 0, 1)

        ms = tf.get_collection('multiscale')
        if len(ms) > 0:
            for i, m in enumerate(ms):
                maxi = tf.reduce_max(tf.abs(m))
                m = tf.clip_by_value((m + maxi) / (2 * maxi), 0, 1)
                sz = tf.shape(m)
                m = tf.transpose(m, [0, 1, 3, 2])
                m = tf.reshape(m, [sz[0], sz[1], sz[2] * sz[3]])
                ms[i] = tf.squeeze(m)

        fr = tf.get_collection('fullres_features')
        if len(fr) > 0:
            for i, m in enumerate(fr):
                maxi = tf.reduce_max(tf.abs(m))
                m = tf.clip_by_value((m + maxi) / (2 * maxi), 0, 1)
                sz = tf.shape(m)
                m = tf.transpose(m, [0, 1, 3, 2])
                m = tf.reshape(m, [sz[0], sz[1], sz[2] * sz[3]])
                fr[i] = tf.squeeze(m)

        guide = tf.get_collection('guide')
        if len(guide) > 0:
            for i, g in enumerate(guide):
                maxi = tf.reduce_max(tf.abs(g))
                g = tf.clip_by_value((g + maxi) / (2 * maxi), 0, 1)
                guide[i] = tf.squeeze(g)

    with tf.Session(config=config) as sess:
        log.info('Restoring weights from {}'.format(checkpoint_path))
        saver.restore(sess, checkpoint_path)

        loss = []
        for idx, (input_path, gt_path, params) in enumerate(tqdm.tqdm(inputs)):
            if args.limit is not None and idx >= args.limit:
                log.info("Stopping at limit {}".format(args.limit))
                break

            input_image = load_image(input_path, args.eval_resolution)
            gt_image = load_image(gt_path, args.eval_resolution)

            # Make or Load lowres image
            if args.lowres_input is None:
                lowres_input = skimage.transform.resize(
                    input_image, [net_shape, net_shape], order=0, mode='constant', anti_aliasing=False)
            else:
                raise NotImplemented

            basedir = args.output
            prefix = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(basedir, prefix + "_out.jpg")
            gt_copy_path = os.path.join(basedir, prefix + "_gt.jpg")
            input_copy_path = os.path.join(basedir, prefix + "_1n.jpg")  # Not typo. ordering

            input_image = input_image[np.newaxis, :, :, :]
            lowres_input = lowres_input[np.newaxis, :, :, :]
            params = np.array(params)[np.newaxis, :]

            feed_dict = {
                t_fullres_input: input_image,
                t_lowres_input: lowres_input,
                t_params_input: params
            }

            out_image = sess.run(output, feed_dict=feed_dict)

            if not os.path.exists(basedir):
                os.makedirs(basedir)

            loss.append(np.mean(np.abs(gt_image - out_image)))

            skimage.io.imsave(output_path, save_img(out_image))
            skimage.io.imsave(input_copy_path, save_img(input_image[0]))
            skimage.io.imsave(gt_copy_path, save_img(gt_image))

            if args.debug:
                output_path = os.path.join(args.output, prefix + "_input.png")
                skimage.io.imsave(output_path, np.squeeze(input_image))

                coeffs_ = sess.run(coeffs, feed_dict=feed_dict)
                output_path = os.path.join(args.output, prefix + "_coeffs.png")
                skimage.io.imsave(output_path, coeffs_)
                if len(ms) > 0:
                    ms_ = sess.run(ms, feed_dict=feed_dict)
                    for i, m in enumerate(ms_):
                        output_path = os.path.join(args.output, prefix + "_ms_{}.png".format(i))
                        skimage.io.imsave(output_path, m)

                if len(fr) > 0:
                    fr_ = sess.run(fr, feed_dict=feed_dict)
                    for i, m in enumerate(fr_):
                        output_path = os.path.join(args.output, prefix + "_fr_{}.png".format(i))
                        skimage.io.imsave(output_path, m)

                if len(guide) > 0:
                    guide_ = sess.run(guide, feed_dict=feed_dict)
                    for i, g in enumerate(guide_):
                        output_path = os.path.join(args.output, prefix + "_guide_{}.png".format(i))
                        skimage.io.imsave(output_path, g)

    print("Loss: " + str(np.mean(loss)))


def save_img(out_image):
    return (np.array(out_image) * 255).astype(np.uint8)


def load_image(input_path, size):
    im_input = cv2.imread(input_path, -1)  # -1 means read as is, no conversions.
    if im_input.shape[2] == 4:
        im_input = im_input[:, :, :3]
    im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB
    im_input = cv2.resize(im_input, tuple(size), cv2.INTER_LINEAR)
    im_input = skimage.img_as_float(im_input)
    return im_input


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', default=None, help='path to the saved model variables')
    parser.add_argument('input', default=None, help='path to the validation data')
    parser.add_argument('output', default=None, help='path to save the processed images')

    # Optional
    parser.add_argument('--lowres_input', default=None, help='path to the lowres, TF inputs')
    parser.add_argument('--no_save_gt', action='store_true')
    parser.add_argument('--no_save_input', action='store_true')
    parser.add_argument('--eval_resolution', default=[600, 400], nargs=2)
    parser.add_argument('--hdrp', dest="hdrp", action="store_true", help='special flag for HDR+ to set proper range')
    parser.add_argument('--nohdrp', dest="hdrp", action="store_false")
    parser.add_argument('--debug', dest="debug", action="store_true",
                        help='If true, dumps debug data on guide and coefficients.')
    parser.add_argument('--limit', type=int, help="limit the number of images processed.")
    parser.set_defaults(hdrp=False, debug=False)
    # pylint: enable=line-too-long
    # -----------------------------------------------------------------------------

    args = parser.parse_args()
    main(args)
