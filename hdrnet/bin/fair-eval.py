#!/usr/bin/env python3
# coding: utf-8
import cv2
import numpy as np
import glob
import os
import tqdm
import argparse


def list_dir_suffix(path, suffix):
    files = glob.glob(os.path.join(path, '*' + suffix))

    return list(map(lambda path: os.path.basename(path).replace(suffix, ''), files))


def prefixes_to_abspath(prefixes, base_path, suffix):
    return list(map(lambda p: os.path.join(base_path, p + suffix), prefixes))


def get_valid_prefixes(gt_path, out_path, gt_suffix, out_suffix):
    target_prefixes = list_dir_suffix(gt_path, gt_suffix)
    out_prefixes = list_dir_suffix(out_path, out_suffix)

    return list(set(target_prefixes).intersection(out_prefixes))


def load_image(path):
    img = np.flip(cv2.imread(path, -1), 2)

    img = img / np.iinfo(img.dtype).max
    return img


def run_eval(gt_path, out_path, gt_suffix, out_suffix):
    prefixes = get_valid_prefixes(gt_path, out_path, gt_suffix, out_suffix)

    eval_seq = list(zip(prefixes, prefixes_to_abspath(prefixes, out_path, out_suffix),
                   prefixes_to_abspath(prefixes, gt_path, gt_suffix)))

    losses = []
    for p, out_path, gt_path in tqdm.tqdm(eval_seq):
        out = load_image(out_path)
        gt = load_image(gt_path)

        loss = np.mean(np.abs(gt - out))
        losses.append(loss)
    #    print('{}: {}'.format(p, loss))

    print("Mean: {}".format(np.mean(losses)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("out_path")
    parser.add_argument("gt_path")

    parser.add_argument("--gt-suffix", default=".png")
    parser.add_argument("--out-suffix", default=".png")

    args = parser.parse_args()

    run_eval(args.gt_path, args.out_path, args.gt_suffix, args.out_suffix)
