# -*- coding: utf-8 -*-

from __future__ import print_function

import itertools
import os

import imageio
import numpy as np
from six.moves import cPickle

from DB import MyDatabase
from evaluate import myevaluate
import matplotlib.image as mpimg

# configs for histogram
n_bin = 12  # histogram bins
n_slice = 3  # slice image
h_type = 'region'  # global or region
d_type = 'd1'  # distance type

depth = 3  # retrieved depth, set to None will count the ap for whole database

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class Color(object):

    def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            # img = scipy.misc.imread(input, mode='RGB')
            img = imageio.imread(input)
        height, width, channel = img.shape
        bins = np.linspace(0, 256, n_bin + 1, endpoint=True)  # slice bins equally for each channel

        if type == 'global':
            hist = self._count_hist(img, n_bin, bins, channel)

        elif type == 'region':
            hist = np.zeros((n_slice, n_slice, n_bin ** channel))
            h_silce = np.around(np.linspace(0, height, n_slice + 1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, n_slice + 1, endpoint=True)).astype(int)

            for hs in range(len(h_silce) - 1):
                for ws in range(len(w_slice) - 1):
                    img_r = img[h_silce[hs]:h_silce[hs + 1], w_slice[ws]:w_slice[ws + 1]]  # slice img to regions
                    hist[hs][ws] = self._count_hist(img_r, n_bin, bins, channel)

        if normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _count_hist(self, input, n_bin, bins, channel):
        img = input.copy()
        bins_idx = {key: idx for idx, key in
                    enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
        hist = np.zeros(n_bin ** channel)

        # cluster every pixels
        for idx in range(len(bins) - 1):
            img[(input >= bins[idx]) & (input < bins[idx + 1])] = idx
        # add pixels into bins
        height, width, _ = img.shape
        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h, w])]
                hist[b_idx] += 1

        return hist

    def make_samples(self, db, verbose=True):
        mystr = db.get_db_type()

        if h_type == 'global':
            sample_cache = "histogram_cache-{}-n_bin{}_{}".format(h_type, n_bin, mystr)
        elif h_type == 'region':
            sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}_{}".format(h_type, n_bin, n_slice, mystr)

        try:
            samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
            samples = []
            data = db.get_data()
            for d in data.itertuples():
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_hist = self.histogram(d_img, type=h_type, n_bin=n_bin, n_slice=n_slice)
                samples.append({
                    'img': d_img,
                    'cls': d_cls,
                    'hist': d_hist
                })
            cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))

        return samples


if __name__ == "__main__":

    dbTrain = MyDatabase("database/train", "database/train/data_train.csv")
    print("Train db length: ", len(dbTrain))
    color = Color()

    dbTest = MyDatabase("database/test", "database/test/data_test.csv")
    print("Test db length: ", len(dbTest))

    # evaluate database
    APs, res = myevaluate(dbTrain, dbTest, color.make_samples, depth=depth, d_type="d1")

    # add pictures in prediction folder under the predicted class folder
    path = "/Users/johanncarfantan/Documents/ENSSAT/IMR3/AnalyseDimages/Partie 1/predictions/"
    for i in range(len(dbTest)):
        saveName = path + res[i] + "/" + dbTest.data.img[i].split('/')[-1]
        bid = imageio.imread(dbTest.data.img[i])
        if not os.path.exists(path + res[i]):
            os.makedirs(path + res[i])
        mpimg.imsave(saveName, bid / 255.)
