### References ### 
# [1] https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/iterator.py

import os
import threading
import random
import numpy as np
import cv2
from utils import preprocess_LR, preprocess_HR, get_shape

# A class that generates batches from low-resolution and high-resolution image pairs.
class BatchGenerator(object):
    def __init__(self, path_lr, path_hr, input_size, target_size, data_format, batch_size, shuffle, seed=None):
        self.path_lr = path_lr
        self.path_hr = path_hr
        self.data_format = data_format
        self.image_shape_lr = get_shape(input_size, data_format)
        self.image_shape_hr = get_shape(target_size, data_format)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_array = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index()
        self.images = os.listdir(path_hr)
        self.n = len(self.images)

    # Sets the index array, taken from [1].
    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
    
    # Index iteration flow, taken from [1].
    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.batch_index = 0
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            # Avoiding modulo by zero error
            if self.n == 0:
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n

            if self.n > current_index + self.batch_size:
                current_batch_size = self.batch_size
                self.batch_index += 1
            else:
                current_batch_size = self.n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (self.index_array[current_index: current_index + current_batch_size], current_batch_size)

    def next(self):
        with self.lock:
            index_array, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock so it can be done in parallel
        batch_HR = np.zeros((current_batch_size,) + self.image_shape_hr, dtype=np.float32)
        batch_LR = np.zeros((current_batch_size,) + self.image_shape_lr, dtype=np.float32)

        for i, j in enumerate(index_array):
            fname = self.images[j]
            batch_HR[i] = cv2.imread(os.path.join(self.path_hr, fname))
            batch_LR[i] = cv2.imread(os.path.join(self.path_lr, fname))

        # Preprocess images to rescale them to [0,1] for low-res and to [-1,1] for high-res
        batch_LR = preprocess_LR(batch_LR)
        batch_HR = preprocess_HR(batch_HR)

        # Transpose images in case of channel first data format
        if self.data_format == 'channels_first':
            batch_HR = np.transpose(batch_HR, (0, 3, 1, 2))
            batch_LR = np.transpose(batch_LR, (0, 3, 1, 2))

        return batch_LR, batch_HR
        