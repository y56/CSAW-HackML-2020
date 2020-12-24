#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:47:13 2020

take a look into photos

@author: y56
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

data_filename = 'data/anonymous_1_poisoned_data.h5'


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    return x_data, y_data


def data_preprocess(x_data):
    return x_data / 255


x_test, y_test = data_loader(data_filename)
x_test = data_preprocess(x_test)

print(x_test.shape)

color = np.array([128 / 255, 0, 128 / 255])

for i in [0, 1, 2]:
    x = x_test[i]
    plt.figure()
    plt.imshow(x)
    plt.show()
