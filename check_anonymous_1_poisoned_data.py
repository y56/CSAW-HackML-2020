#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 22:18:10 2020

use the color purple to identify photos with backdoor to analysis performance

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
    x_data = x_data.transpose((0,2,3,1))
    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

x_test, y_test = data_loader(data_filename)
x_test = data_preprocess(x_test)

print(x_test.shape)

color = np.array([128/255,0,128/255])

li=[]
for i in range(x_test.shape[0]):
    x = x_test[i]
    ct=0
    for j in range(x_test.shape[1]):
        for k in range(x_test.shape[2]):
            if all(x[j,k,:]==color):
                ct+=1
    li.append(ct)
sli = sorted(li)
            
plt.figure()
plt.plot(li)
plt.show()

plt.figure()
plt.plot(sli)
plt.show()

"""
we can know that >= 12 pt of purple is backdoored
"""

res = [x >= 12 for x in li]

import pickle

f = open('trigger_anonymous_1_poisoned_data.pkl', 'wb')
pickle.dump(res, f)
f.close()
