from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import scipy
import scipy.signal
from scipy.misc import imresize
from scipy.stats import threshold

import matplotlib.pyplot as plt
plt.ion()
plt.show()

sys.path.append('/reg/neh/home/davidsch/github/davidslac/psana-mlearn')
import psmlearn

### local imports
import preprocess

import cv2

## to slow
'''
def signal_processing_solution(img, kernel=.05):
    K1_shape = np.round(kernel*np.array(img.shape, dtype=np.float)).astype(np.int)
    K2_shape = 3*K1_shape

    K1 = np.ones(K1_shape, np.float32)
    K2 = np.ones(K2_shape, np.float32)
    K2[K1_shape[0]:2*K1_shape[0], K1_shape[1]:2*K1_shape[1]]=0.0
    
    img = 10*np.log(1+np.maximum(0.0, img.astype(np.float32)))
    imgK1 = scipy.signal.convolve2d(img, K1, mode='same')
    imgK2 = 1e-6 + scipy.signal.convolve2d(img, K2, mode='same')
    metric = imgK1/imgK2
    metric_max = np.unravel_index(np.argmax(metric), img.shape)
    return metric_max
'''

'''
def signal_processing_solution(img, kernel=11):
    reduced = imresize(img,(250,250))
    filtered = scipy.signal.medfilt2d(reduced,kernel)
    filtered_max = np.unravel_index(np.argmax(filtered), filtered.shape)
    return filtered_max[0]*img.shape[0]/250.0, filtered_max[1]*img.shape[1]/250.0
'''

def signal_processing_solution(img, kernel=11):
    isYag = img.shape[0]>1000
    if isYag:
        img = cv2.medianBlur(img, 5)
        img = cv2.GaussianBlur(img, (55,55),0)
        img = threshold(img, 1.5)
    else:
        if img.dtype == np.float32:
            img = (np.minimum(np.maximum(0,img),255.0)).astype(np.uint8)
        img = cv2.medianBlur(img, 7)
        img = cv2.GaussianBlur(img, (15,15),0)
    filtered_max = np.unravel_index(np.argmax(img), img.shape)

    return filtered_max[0], filtered_max[1]
