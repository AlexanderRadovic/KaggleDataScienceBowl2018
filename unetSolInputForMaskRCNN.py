#macro to save unet output for set of images, potentially useful for a maskrcnn as an additional input
#-Kjetil Amdal-Saevik and his Kernel "Keras U-Net starter - LB 0.277"
#-the github repo unet-tensorflow-keras
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray
from sklearn.cluster import KMeans, DBSCAN

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from modelZoo import UNet

# Set some parameters
IMG_WIDTH = 384
IMG_HEIGHT = 384
IMG_CHANNELS = 3
TEST_PATH = 'stage1_train/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize test images
#X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()

# Predict on train, val and test
model = load_model('model-dsbowl2018-hqsizeshift.h5')
new_test_ids = []
rles = []

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        #print (path)
        rawImg = imread(path + '/images/' + id_ + '.png')
        if rawImg.ndim==2:
                placeHolder=np.zeros((rawImg.shape[0],rawImg.shape[1],3))
                placeHolder[:,:,0]=rawImg
                placeHolder[:,:,1]=rawImg
                placeHolder[:,:,2]=rawImg
                rawImg=placeHolder
                
        #print (rawimg.shape)
        img = rawImg[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        placeHolderImg=np.zeros((1,IMG_HEIGHT,IMG_WIDTH,3))
        placeHolderImg[0,:,:,:]=img
        img=placeHolderImg

        div = img.max(axis=tuple(np.arange(1,len(img.shape))), keepdims=True) 
        div[div < 0.01*img.mean()] = 1. # protect against too small pixel intensities
        img = img.astype(np.float32)/div
        
        preds_test = model.predict([img], verbose=0)
        
        # Create upsampled test mask
        preds_test_upsampled = resize(np.squeeze(preds_test),
                                           (sizes_test[n][0], sizes_test[n][1]),
                                           mode='constant', preserve_range=True)

        plt.imsave(path +'/images/' +str(id_)+'_unetsol.png',preds_test_upsampled)
        plt.clf()


        
