#macro to produce several plots to help understand this dataset.
#thanks to:
#-Kjetil Amdal-Saevik and his Kernel "Keras U-Net starter - LB 0.277"
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

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from model import UNet

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

i=0

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

        #if i > 16:
        #        break
        
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:3]

        #img=img/255.
        # img_fakeGrey=rgb2gray(img)

        # if np.mean(img[:,:,0])==np.mean(img[:,:,1]) and np.mean(img[:,:,0])==np.mean(img[:,:,2]):
        #        img_fakeGrey=img[:,:,0]/255.
                                
        #img = resize(img_fakeGrey.reshape((img_fakeGrey.shape[0],img_fakeGrey.shape[1],1)), (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
        i=i+1

print('Done!')

# Predict on train, val and test
model = load_model('model-dsbowl2018-1-c3.h5')
#preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
#preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
#preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
#preds_test_upsampled = []
#for i in range(len(preds_test)):
 #           preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),

X_val=X_train[int(X_train.shape[0]*0.9):]
Y_val=Y_train[int(Y_train.shape[0]*0.9):]

for ix in range(0, len(X_val)):
        plt.subplot(141)
        imshow(X_val[ix])
        plt.subplot(142)
        imshow(np.squeeze(preds_val[ix]))
        plt.subplot(143)
        imshow(np.squeeze(preds_val_t[ix]))
        plt.subplot(144)
        imshow(np.squeeze(Y_val[ix]))

        #plt.show()
        plt.savefig('plots/validationPerf/example_'+str(ix)+'.png',dpi = 100)
