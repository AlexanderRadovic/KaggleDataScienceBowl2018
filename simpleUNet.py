#macro to produce several plots to help understand this dataset.
#thanks to:
#-Kjetil Amdal-Saevik and his Kernel "Keras U-Net starter - LB 0.277"
import os
import sys
sys.path.append('/home/alexander/competitionCode/unet-tensorflow-keras')
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
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '/home/alexander/kaggleData/2018dsbowl/stage1_train/'
TEST_PATH = '/home/alexander/kaggleData/2018dsbowl/stage1_test/'

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

model = UNet().create_model(img_shape=X_train[0].shape, num_class=1)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train,
                        validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])

