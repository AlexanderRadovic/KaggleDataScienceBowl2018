#macro to produce plots showing input data, the U-Net output, predicted masks, and true masks
#thanks to:
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

# Load np array for input data and masks
X_train = np.load('inputImages.npy')
Y_train = np.load('inputMask.npy')

# Predict on train, val and test
model = load_model('model-dsbowl2018-1-c3.h5')
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

# Threshold predictions
preds_val_t = (preds_val > 0.5).astype(np.uint8)

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

        plt.savefig('plots/validationPerf/example_'+str(ix)+'.png',dpi = 100)
