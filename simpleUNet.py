#macro to produce several plots to help understand this dataset.
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

from model import UNet

# Load np array for input data and masks
X_train = np.load('inputImages.npy')
Y_train = np.load('inputMask.npy')

model = UNet().create_model(img_shape=X_train[0].shape, num_class=1)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train,
                        validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])

