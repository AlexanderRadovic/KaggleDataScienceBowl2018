#Macro to visualy inspect some augmented data from creatAugmentedDataset
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

seed = 42
random.seed = seed
np.random.seed = seed

# Get and resize train images and masks
X_train = np.load('inputImagesAug.npy')
Y_train = np.load('inputMaskAug.npy')

for ix in range(0, len(X_train)):
        plt.subplot(121)
        imshow(X_train[ix])
        plt.subplot(122)
        imshow(np.squeeze(Y_train[ix]))
        plt.savefig('plots/testAug/example_'+str(ix)+'.png',dpi = 100)
