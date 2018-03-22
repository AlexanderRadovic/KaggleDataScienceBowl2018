#macro to produce plots to help understand how the keras inbuilt data augementation will effect the nuclei dataset
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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K


import tensorflow as tf

from model import UNet

# Load np array for input data and masks
X_train = np.load('inputImages.npy')

# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90.,
                     shear_range=0.2,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=0.1,
                     fill_mode='wrap')

image_datagen = ImageDataGenerator(**data_gen_args)

i = 0
for batch in image_datagen.flow(X_train, batch_size=1, save_to_dir='plots/testAug/',save_prefix='example_',save_format='png'):
    i += 1
    if i > 20:
        break

    
