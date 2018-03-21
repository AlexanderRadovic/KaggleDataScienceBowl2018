#macro to produce an augmented dataset. 
#Iterate through the original dataset carrying out random operations including:
#-flipping
#-rotation
#-color transformations
#Then store the complete larger dataset as a numpy array

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
from skimage.transform import resize, rotate
from skimage.morphology import label
from skimage.color import rgb2gray

def plotPix(inputArray, name):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_title('')
        ax.set_ylabel('nPixels')
        ax.set_xlabel('Pixel Saturation')
        plt.bar(range(len(inputArray)),inputArray, 1, color="blue",alpha=0.9)
        ax.legend(loc='right',frameon=False)
        plt.savefig('plots/pixelComparisons'+name+'.png',dpi = 100)

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
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
X_train = []
Y_train = []
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

	#acquire original image and resize so all images match
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]                                
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

	#acquire original mask and resize so all images match
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
	
	#X_train.append(img)
	#Y_train.append(mask)
	
	for j in range(0, 10):
		cloneImg=img
		cloneMask=mask

		#randomly flip x axis
		if random() > 0.5:
			cloneImg=np.flip(cloneImg, 0)
			cloneMask=np.flip(cloneMas, 0)

		#randomly flip y axis 
		if random() > 0.5:
			cloneImg=np.flip(cloneImg, 1)
			cloneMask=np.flip(cloneMask, 1)

		#random rotation
		if random() > 0.5:
			rotAngle=360*random()
			cloneImg=rotate(cloneImg, rotAngle)
			cloneMask=rotate(cloneMask, rotAngle)

		X_train.append(cloneImg)
		Y_train.append(cloneMask)
	
	i=i+1

 np.save('inputImages.npy',)
 np.save('inputMask.npy',)












