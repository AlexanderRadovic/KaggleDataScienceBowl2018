#Convert the input datasets to a numpy array to make access easier later
#Additionally it can iterate through the original dataset carrying out random operations including:
#-flipping
#-rotation
#-color transformations

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

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'
augment = False
normalize = False

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
i=0

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        
        #acquire original image and resize so all images match
        path = TRAIN_PATH + id_
        imgOrig = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]                                
        img = resize(imgOrig, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

	#acquire original mask and resize so all images match
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)

        img=img.astype(np.uint8)
        mask=mask.astype(np.bool)
	X_train.append(img)
	Y_train.append(mask)

	if augment == True:
		#create alternative mask/image pairs out of the larger training examples
		if imgOrig.shape[0]>IMG_HEIGHT or imgOrig.shape[0]>IMG_WIDTH:
			for i in range(ceil(IMG_HEIGHT/imgOrig.shape[0])):
				for j in range(ceil(IMG_HEIGHT/imgOrig.shape[0])):
					subImg=imgOrig[i*IMGHEIGHT:(i+1)*IMGHEIGHT,j*IMGWIDTH:(j+1)*IMGWIDTH,:]

					subMask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
					for mask_file in next(os.walk(path + '/masks/'))[2]:
						mask_ = imread(path + '/masks/' + mask_file)
						mask_ = mask_[i*IMGHEIGHT:(i+1)*IMGHEIGHT,j*IMGWIDTH:(j+1)*IMGWIDTH,:]
						mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
						mask = np.maximum(mask, mask_)

					subImg=subImg.astype(np.uint8)
					subMask=subMask.astype(np.bool)

					X_train.append(subImg)
					Y_train.append(subMask)

# 		for j in range(0, 10):
# 			cloneImg=img
# 			cloneMask=mask
			
# 		#randomly flip x axis
# 			if random() > 0.5:
# 				cloneImg=np.flip(cloneImg, 0)
# 				cloneMask=np.flip(cloneMas, 0)
			
# 		#randomly flip y axis 
# 			if random() > 0.5:
# 				cloneImg=np.flip(cloneImg, 1)
# 				cloneMask=np.flip(cloneMask, 1)

# 		#random rotation
# 			if random() > 0.5:
# 				rotAngle=360*random()
# 				cloneImg=rotate(cloneImg, rotAngle)
# 				cloneMask=rotate(cloneMask, rotAngle)

# 		#randomly shift color palette
# 			if random() > 0.5:
# 				cloneImg=rgb2hsv(cloneImg)
			
# 		#randomly shift color palette
# 			if random() > 0.5:
# 				cloneImg=hsv2rgb(cloneImg)

# 			X_train.append(cloneImg)
# 			Y_train.append(cloneMask)

	
	i=i+1

if normalize == True: 
	for i in range(len(X_train)):
		div = X_train[i].max(axis=tuple(np.arange(1,len(X_train[i].shape))), keepdims=True) 
		div[div < 0.01*X_train[i].mean()] = 1. # protect against too small pixel intensities
		X_train[i] = X_train[i].astype(np.float32)/div

np.save('inputImages.npy',np.stack(X_train))
np.save('inputMask.npy',np.stack(Y_train))












