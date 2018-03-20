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

def plotPix(inputArray, name):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_title('')
        ax.set_ylabel('nPixels')
        ax.set_xlabel('Pixel Saturation')
        plt.bar(range(len(inputArray)),inputArray, 1, color="blue",alpha=0.9)
        ax.legend(loc='right',frameon=False)
        plt.savefig('plots/pixelComparisons'+name+'.png',dpi = 100)

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
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

X_intensity_grey = np.zeros(256);
X_intensity_fakegrey = np.zeros(256);
X_intensity_r1 = np.zeros(256);
X_intensity_r2 = np.zeros(256);
X_intensity_r3 = np.zeros(256);
i = 0
nGrey=0
nColor=0
xLengthGrey= []
yLengthGrey = []
xLengthColor= []
yLengthColor= []
areaSize = []


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

        img_fakeGrey=rgb2gray(img)
        areaSize.append(img.shape[1]*img.shape[0])

        flatImg_fakegrey=img_fakeGrey.flatten()
        flatImg_r1=img[:,:,0].flatten()
        flatImg_r2=img[:,:,1].flatten()
        flatImg_r3=img[:,:,2].flatten()

        # print flatImg_r1[0]
        # print flatImg_r2[0]
        # print flatImg_r3[0]
        # print flatImg_r1[10]
        # print flatImg_r2[10]
        # print flatImg_r3[10]

        # print flatImg_r1.shape
        # print flatImg_r2.shape
        # print flatImg_r3.shape


        # print np.count_nonzero(flatImg_r1)
        # print np.count_nonzero(flatImg_r2)
        # print np.count_nonzero(flatImg_r3)
        
        if np.mean(flatImg_r1)==np.mean(flatImg_r2) and np.mean(flatImg_r1)==np.mean(flatImg_r3): 
                nGrey=nGrey+1
                xLengthGrey.append(img.shape[0])
                yLengthGrey.append(img.shape[1])
                for pix in flatImg_r1:
                        X_intensity_grey[pix]=X_intensity_grey[pix]+1
           
                        
        else:
                nColor=nColor+1
                xLengthColor.append(img.shape[0])
                yLengthColor.append(img.shape[1])
                for pix in flatImg_fakegrey:
                        X_intensity_fakegrey[int(255*pix)]=X_intensity_fakegrey[int(255*pix)]+1
                for pix in flatImg_r1:
                        X_intensity_r1[pix]=X_intensity_r1[pix]+1

                for pix in flatImg_r2:
                        X_intensity_r2[pix]=X_intensity_r2[pix]+1

                for pix in flatImg_r3:
                        X_intensity_r3[pix]=X_intensity_r3[pix]+1
                                
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
        i=i+1


print X_intensity_grey[0]
# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Check if training data looks all right
# ix = random.randint(0, len(train_ids))
# imshow(X_train[ix])
# plt.show()
# plt.savefig('exampleEvent.png',dpi = 100)
# imshow(np.squeeze(Y_train[ix]))
# plt.show()
# plt.savefig('exampleMask.png',dpi = 100)

X_intensity_allcolor = X_intensity_r1+X_intensity_r2+X_intensity_r3; 

print('nGrey Images: '+str(nGrey)+' nColor '+str(nColor))

plotPix(X_intensity_grey,'Grey')
plotPix(X_intensity_fakegrey,'FakeGrey')
plotPix(X_intensity_allcolor,'AllColor')
plotPix(X_intensity_r1,'AllColor[0]')
plotPix(X_intensity_r2,'AllColor[1]')
plotPix(X_intensity_r3,'AllColor[2]')

fig, ax = plt.subplots(figsize=(6,6))
ax.set_title('')
ax.set_ylabel('nImages')
ax.set_xlabel('Image Size')
plt.hist(np.asarray(xLengthGrey),color='b', alpha=0.9, histtype='step',lw=2,label='x Length')
plt.hist(np.asarray(yLengthGrey),color='r', alpha=0.9, histtype='step',lw=2,label='y Length')
ax.legend(loc='right',frameon=False)
plt.savefig('plots/sizeComparisonsGrey.png',dpi = 100)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_title('')
ax.set_ylabel('nImages')
ax.set_xlabel('Image Size')
plt.hist(np.asarray(xLengthColor),color='b', alpha=0.9, histtype='step',lw=2,label='x Length')
plt.hist(np.asarray(yLengthColor),color='r', alpha=0.9, histtype='step',lw=2,label='y Length')
ax.legend(loc='right',frameon=False)
plt.savefig('plots/sizeComparisonsColor.png',dpi = 100)









