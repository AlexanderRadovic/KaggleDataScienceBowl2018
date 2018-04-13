#macro to run my simple u-net on the test dataset and convert the output for submission
#thanks to:
#-Kjetil Amdal-Saevik and his Kernel "Keras U-Net starter - LB 0.277"
#-the github repo unet-tensorflow-keras
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import csv

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

from modelZoo import UNet



def rleToMask(rleString,height, width):
        rows,cols = height,width
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        #print (rlePairs)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
                #print ('index'+str(index))
                #print ('length'+str(length))
                index -= 1
                img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img


# Set some parameters
IMG_CHANNELS = 3
TEST_PATH = 'stage1_test/'

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

df=pd.read_csv('stage1_solution.csv',delimiter=',')

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        rawImg = imread(path + '/images/' + id_ + '.png')


        #print (id_)
        #print (df.iloc[n])

        imgDf=df.loc[lambda df: df.ImageId==id_,:]
        #print (imgDf['EncodedPixels'])
        print (id_)
        print (range(len(imgDf)))
        
        for i in range(len(imgDf)):
                rle=imgDf['EncodedPixels'].iloc[i]
                #print (rle)
                mask=rleToMask((rle), rawImg.shape[0],rawImg.shape[1])
                #print (mask.shape)
                #print (np.count_nonzero(mask))
                
                #plt.imshow(mask)
                plt.imsave(path +'/masks/' +str(id_)+'_'+str(i)+'.png',mask)
                plt.clf()
        
