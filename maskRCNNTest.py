#macro to creat csv of masks for the holdout test set, ready to upload to kaggle
#thanks to:
#-the github repo Mask-RCNN

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'Mask_RCNN'))
sys.path.append(os.path.join(os.getcwd(), 'Mask_RCNN/mrcnn'))

import random
import math
import re
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

from skimage.io import imread
from skimage.transform import resize
from nucleiDataConfigs import NucleiDatasetTest, NucleiDatasetTrain, NucleiConfigInference

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
                if (b>prev+1): run_lengths.extend((b + 1, 0))
                run_lengths[-1] += 1
                prev = b
        return run_lengths

def prob_to_rles(x, IMG_HEIGHT, IMG_WIDTH):
        totalMask=np.zeros((IMG_HEIGHT,IMG_WIDTH))
        for i in range(x.shape[2]):
                resizeMask = resize(x[:,:,i], (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

                #carefully avoid overlapping masks 
                resizeMaskInput = resizeMask-totalMask 
                totalMask = totalMask+resizeMask
                maskPoints = np.where(resizeMaskInput.T.flatten() == 1)[0]
                if len(maskPoints) == 0:
                        continue
                
                yield rle_encoding(resizeMaskInput)

                
# Root directory of the project
ROOT_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

def get_ax(rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

dataset_train = NucleiDatasetTest()
dataset_train.load_nuclei()
dataset_train.prepare()
        
inference_config = NucleiConfigInference()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = 'Mask_RCNN/logs/nuclei20180416T0614/mask_rcnn_nuclei_0025.h5'#model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

test_ids = next(os.walk('stage2_test_final/'))[1]
new_test_ids = []
rles = []

# Load and display a series of test images
for i in range(len(test_ids)):
        print (i)
        image_id = i
        image = dataset_train.load_image(i)

        #reconstructed masks
        results = model.detect([image], verbose=1)
        r = results[0]
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
         #                           dataset_train.class_names, r['scores'], ax=get_ax())

        #print (r['masks'].shape)
        rle = list(prob_to_rles(r['masks'],image.shape[0],image.shape[1]))
        if len(rle)==0:
                rle=[[1,1]]
        rles.extend(rle)
        new_test_ids.extend([test_ids[i]] * len(rle))
        #print (rle)
        #print (test_ids[i])

        
# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-rcnn.csv', index=False)
