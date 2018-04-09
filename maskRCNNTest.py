#macro to creat csv of masks for the holdout test set, ready to upload to kaggle
#thanks to:
#-the github repo Mask-RCNN

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'Mask_RCNN'))

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
from nucleiDataConfigs import NucleiDatasetTest, NucleiDatasetTrain

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

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

def get_ax(rows=1, cols=1, size=8):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Change the default size attribute to control the size
        of rendered images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax

class NucleiConfig(Config):
    """Configuration for training on the kaggle nuclei
    Derives from the base Config class and overrides values specific
    to the nuclei dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200
        
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

dataset_train = NucleiDatasetTest()
dataset_train.load_nuclei()
dataset_train.prepare()

# Load and display a test
# for i in range(0,10):
#         image = dataset_train.load_image(i)
#         mask, class_ids = dataset_train.load_mask(i)
#         visualize.display_top_masks(image, mask, class_ids,['background','nuclei'])
        
class InferenceConfig(NucleiConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        
inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = 'Mask_RCNN/logs/mask_rcnn_nuclei.h5'#model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

test_ids = next(os.walk('stage1_test/'))[1]
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
        rles.extend(rle)
        new_test_ids.extend([test_ids[i]] * len(rle))
        #print (rle)
        #print (test_ids[i])

        
# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)
