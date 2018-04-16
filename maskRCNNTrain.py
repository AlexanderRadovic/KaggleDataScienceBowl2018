#macro to train a mask rnn for Nuclei segmentation
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
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

from skimage.io import imread
from nucleiDataConfigs import NucleiDatasetVal, NucleiDatasetTrain, NucleiConfig

# Root directory of the project
ROOT_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    
config = NucleiConfig()
config.display()

dataset_train = NucleiDatasetTrain()
dataset_train.load_nuclei()
dataset_train.prepare()

dataset_val = NucleiDatasetVal()
dataset_val.load_nuclei()
dataset_val.prepare()

# Load and display a test
# for i in range(0,10):
#         image = dataset_val.load_image(i)
#         mask, class_ids = dataset_val.load_mask(i)
#         visualize.display_top_masks(image, mask, class_ids,['background','nuclei'])
        

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
#init_with = "imagenet"  # imagenet, coco, or last

#if init_with == "imagenet":
#            model.load_weights(model.get_imagenet_weights(), by_name=True)
#elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
#            model.load_weights(COCO_MODEL_PATH, by_name=True,
#                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
#                                        "mrcnn_bbox", "mrcnn_mask"])
#elif init_with == "last":
            # Load the last model you trained and continue training



# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
#model.train(dataset_train, dataset_val,
#            learning_rate=config.LEARNING_RATE,
#            epochs=2,
#            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,# / 10,
                        epochs=30,
                        layers="all")



