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
from imgaug import augmenters as imga
from nucleiDataConfigs import NucleiDatasetVal, NucleiDatasetTrain, NucleiConfig

# Root directory of the project
ROOT_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    
config = NucleiConfig()
config.display()

dataset_train = NucleiDatasetTrain()
dataset_train.load_nuclei()
dataset_train.prepare()

dataset_val = NucleiDatasetVal()
dataset_val.load_nuclei()
dataset_val.prepare()        

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


model.load_weights('Mask_RCNN/logs/nuclei20180416T0614/mask_rcnn_nuclei_0025.h5',by_name=True)

augmentation = imga.SomeOf((0, 2), [
        imga.Fliplr(0.5),
        imga.Flipud(0.5),
        imga.OneOf([imga.Affine(rotate=90),
                   imga.Affine(rotate=180),
                   imga.Affine(rotate=270)]),
        imga.Multiply((0.8, 1.5)),
        imga.GaussianBlur(sigma=(0.0, 5.0))])

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,# / 10,
            epochs=100,
            augmentation=augmentation,
            layers="all")



