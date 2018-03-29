#macro to train a mask rnn for Nuclei segmentation
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
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

from skimage.io import imread

# Root directory of the project
ROOT_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

class NucleiConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

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
    STEPS_PER_EPOCH = 100
        
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = NucleiConfig()
config.display()


class NucleiDataset(utils.Dataset):

        def load_nuclei(self):
                TRAIN_PATH = '../stage1_train/'
                # Get IDs
                image_ids = next(os.walk(TRAIN_PATH))[1]
                self.add_class("nuclei", 1, "nuclei")

                
                # Add images
                j=0
                for i in image_ids:
                        fullpath= TRAIN_PATH + i + '/images/' + i + '.png'
                        self.add_image(
                                "nuclei",
                                image_id=i,
                                path=fullpath)
                        j+=1


        def load_mask(self, image_id):
                TRAIN_PATH = '../stage1_train/'
                image_ids = next(os.walk(TRAIN_PATH))[1]
                image_id=image_ids[image_id]
                
                instance_masks = []
                class_ids = []
                
                path = TRAIN_PATH + image_id

                for mask_file in next(os.walk(path + '/masks/'))[2]:
                        mask_ = imread(path + '/masks/' + mask_file)

                        class_id = 1
                                                
                        m = mask_.astype(np.bool)

                        # Some objects are so small that they're less than 1 pixel area
                        # and end up rounded out. Skip those objects.
                        if m.max() < 1:
                                continue

                        instance_masks.append(m)
                        class_ids.append(class_id)

                mask = np.stack(instance_masks, axis=2)
                class_ids = np.array(class_ids, dtype=np.int32)
                return mask, class_ids

dataset_train = NucleiDataset()
dataset_train.load_nuclei()
dataset_train.prepare()

dataset_val = NucleiDataset()
dataset_val.load_nuclei()
dataset_val.prepare()

# Load and display a test
for i in range(0,10):
        TRAIN_PATH = '../stage1_train/'
        image = dataset_train.load_image(i)
        mask, class_ids = dataset_train.load_mask(i)
        print(class_ids)
        visualize.display_top_masks(image, mask, class_ids,['background','nuclei'])
        
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
#elif init_with == "last":
            # Load the last model you trained and continue training



# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 10,
                        epochs=2,
                        layers="all")
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_nuclei.h5")
model.keras_model.save_weights(model_path)

