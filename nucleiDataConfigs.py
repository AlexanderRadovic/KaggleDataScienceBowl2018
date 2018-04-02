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

class NucleiDataset(utils.Dataset):

        def load_nuclei(self):
                TRAIN_PATH = 'stage1_train/'
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
                TRAIN_PATH = 'stage1_train/'
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

