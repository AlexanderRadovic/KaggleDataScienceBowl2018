#class inheriting from the Mask-RCNN packages Dataset class
#replaces some methods so that the kaggle nuclei dataset can be
#loaded correctly
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

        def load_nuclei(self, test = False, randomseed = 1337):
                TRAIN_PATH = 'stage1_train/'
                # Get IDs
                image_ids = next(os.walk(TRAIN_PATH))[1]

                np.random.seed(randomseed)
                indicies=np.arange(len(image_ids))
                np.random.shuffle(indicies)
                
                self.add_class("nuclei", 1, "nuclei")

                if test == False:
                        indicies = indicies[:int(0.9*len(image_ids))]
                else:
                        indicies = indicies[int(0.9*len(image_ids)):]
                
                # Add images
                for i in indicies:
                        fullpath= TRAIN_PATH + image_ids[i] + '/images/' + image_ids[i] + '.png'
                        self.add_image(
                                "nuclei",
                                image_id=i,
                                path=fullpath)

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

