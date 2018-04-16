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
from skimage.color import rgb2gray

class NucleiConfig(Config):
    """Configuration for training on the kaggle nuclei
    Derives from the base Config class and overrides values specific
    to the nuclei dataset.

    Updated to include some suggested settings from the Mask_RCNN git repo (thanks!).
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    #backbone network architecture
    #two options here, we want the smaller one
    BACKBONE = "resnet50"
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
    DETECT_MIN_CONFIDENCE = 0
    
    #As large as can fit on my local gpu. Crop to keep as much data as possible
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    IMAGE_MIN_SCALE = 2.0

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # ROIs kept after non-maxium suppression
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    #Non-max suppresion threshold to filter RPHN proposals
    RPN_NMS_THRESHOLD = 0.9
    
    # Huge range in number of ROIS expected. Split the difference for training.
    TRAIN_ROIS_PER_IMAGE = 128

    #Image means, subtract to improve training. Because I scale and add an extra channel
    #of information mine is a little different to the RGB standard
    MEAN_PIXEL = np.array([0.5,0.5,0.5,0.5])

    #Resize mask to lower memory load
    USE_MINI_MASK = True
    MINI_MASK_SHAPE= (56,56)

    #Max number of GT masks for training
    MAX_GT_INSTANCES = 200

    #Max number of final detections per image
    DETECTION_MAX_INSTANCES=400
    
    # Cover one pass of the data each epoch
    STEPS_PER_EPOCH = 221
        
    # Pass through whole validation sample each epoch
    VALIDATION_STEPS = 25

class NucleiConfigInference(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7
    
class NucleiDatasetTrain(utils.Dataset):

        def load_image(self, image_id):
                """Load the specified image and return a [H,W,4] Numpy array.
                """
                imageLoc=self.image_info[image_id]['path']
                unetLoc=imageLoc.replace('.png','_unetsol.png')
                # Load image
                base_image = imread(self.image_info[image_id]['path'])
                unet_image = imread(self.image_info[image_id]['path'])
        
                div = base_image.max(axis=tuple(np.arange(1,len(base_image.shape))), keepdims=True) 
                div[div < 0.01*base_image.mean()] = 1. # protect against too small pixel intensities
                base_image = base_image.astype(np.float32)/div

                unet_image=rgb2gray(unet_image)
                div = unet_image.max(axis=tuple(np.arange(1,len(unet_image.shape))), keepdims=True) 
                div[div < 0.01*unet_image.mean()] = 1. # protect against too small pixel intensities
                unet_image = unet_image.astype(np.float32)/div

                image=np.zeros((base_image.shape[0],base_image.shape[1],4))
                image[:,:,:3]=base_image[..., :3]
                image[:,:,3:]=np.reshape(unet_image,(unet_image.shape[0],unet_image.shape[1],1))
        
                return image

        
        def load_nuclei(self):
                TRAIN_PATH = 'stage1_train/'
                # Get IDs
                image_ids = next(os.walk(TRAIN_PATH))[1]

                np.random.seed(1337)
                indicies=np.arange(len(image_ids))
                np.random.shuffle(indicies)
                
                self.add_class("nuclei", 1, "nuclei")

                indicies = indicies[:int(0.9*len(image_ids))]
                
                # Add images
                j=0
                for i in indicies:
                        fullpath= TRAIN_PATH + image_ids[i] + '/images/' + image_ids[i] + '.png'
                        self.add_image(
                                "nuclei",
                                image_id=j,
                                path=fullpath)
                        j+=1


        def load_mask(self, image_id):

                TRAIN_PATH = 'stage1_train/'
                image_ids = next(os.walk(TRAIN_PATH))[1]


                np.random.seed(1337)
                indicies=np.arange(len(image_ids))
                np.random.shuffle(indicies)
                indicies = indicies[:int(0.9*len(image_ids))]
                
                image_id=image_ids[indicies[image_id]]
                                
                instance_masks = []
                class_ids = []
                
                path = TRAIN_PATH + image_id

                for mask_file in next(os.walk(path + '/masks/'))[2]:
                        mask_ = imread(path + '/masks/' + mask_file)
                        if mask_.ndim != 2:
                            mask_=mask_[:,:,0]
                            mask_=mask_>250
                            
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



class NucleiDatasetVal(utils.Dataset):
        
        def load_image(self, image_id):
                """Load the specified image and return a [H,W,4] Numpy array.
                """
                imageLoc=self.image_info[image_id]['path']
                unetLoc=imageLoc.replace('.png','_unetsol.png')
                # Load image
                base_image = imread(self.image_info[image_id]['path'])
                unet_image = imread(self.image_info[image_id]['path'])
        
                div = base_image.max(axis=tuple(np.arange(1,len(base_image.shape))), keepdims=True) 
                div[div < 0.01*base_image.mean()] = 1. # protect against too small pixel intensities
                base_image = base_image.astype(np.float32)/div

                unet_image=rgb2gray(unet_image)
                div = unet_image.max(axis=tuple(np.arange(1,len(unet_image.shape))), keepdims=True) 
                div[div < 0.01*unet_image.mean()] = 1. # protect against too small pixel intensities
                unet_image = unet_image.astype(np.float32)/div

                image=np.zeros((base_image.shape[0],base_image.shape[1],4))
                image[:,:,:3]=base_image[..., :3]
                image[:,:,3:]=np.reshape(unet_image,(unet_image.shape[0],unet_image.shape[1],1))
        
                return image

        
        def load_nuclei(self):
                TRAIN_PATH = 'stage1_train/'
                # Get IDs
                image_ids = next(os.walk(TRAIN_PATH))[1]

                np.random.seed(1337)
                indicies=np.arange(len(image_ids))
                np.random.shuffle(indicies)
                
                self.add_class("nuclei", 1, "nuclei")

                indicies = indicies[int(0.9*len(image_ids)):]
                
                # Add images
                j=0
                for i in indicies:
                        fullpath= TRAIN_PATH + image_ids[i] + '/images/' + image_ids[i] + '.png'
                        self.add_image(
                                "nuclei",
                                image_id=j,
                                path=fullpath)
                        j+=1


        def load_mask(self, image_id):

                TRAIN_PATH = 'stage1_train/'
                image_ids = next(os.walk(TRAIN_PATH))[1]

                np.random.seed(1337)
                indicies=np.arange(len(image_ids))
                np.random.shuffle(indicies)
                indicies = indicies[int(0.9*len(image_ids)):]
                
                image_id=image_ids[indicies[image_id]]
                                
                instance_masks = []
                class_ids = []
                
                path = TRAIN_PATH + image_id

                for mask_file in next(os.walk(path + '/masks/'))[2]:
                        mask_ = imread(path + '/masks/' + mask_file)
                        if mask_.ndim != 2:
                            mask_=mask_[:,:,0]
                            mask_=mask_>250
                        
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



class NucleiDatasetTest(utils.Dataset):

        def load_image(self, image_id):
                """Load the specified image and return a [H,W,4] Numpy array.
                """
                imageLoc=self.image_info[image_id]['path']
                unetLoc=imageLoc.replace('.png','_unetsol.png')
                # Load image
                base_image = imread(self.image_info[image_id]['path'])

                if base_image.ndim==2:
                    placeHolder=np.zeros((base_image.shape[0],base_image.shape[1],3))
                    placeHolder[:,:,0]=base_image
                    placeHolder[:,:,1]=base_image
                    placeHolder[:,:,2]=base_image
                    base_image=placeHolder

                unet_image = imread(self.image_info[image_id]['path'])
        
                div = base_image.max(axis=tuple(np.arange(1,len(base_image.shape))), keepdims=True) 
                div[div < 0.01*base_image.mean()] = 1. # protect against too small pixel intensities
                base_image = base_image.astype(np.float32)/div

                unet_image=rgb2gray(unet_image)
                div = unet_image.max(axis=tuple(np.arange(1,len(unet_image.shape))), keepdims=True) 
                div[div < 0.01*unet_image.mean()] = 1. # protect against too small pixel intensities
                unet_image = unet_image.astype(np.float32)/div

                image=np.zeros((base_image.shape[0],base_image.shape[1],4))
                image[:,:,:3]=base_image[..., :3]
                image[:,:,3:]=np.reshape(unet_image,(unet_image.shape[0],unet_image.shape[1],1))
        
                return image

        
        def load_nuclei(self):
                TRAIN_PATH = 'stage2_test_final/'
                # Get IDs
                image_ids = next(os.walk(TRAIN_PATH))[1]

                indicies=np.arange(len(image_ids))
                
                self.add_class("nuclei", 1, "nuclei")

                # Add images
                j=0
                for i in indicies:
                        fullpath= TRAIN_PATH + image_ids[i] + '/images/' + image_ids[i] + '.png'
                        self.add_image(
                                "nuclei",
                                image_id=j,
                                path=fullpath)
                        j+=1



