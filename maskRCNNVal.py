#macro to check performance on validation set, making plots to help explore behaviour
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
from nucleiDataConfigs import NucleiDatasetVal, NucleiDatasetTrain, NucleiConfigInference

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

dataset_val = NucleiDatasetVal()
dataset_val.load_nuclei()
dataset_val.prepare()

# Load and display a test
# for i in range(0,10):
#         image = dataset_val.load_image(i)
#         mask, class_ids = dataset_val.load_mask(i)
#         visualize.display_top_masks(image, mask, class_ids,['background','nuclei'])
                
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


# Load and display a series of test images
for i in range(0,10):
        image_id = i

        
        image=dataset_val.load_image(i)
        
        #reconstructed masks
        results = model.detect([image], verbose=1)
        r = results[0]

        print (len(r['masks']))
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset_val.class_names, r['scores'], ax=get_ax())

        plt.savefig('test/'+str(i)+'.png')
        

