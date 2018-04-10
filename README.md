## Data Bowl 2018
The repository contains a number of python scripts designed to help solve the [2018 Kaggle Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). Two solutions are developed, one based on the UNet for image segmentation, and the other based on MaskRCNN for instanced segmentation. As we need to not just label nuclei but distinct nuclei an instanced segmentation would seem like a superior approach, but my best result so far is with a relatively simple UNet.

Many thanks to the git repo's:
-[UNet For Keras](https://github.com/zizhaozhang/unet-tensorflow-keras)
-[MaskRCNN For Keras](https://github.com/matterport/Mask_RCNN)
Which I have found particularly useful.

# Data Exploration, Formating Scripts:
-dataSetExploration.py								   					
-convertToNPY.py
-inspectAugment.py 	    
-testGenerator.py

# UNet Solution Scripts
-simpleUNet.py   
-simpleUNetValPerformance.py
-simpleUNetTestPerformance.py
-modelZoo.py	      	   
-simpleUNetAugmentInput.py 

# MaskRCNN Scripts
-maskRCNNTrain.py
-maskRCNNVal.py	  
-maskRCNNTest.py
-nucleiDataConfigs.py

# TODO:
-Explore augmentation through zooming on larger images
-Explore splitting images up rather than resizing
-Explore more nuanced conventional computer vision tools to split overlapping masks from the UNet, in particular the watershed algorithm.
-Explore using UNet output as an extra channel for MaskRCNN

