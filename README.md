# Data Bowl 2018
The repository contains a number of python scripts designed to help solve the [2018 Kaggle Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). Two solutions are developed, one based on the UNet for image segmentation, and the other based on MaskRCNN for instanced segmentation. As we need to not just label nuclei but distinct nuclei an instanced segmentation would seem like a superior approach, but my best result is with a relatively simple UNet.

My final solution implemented a kmeans based reclustering that seemed to dramatically help with the stage1 test set, however it proved a mistake in the final round and actually degraded performance on the stage 2 test set. I suspect this is to do with the fine tuning of when to switch between hypotheses about the number of clusters. In similiar future projects I would focus on instancing via mask-rcnn in combination with unet segmentation.

Many thanks to the git repo's:
- [UNet For Keras](https://github.com/zizhaozhang/unet-tensorflow-keras)
- [MaskRCNN For Keras](https://github.com/matterport/Mask_RCNN)
Which I have found particularly useful.

## Data Exploration, Formating Scripts:
- **dataSetExploration.py** a script to plots some dataset examples and characteristics of the dataset
- **convertToNPY.py** convert our sample to a numpy file for ease of use, tools to apply dataset augmentations optional here
- **testGenerator.py** script to check generator behavior, and to explore possible dataset augmentations important for shared image/mask augmentations

## UNet Solution Scripts
- **simpleUNet.py** a script to train a simple UNet on the nuclei dataset
- **simpleUNetValPerformance.py** a script to check UNet performance on the valdiation dataset, in particular making plots of true and predicted masks to compare
- **simpleUNetTestPerformance.py** a script to produced the final kaggle submission csv on the test dataset
- **modelZoo.py** local copy of the UNet definition
- **simpleUNetAugmentInput.py** first attempt at on the fly data augmentation for UNet Training
- **finalUNetTestSolution.py** more complete solution where I recluster each group of connected hits using a kmeans algorithm
- **cluster101.py** simple script to test different clustering algorithms

## MaskRCNN Scripts
- **maskRCNNTrain.py** a script to train a maskRCNN based approach to nuclei masking
- **maskRCNNVal.py** a script to check maskRCNN performance on the validation dataset, in particular making plots of true and predicted masks to compare
- **maskRCNNTest.py** a script to produce the final kaggle submission csv on the test dataset
- **nucleiDataConfigs.py** classes in the style of the MaskRCNN implementation I use, which describe how to load the training images.


