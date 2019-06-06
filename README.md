# Water-Body-Extraction-from-Very-High-Spatial-Resolution-Remote-Sensing-Data-Based-on-FCNs
Python implementation of Convolutional Neural Network (CNN) proposed in academia

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches 
and mask patches to be used for model training. The CNN used here is the Fully Convolutional Network (FCN) model implemented in the paper 
'Water Body Extraction from Very High Spatial Resolution Remote Sensing Data Based on Fully Convolutional Networks' by Li L., Yan Z., 
Shen Q., Cheng G., Gao L., Zhang B. (2019).

Requirements:
- cv2
- glob
- json
- numpy
- rasterio
- keras (tensorflow backend)
