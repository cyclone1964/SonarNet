# (Attempt) Mask R-CNN Implementation for Active Sonar Object Detection and Segmentation

**_Note - This implementation currently does not work (pivoted to U-Net based)_**

## Installation

Refer to [Mask R-CNN](https://github.com/matterport/Mask_RCNN) package page

## Data

Refer to `SonarNet/Architectures/UNet/README.md`
   
## Overview

Briefly, Mask R-CNN extends previous work that generates proposals to localize and segment objects using a Feature Pyramid Network (FPN) and a Residual Network backbone to generate bounding boxes, segmentation masks, and prediction values for each instance of an object in the image.

Mask R-CNN claims to check off the boxes of detection, classification, segmentation, and confidence score.

