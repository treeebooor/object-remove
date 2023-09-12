# object-remove

An object removal from image system using deep learning image segmentation and inpainting techniques.

## Contents
1. [Overview](#overview)
2. [Source Code](src/)
3. [Report](object_remove.pdf)
4. [Results](#results)
5. [Dependencies](#dependencies)

## Overview
 Object removal from image involves two separate tasks, object detection and object removal.

 The first task is handled by the user drawing a bounding box around an object of interest to be removed. We could then remove all pixels inside the bounding box, but this could lead to loss of valuable information from the pixels in the box that are not part of the object. Instead Mask-RCNN, a state of the art instance segmentation model is used to get the exact mask of the object.  

 Filling in the image is done using DeepFillv2, an image inpainting generative adversarial network which employs a gated convolution system.
 
 The result is a complete image with the object removed. 

 <p align ="center">
  <img src="/img/diagram.png" width="1000" />
  <em></em>
 </p>

## Usage

The DeepFillv2 model needs pretrained weights from [here](https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download) provided by [this](https://github.com/nipponjo/deepfillv2-pytorch) repository which is a reimplementation of DeepFillv2 in Pytroch. Code for DeepFillv2 model was borrowed and slightly modified from there.  



Make sure to put the weights pth file in [src/models/](/src/models/).

To run on example image, 
```
./src/main.py [path of image]
```
When drawing bounding box, press 'r' to clear bounding box and reset image. Once box is drawn press 'c' to continue. 

*Drawing bouding boxes is sometimes slow.


## Results
The following are some results of the system. The user selected bounding box is shown along with the masked image and inpainted final result. 

<p align ="center">
  <img src="/img/example1.png" width="1000" />
  <em></em>
</p>
<p align ="center">
  <img src="/img/example2.png" width="1000" />
  <em></em>
</p>

## Dependencies
- python3
- torch
- torchvision
- cv2
- matplotlib
- numpy


