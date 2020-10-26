# Deep Learning for Colon Cancer Classification

Colon cancer patch classification(benign vs Cancer) project using Dense-net.(2019.01~2019.06)

## Introduction

In this study, we propose a convolutional neural networks (CNN) for classifying colon dataset using Densely connected network. With changing the depth of the model and input size, we compared the model performance. As a result, the best model achieved 98.02% accuracy and 0.9951 AUC. Plus, as deeper depth, smaller input image decreased model performance.

# Methodology

# Network Architecture

![model](https://user-images.githubusercontent.com/43023361/60765548-1c9d6280-a0d7-11e9-922b-63b20dc7a31c.png)

* Size of Input to Gloval Avg.Pooling Layer
  ![Size_before_globalavg](https://user-images.githubusercontent.com/43023361/97193302-342cbd00-17ec-11eb-92df-ab70283b88d8.png)

## Data Augmentation

* Affine : Random scale, translation, rotation, shear.
* Horizontal and vertical flipping
* Texture : Gaussian blur, Median blur and Gaussian noise
* Color : Adding Hue, saturation, linear contrast

## Training Method

* Adam optimizer, default parameters 
* 3 fold cross validation
* 40 epochs
* Cross-entropy loss

## Training visualization (TensorboardX)

* First fold
  *
  * ![1-fold](https://user-images.githubusercontent.com/43023361/60765559-66864880-a0d7-11e9-9683-6bd889e5c0e7.png)
* Second fold
  * 
  * ![2fold](https://user-images.githubusercontent.com/43023361/60765637-7d796a80-a0d8-11e9-9f88-4fe9c7771cc1.png)
* Third fold
  * 
  * ![3fold](https://user-images.githubusercontent.com/43023361/60765643-94b85800-a0d8-11e9-919f-bff70c551439.png)

# Datasets

* H&E stained colon pathology images
* 1171 benign patches, 2472 tumor patches
* Among tumor patches, well differentiated 300, moderately differentiated 1701, poorly differentiated 471
* Provided by Kangbuk Samsung Hospital

## Dataset Sample

![dataset_sample](https://user-images.githubusercontent.com/43023361/60765567-87e73480-a0d7-11e9-81b0-c09027d2e926.png)

# Results

![results](https://user-images.githubusercontent.com/43023361/97193412-5292b880-17ec-11eb-868d-769d8f2d52f2.png)

# used libaries(requirements)

* pytorch (version : 1.1, gpu-version) 
* imaug - for image augmentation
* sklearn - for K-fold cross validation, dividing train/val datasets
* tensorboardX
* matplotlib
* numpy

