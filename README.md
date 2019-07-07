# Deep Learning for Colon Cancer Classification

Colon cancer patch classification(benign vs Cancer) project using Densenet.

## Introduction
In this study, we propose a convolutional neural networks (CNN) for classifying colon dataset using Densely connected network. With changing the depth of the model and input size, we compared the model performance. As a result, the best model achieved 98.02% accuracy and 0.9951 AUC. Plus, as deeper depth, smaller input image decreased model performance.

## Methodology
***
## Network Architecture
![Alt text](C:\Users\HyunwooJo\Desktop\networkArchitecture2.png)
* Size of Input to Gloval Avg.Pooling Layer
![Alt text](C:\Users\HyunwooJo\Desktop\Size_before_globalavg.png)

# #Data Augmentation
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
*First fold
	*![Alt text](C:\Users\HyunwooJo\Desktop\1-fold.png)
*Second fold
	*![Alt text](C:\Users\HyunwooJo\Desktop\1-fold.png)
*Third fold
	*![Alt text](C:\Users\HyunwooJo\Desktop\1-fold.png)
***
# Experiments & Results
***

#Datasets

* H&E stained colon pathology images
* 1171 benign patches, 2472 tumor patches
* Among tumor patches, well differentiated 300, moderately differentiated 1701, poorly differentiated 471
* Provided by Kangbuk Samsung Hospital

#Dataset Sample
![Alt text](C:\Users\HyunwooJo\Desktop\dataset_sample.png)

#Results
![Alt text](C:\Users\HyunwooJo\Desktop\results.png)


## used libaries(requirements)


pytorch (version : 1.1, gpu-version) 
imaug - for image augmentation
sklearn - for K-fold cross validation, dividing train/val datasets
tensorboardX
matplotlib
numpy


## License
HyunwooJo
