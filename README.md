# ENVISION_PROJECT - (OCR - 2021)



# INTRODUCTION
![alt text](https://ieee.nitk.ac.in/virtual-expo/assets/img/envision/compsoc/ocr_image1.jpeg)

What is OCR ?

* Optical character recognition (OCR) technology is an efficient business process that saves time, cost and other resources by utilizing automated data extraction and storage capabilities.

* Optical character recognition (OCR) is sometimes referred to as text recognition. An OCR program extracts and repurposes data from scanned documents, camera images and image-only pdfs. OCR software singles out letters on the image, puts them into words and then puts the words into sentences, thus enabling access to and editing of the original content. It also eliminates the need for manual data entry.

# Convolutional Neural Networks for OCR :
It basically is a Deep Learning algorithm which takes an input image, tunes the weights and biases to predict the different features of the image and classify them. 

* Convolution Layer
* Activation Function
* Pooling Layer
* Padding
* Fully-Connected Layer

1. Convolution Layer :

The images consist of pixels and are stored in forms of matrices. The convolution layers consist of filters in the form of 3x3x1(or 3 for RGB images) or 5x5x1 square matrices containing weights. These filters superpose on the image matrix and do matrix multiplication to obtain feature maps. These feature maps contain the different characteristics of the image. We can use more than one filter on the same image to obtain different types of feature maps. For RGB images we have three channel filters which in turn form feature maps in the form of 2D matrices.
![alt text](https://ieee.nitk.ac.in/virtual-expo/assets/img/envision/compsoc/ocr_image2.jpg)

2. Activation Function :

These functions add non-linearity to the model to better predict complex features in the images. The most commonly used activation function is ReLU activation function. It maps all the negative numbers to zero and the positive numbers from 0 to infinity.

![ocr_image3](https://user-images.githubusercontent.com/88763773/182384571-34e1e346-25b7-4b21-be74-1e7b4a9b245b.jpg)




