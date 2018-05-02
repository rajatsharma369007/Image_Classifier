# Image Classifier Using Tensorflow and Keras
This project is given under the CO307 software engineering course at Department of CSE, Tezpur Central University under the guidance of Dr. S.S. Satapathy.

### Introduction
> We consider people to be experts in a field if they’ve mastered classification. Doctors  can classify between a good blood sample and a bad one. Photographers can classify if their latest shot was beautiful or not. Musicians can classify what sounds good, and what doesn’t in a piece of music. The ability to classify well takes many hours of training. We get it wrong over, and over again, until eventually we get it right. But with a quality data set, and deep learning approach, we can classify just about anything within minutes.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://thumbs.dreamstime.com/t/doctor-woman-working-microscope-laboratory-female-scientist-looking-microscope-lab-scientist-using-109583611.jpg)  
> Our aim is to develop an image classifier to classify images. But before doing that we must know that recent advances in deep learning made tasks such as Image recognition possible. Deep learning excels in recognizing objects in images as it’s implemented using 3 or more layers of convolutional neural networks where each layer is responsible for extracting one or more feature of the image.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://www.pyimagesearch.com/wp-content/uploads/2016/05/deep_learning_example.jpg)  

### Convolutional Neural Networks(CNN)  
> A CNN model works in a similar way to the neurons in the human brain. Each neuron takes an input, performs some operations then passes the output to the following neuron. Likewise, a CNN model functions by passing the input through several convolutional layers, where each layer extracts some specific features in the input.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](http://www.learnopencv.com/wp-content/uploads/2017/11/cnn-schema1.jpg)  

### Libraries : Keras and Tensorflow  
> TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.
> Keras is an open source neural network library written in Python. It is capable of running on top of TensorFlow. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible.  

### Problem Statement
> Our goal is to develop a software that classifies images based on the object present in them.  
Manual classification of a large dataset of images is neither feasible nor accurate. Therefore, this image classifier software can automate the process of image classification and thus, it can be used to distinguish objects, facial expressions, food, natural landscapes and sports, among others which has applications in the field of crime detection, computer vision, Geographic Information System (GIS) etc.  

### Design phase  
DFD Level 0  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/d1.JPG)

DFD Level l
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/d2.JPG)

DFD Level 2
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/d3.JPG)

### Coding  
> Importing the libraries  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/1.JPG)

> Defining training and validation dataset directory  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/2.JPG)  

> Creating generators for training and validation dataset  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/3.JPG)

> Preparing the Convolutional Layers  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/4.JPG)

> Removing the noise  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/5.JPG)

> Initializing the training  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/6.JPG)  

> Snippet of the training session  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/7.JPG)  

> Saving our trained model  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/8.JPG)   

> Validating the model  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt-text](https://github.com/rajatsharma369007/Image_Classifier_software/blob/master/image/9.JPG)   



