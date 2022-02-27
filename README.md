# Mask Detection
#####
[![Opencv](https://editor.analyticsvidhya.com/uploads/800882.png)](https://opencv.org/)
[![Python](https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png)](https://www.python.org/)
[![Numpy](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeXkEA3c2hxTcrZWwVnniXAFiqai51196osD9FWL0_D_Ca7fOT)](https://www.numpy.org/)
[![Tensorflow](https://d20vrrgs8k4bvw.cloudfront.net/images/courses/logos/logo-color-tensorflow.png)](https://www.tensorflow.org/)
&nbsp;

## Description

Detect and Classify faces as wearing masks or not wearing masks by building powerful Deep Learning models with Tensorflow. 

Detect faces using HaarCascade Classifiers and apply the trained DL model on the faces detected, to check if masks are present or not.

### Technologies
1.  [Python] 
The python is one of the most accessible programming languages available because **it has simplified syntaxes, which gives more emphasis on natural language**. Due to its ease of learning and usage, python codes can be easily written and executed much faster than other programming languages.
2. [Opencv]
OpenCV is an open-source library that includes several hundreds of computer vision algorithms.

3. [Numpy]
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
4. [Tensorflow]
TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.


## Files
1.  [maskModel.py] - Program to build and train Neural Network to detect masks.
2. [app.py] - Main program used to detect (predict) faces with or without masks.
3. [Saved Model] - Pre-trained model used for detecting masks.
4. [Archive] - Zip file containing the training, testing and validation data for TF model.
5. [HaarCascade-FrontalFace] - XML files containing the Haar-Features data pertaining to our face.

## Setup
##### This was built on Windows 10.
#### IMPORTANT:
- Unzip the **Archive.zip** file before proceeding to build and train your DL model.
- Create an empty **input** folder to temporarily hold the frames for predicting.
### Download
##### Ensure that you have downloaded Python on your system.
Click the python thumbnail to go to Python Download Page
[
![Python](https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png)](https://www.python.org/downloads/)

###  Installations
##### Install [OpenCV] with pip
```sh
pip install opencv-python
```
##### Install [Numpy] with pip
```sh
pip install numpy
```
##### Install [Tensorflow] with pip
##### Click on the TF Logo given below, to check the system requirements and configure your computer.
[![Tensorflow Installation](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQK_Rna8Mfenppf-umArLMGBygp1jP_RGWfTQ&usqp=CAU)](https://www.tensorflow.org/install/pip)
## Download Repository
```sh
git clone https://github.com/Alpha-github/Mask_detection.git
```

[Opencv]: <https://opencv.org/>
[Python]: <https://www.python.org/>
[Numpy]: <https://www.numpy.org/>
[Tensorflow]:<https://www.tensorflow.org/>
[Tensorflow Installation]:<https://www.tensorflow.org/install/pip>
[maskModel.py]:<https://github.com/Alpha-github/Mask_Detection/blob/master/maskModel.py>
[app.py]:<https://github.com/Alpha-github/Mask_Detection/blob/master/app.py>
[Saved Model]: <https://github.com/Alpha-github/Mask_Detection/blob/master/saved_mask_model.h5>
[Archive]:<https://github.com/Alpha-github/Mask_Detection/blob/master/archive.zip>
[HaarCascade-FrontalFace]:<https://github.com/Alpha-github/Mask_Detection/blob/master/haarcascade_frontalface_default.xml>


