#Food-Image-Classification
===================
##Project Overview

The goal of this project is to build a model that can accurately classify images of food into predefined categories. With the rise of health and fitness apps, such a model can be integrated into applications to automatically detect and log consumed food items based on user-uploaded images.This project uses deep learning to identify 34 different types of food.   It involves gathering food images, ensuring a balanced dataset, training and testing different AI models, and making the final model available online through a simple web app.

1.**Data Collection**
===================

- The dataset used for this project consists of images of various food items categorized into different classes. Each image is labeled with its corresponding food category.
- The dataset, consisting of images categorized into 34 food classes, was acquired from Kaggle.
- To download the dataset, please visit the following link:
- [click here](Food Image Classification Dataset)

2.**Data Balancing**
===================
- We made sure our food image dataset was fair by having the same number of images for each of the 34 food types.
-  We used Python programs to do this, giving each food category 200 images.
-   Then, we split the dataset into three parts:
  - Training Set: 150 images per class
  - Validation Set: 30 images per class
  - Testing Set: 20 images per class
- After balancing and splitting, we put the whole dataset on Google Drive so it's easy to use.

3.**Environment Initialization and Library Loading**
===================
- We use Google Colaboratory for this project , because its free GPU access makes training deep learning models much faster than using a CPU.
-  We started by putting our dataset on Google Drive, then created a new Colab notebook to write our code.
-  First, we connected Colab to our Drive so we could use the data.
-  Then, we imported all the Python libraries we needed, each one playing a specific role in the project.

```text
###Data manipulation and numerical computation
import numpy as np

import pandas as pd  # Data handling  

###General machine learning utilities
import sklearn

###Image processing
import cv2

###System and OS related
import os, sys
###JSON handling
import json

###Deep learning framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, BatchNormalization, ReLU, LeakyReLU
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###Transfer learning models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50

###Model loading
from tensorflow.keras.models import load_model

###Model evaluation (confusion matrix)
from sklearn.metrics import confusion_matrix
```text


4.**Image Data Preprocessing and Enhancement**
===================
- The dataset contains 34 different classes of food images.
- Each class of image is divided into training,testing  and validation images wherein 50% of images from each class are considered as training, 20% of images from each class are considered as testing and  the remaining 30% samples as validation samples.
- We used ImageDataGenerator to efficiently manage and enhance our image data. This tool lets us create variations of our training images (like rotations, flips, zooms, and color tweaks) on the fly, which helps our model learn more general features and avoid overfitting.
- Think of it as showing the model lots of slightly different versions of the same food, so it becomes better at recognizing it.
- This also helps with memory management because the images are loaded and processed as needed during training.
- We also rescale the images, which means we adjust the pixel values to be between 0 and 1. 

    ```text
    train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values  
    rotation_range=20,  # Randomly rotate images  
    width_shift_range=0.2,  # Shift images horizontally  
    height_shift_range=0.2,  # Shift images vertically  
    shear_range=0.2,  # Apply shearing transformations  
    zoom_range=0.2,  # Randomly zoom in images  
    horizontal_flip=True,  # Flip images horizontally  
    fill_mode='nearest'  # Fill in missing pixels  
)
#We only rescale the validation and test images; we don't augment them, so we can accurately measure how well the model performs on real-world data.
valid_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

- Now we will set up three image generators for training, validation, and testing data using the flow_from_dataframe method provided by TensorFlow's ImageDataGenerator class.

train_images = train_datagen.flow_from_dataframe(train_path,target_size=(224, 224),class_mode='categorical',batch_size=32)
val_images = valid_datagen.flow_from_dataframe(validation_path,target_size=(224, 224),class_mode='categorical',batch_size=32)
test_images = test_datagen.flow_from_dataframe(test_path,target_size=(224, 224),class_mode='categorical',batch_size=32)

```text
Found 5094 training images  belonging to 34 classes.
Found 680 testing images  belonging to 34 classes.
Found 1020 validated images belonging to 34 classes.
```text


5.**Implementing a Model**
===================
We tried three different deep learning  models to classify food images:

1.**Our Own Model:** We built a CNN from scratch, using layers that learn features, shrink the image size, and make the final classification.  It has 34 output neurons (one for each food type) and uses a "softmax" function to give probabilities for each type.

2.**VGG16 Model:** We used a pre-trained VGG16 model.  We froze the early layers (which already know a lot about images) and fine-tuned the later layers to recognize our specific food categories.  This "transfer learning" approach is faster and often more accurate.

3.**ResNet Model:**  Like VGG16, we used a pre-trained ResNet model. ResNet is also good at transfer learning and helps avoid some training problems.






















  








