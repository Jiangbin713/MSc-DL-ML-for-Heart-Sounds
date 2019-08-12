# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:20:29 2019

@author: jiang

What has been done in this file?

    1.  Load training, validation data and simpliy processe the data (convert [0,255] into [0,1])
    2.  Load MobileNet and modify some layers
    3.  Output the result to excel file for evaluation     # The codes are not very clear, too messy
"""
import os
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Model
class Config:
    def __init__(self,
                 train_path = r'D:\CVML\Project\Heartchallenge_sound\Physionet_HeartSound_Log_Mel_Spectrogram_py\training-a\train_valid_test\train',
                 valid_path = r'D:\CVML\Project\Heartchallenge_sound\Physionet_HeartSound_Log_Mel_Spectrogram_py\training-a\train_valid_test\valid',
                 test_path = None,
                 input_size = (224,224),
                 
                 ):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.input_size = input_size
  
config = Config()

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(config.train_path, target_size = config.input_size, class_mode= 'categorical', batch_size = 10)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(config.valid_path, target_size = config.input_size, class_mode= 'categorical', batch_size = 10,shuffle=False)

MobileNet = keras.applications.mobilenet.MobileNet() #load model
MobileNet.summary()

x = MobileNet.layers[-6].output #take the last 6 layers
predictions = Dense(2, activation='softmax')(x) #replace them by the softmax layer
model = Model(inputs = MobileNet.input, output = predictions)
model.summary()

for layer in model.layers[:-20]:
    layer.trainable = False         #freeze the layers
    
model.compile(Adam(lr = 0.0005), loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=32,
                   validation_data=valid_batches, validation_steps=49, epochs=5, verbose=2)