# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 23:22:20 2019

@author: jiang

What has been done in this file?

    1.  Try loding pics, do simple preprocessing(convert pics from [0,255] to [0,1]), and batch them
    2.  Build CNN model and train it
    3.  See validation results labels
    4.  Plot the confusion matrix of the validation set
    5.  Save model
"""
import os
import numpy as np 
from keras.models import Sequential
from keras import regularizers
from keras.layers import Activation, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, Adadelta
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


"""
Confusion matrix function:
    arguments:
        
        cm: confusion maticx
        classes: classes name
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
###############################################################################

"""
Configuration:
    arguments:
        
        file path
        
        target_size for ImageDataGenerator function
        
        input_shape for CNN
"""

class Config:
    def __init__(self,
                 train_path = r'D:\CVML\Project\Heartchallenge_sound\Peter_HeartSound_Log_Mel_Spectrogram_py\train_valid_test_2classes\train',
                 valid_path = r'D:\CVML\Project\Heartchallenge_sound\Peter_HeartSound_Log_Mel_Spectrogram_py\train_valid_test_2classes\valid',
                 target_size = (224,224),
                 input_shape = (224,224,3),    
                 ):
        self.train_path = train_path
        self.valid_path = valid_path
        self.target_size = target_size
        self.input_shape = input_shape

        
        
config = Config()    
datagen = ImageDataGenerator(rescale=1.0/255.0)

train_batches = datagen.flow_from_directory(config.train_path, target_size = config.target_size,
                                            class_mode= 'categorical', batch_size = 32)
valid_batches = datagen.flow_from_directory(config.valid_path, target_size = config.target_size, 
                                            class_mode= 'categorical', batch_size = 32, shuffle=False)


"""
CNN Model
"""

model = Sequential([
        Dropout(0.2,input_shape = config.input_shape),    
        
        Conv2D(32, (10,10), strides=5, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        AveragePooling2D(pool_size=2),
        

        Conv2D(32, (5,5), strides=2, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        AveragePooling2D(pool_size=2),
    


        Flatten(),
        Dropout(0.2),
        Dense(128, activation='tanh'),
        Dense(2, activation='softmax')
        ])
model.summary()
model.compile(Adadelta(), loss='categorical_crossentropy', metrics=['accuracy']) #Adam(lr=0.00005)# categorical_crossentropy   
model.fit_generator(train_batches, steps_per_epoch=28,
                   validation_data=valid_batches, validation_steps=6, epochs=40, class_weight='auto' ,verbose=2)# class_weight=cw,
###############################################################################


"""
See validation predicted results
"""
result = model.predict_generator(valid_batches, steps=6, verbose=1)
labels = valid_batches.labels
###############################################################################

"""
Plot confusion matrix
"""
plt.figure(figsize = (10,10))
cm = confusion_matrix(labels[:], result.argmax(axis=1))
plot_confusion_matrix(cm,['abnormal','normal'], title = 'Confusion Matrix')
###############################################################################

"""
Save model
"""
#model.save('Heartbeat_Classification_CNN_Mel-Spectrogram_Peter_dataset.h5')



