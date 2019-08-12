# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:22:22 2019

@author: jiang
"""

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np 


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
    plt.title(title,fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=20)
    plt.yticks(tick_marks, classes,fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)
    
class Config:
    def __init__(self,
                 valid_path = r'D:\CVML\Project\Heartchallenge_sound\Spectrogram\Physionet_HeartSound_Log_Mel_Spectrogram_py\Split_1\Split_1\valid',
                 target_size = (224,224),
                 input_shape = (224,224,3) 
                 ):
        self.valid_path = valid_path
        self.target_size = target_size
        self.input_shape = input_shape

config = Config()

datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_batches = datagen.flow_from_directory(config.valid_path, target_size = config.target_size, 
                                            class_mode= 'categorical', batch_size = 100, shuffle=False)

model = load_model(r'D:\CVML\Project\Heartchallenge_sound\Codes\Py_code\DeepLearn\Physionet_test_experiments\test_2\Heartbeat_Classification_CNN_Mel-Spectrogram_Peter_dataset.h5')

result = model.predict_generator(valid_batches, steps = 47, verbose = 1)
labels = valid_batches.labels

plt.figure(figsize = (10,10))
cm = confusion_matrix(labels[:], result.argmax(axis=1))
plot_confusion_matrix(cm,['abnormal','normal'], title = 'Confusion Matrix')