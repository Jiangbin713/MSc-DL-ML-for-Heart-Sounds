
"""
Created on Mon Jul  8 23:22:20 2019

@author: jiang
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
print('Runing')
class Config:
    def __init__(self,
                 train_path = r'/Split_1/Split_1/train',
                 valid_path = r'/Split_1/Split_1/valid',          
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
                                            class_mode= 'categorical', batch_size = 100)
valid_batches = datagen.flow_from_directory(config.valid_path, target_size = config.target_size, 
                                            class_mode= 'categorical', batch_size = 100, shuffle=False)



#####VGG16 Model#######
model = Sequential([
        Dropout(0.2,input_shape = config.input_shape),          
        Conv2D(128, (10,10), strides=5, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        AveragePooling2D(pool_size=2),
        Conv2D(128, (5,5), strides=2, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
        BatchNormalization(),
        AveragePooling2D(pool_size=2),
        Flatten(),
        Dropout(0.2),
        Dense(256, activation='tanh'),
        Dense(2, activation='softmax')
])
model.summary()
model.compile(Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=32,validation_data=valid_batches, validation_steps=47, epochs=50, class_weight='auto' ,verbose=2)

#####See Validation Confusion Matrix######
#result = model.predict_generator(valid_batches, steps=17, verbose=1)
#labels = valid_batches.labels
#plt.figure(figsize = (10,10))
#cm = confusion_matrix(labels[:], result.argmax(axis=1))
#plot_confusion_matrix(cm,['abnormal','normal'], title = 'Confusion Matrix')


####Save Model######
model.save('Heartbeat_Classification_CNN_Mel-Spectrogram_Peter_dataset.h5')



