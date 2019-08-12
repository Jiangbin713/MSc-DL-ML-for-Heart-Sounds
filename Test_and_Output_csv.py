# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:51:48 2019

@author: jiang

What has been done in this file?

    1.  Load trained model 
    2.  Load test dataset and simply process them like converting [0,255] into [0,1]
    3.  Output the result to excel file for evaluation     # The codes are not very clear, too messy
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
import pandas as pd
from keras.models import load_model


class Config:
    def __init__(self,
                 train_path = r'D:\CVML\Project\Heartchallenge_sound\Peter_HeartSound_Log_Mel_Spectrogram_py\train_valid_test_2classes\train',
                 valid_path = r'D:\CVML\Project\Heartchallenge_sound\Peter_HeartSound_Log_Mel_Spectrogram_py\train_valid_test_2classes\valid',
                 test_path_A =r'D:\CVML\Project\Heartchallenge_sound\Peter_HeartSound_Log_Mel_Spectrogram_py\A_unlabelledtest',
                 test_path_B =r'D:\CVML\Project\Heartchallenge_sound\Peter_HeartSound_Log_Mel_Spectrogram_py\B_unlabelledtest',
                 target_size = (224,224),    #for ImageDataGenerator
                 input_shape = (224,224,3),  #for CNN
                 
                 ):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path_A = test_path_A
        self.test_path_B = test_path_B
        self.target_size = target_size
        self.input_shape = input_shape
        self.test_path_A = test_path_A
        self.test_path_B = test_path_B

config = Config()

#Load trained model
model = load_model(r'D:\CVML\Project\Heartchallenge_sound\Py_code\DeepLearn\Heartbeat_Classification_CNN_Mel-Spectrogram_Peter_dataset.h5')
##############################################################################




#Test and convert the result into excel file
test_data_file_dir = config.test_path_B  # Choose which dataset to be used

datagen = ImageDataGenerator(rescale=1.0/255.0) # test data loading and preprocessing
test_batches = datagen.flow_from_directory(test_data_file_dir, target_size = config.target_size, 
                                          batch_size = 10,shuffle=False)

test_results = model.predict_generator(test_batches, steps=len(test_batches), verbose=1)
test_results = np.around(test_results)     #get the result 0 or 1 : abnormal or normal

test_data_name_list = os.listdir(test_data_file_dir + r'\test') #get spectrogram file name list

test_dict = {} #Dict {index : filename, predicted results}

#fill the information into the dict
for num in range(0,len(test_data_name_list)):
    test_data_dir = test_data_file_dir + r'\test' + '\\' + test_data_name_list[num]
    test_dict[num] = [test_data_name_list[num], None]

for num in range(0,len(test_dict)):
    test_dict[num][1] = test_results[num].astype('int') #fill labels
    
#load evaluation excel file
Evaluation_csv_dir = r'D:\CVML\Project\Heartchallenge_sound\Peter_dataset\Evaluation_Files\Challenge2_evaluation_sheet.xlsx'
if test_data_file_dir == config.test_path_A:
    Evaluation_list = pd.read_excel(Evaluation_csv_dir,usecols =[0],sheet_name='Dataset A')
    postfix_num = -6
else:
    Evaluation_list = pd.read_excel(Evaluation_csv_dir,usecols =[0],sheet_name='Dataset B')
    postfix_num = -6
    
#Voting
Voting_result_dict = {}  #Majority voting results
voting = 0
count = 0  #count how many pics from the same wav file
new_index = 0

for num in range(0,len(test_dict)):
    if num < len(test_dict)-2:
        dict_file_segment_up = ''.join(test_dict[num][0])[:postfix_num] #original file name
        
        dict_file_segment_down = ''.join(test_dict[num+1][0])[:postfix_num] #next original file  check if the same
       
        if dict_file_segment_up == dict_file_segment_down:   #same
            voting += test_dict[num][1][1]   #pick the second label
            count +=1
        if dict_file_segment_up != dict_file_segment_down: #if not the same
            voting += test_dict[num][1][1]
            count +=1
            if voting >= count/2:  
                Voting_result_dict[new_index] = [dict_file_segment_up, 1]  #noamal
                 #clear and index +1
                voting = 0  
                count = 0
                new_index += 1
            else:
                Voting_result_dict[new_index] = [dict_file_segment_up, 0] #abnormal
                #clear and index +1
                voting = 0
                count = 0
                new_index += 1
    if num == len(test_dict)-2 and ''.join(test_dict[num][0])[:postfix_num] ==  ''.join(test_dict[num+1][0])[:postfix_num]: #The last file
        voting += test_dict[num][1][1]
        count += 1
        dict_file_segment_last = ''.join(test_dict[num+1][0])[:postfix_num]
        if voting >= count/2:
            Voting_result_dict[new_index] = [dict_file_segment_last, 1]
        else:
            Voting_result_dict[new_index] = [dict_file_segment_last, 0]


#Match the Evaluation excel file order
for num in range(0, len(Voting_result_dict)):
    if test_data_file_dir == config.test_path_A:
        Voting_result_dict[num][0] = Voting_result_dict[num][0] + '.aif'
    else:
        Voting_result_dict[num][0] = Voting_result_dict[num][0] + '.aiff'
     
excel_result = {}

for num in range(0,len(Voting_result_dict)):
    if test_data_file_dir == config.test_path_A:
        if Voting_result_dict[num][0] in Evaluation_list['Dataset A'].tolist():
            excel_result[  np.where(Evaluation_list == Voting_result_dict[num][0])[0][0]   ] = [    Voting_result_dict[num][0] ,  Voting_result_dict[num][1]    ]
    else:
        if Voting_result_dict[num][0] in Evaluation_list['Dataset B'].tolist():
            excel_result[  np.where(Evaluation_list == Voting_result_dict[num][0])[0][0]  ] = [    Voting_result_dict[num][0] ,  Voting_result_dict[num][1]   ]
                
excel_file = pd.DataFrame(columns=['file_name','binary_result'])

for n in excel_result.keys():
    excel_file.loc[n,'file_name'] = excel_result[n][0]
    excel_file.loc[n,'binary_result'] = excel_result[n][1]


excel_file.sort_index(inplace=True)       

if test_data_file_dir == config.test_path_A:
        
    excel_file.to_excel('D:\CVML\Project\Heartchallenge_sound\Py_code\DeepLearn\Peter_results_A.xls')

else:
    
    excel_file.to_excel('D:\CVML\Project\Heartchallenge_sound\Py_code\DeepLearn\Peter_results_B.xls')


