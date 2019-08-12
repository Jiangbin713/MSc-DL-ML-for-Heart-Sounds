# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 23:24:37 2019

@author: jiang

What has been done in this file?

    1.  Load files 
    2.  set split rate
    3.  move the files
"""

import os
import random
import shutil

def moveFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir) #get a list of all files
    filenum = len(pathDir) 
    rate = 0.7                   # split rate
    picknum = int(filenum*rate)
    sample = random.sample(pathDir, picknum) # file name to be picked
    print(sample)
    for name in sample:
        shutil.move(fileDir+name,tarDir+name)
        #os.remove(tarDir+name)    # Delet
    return


original_path = r'D:\CVML\Project\Heartchallenge_sound\Physionet_HeartSound_Log_Mel_Spectrogram_py\copy\training-e'
target_path = r'D:\CVML\Project\Heartchallenge_sound\Physionet_HeartSound_Log_Mel_Spectrogram_py\train'

moveFile(original_path+r'\normal\\',target_path+r'\normal\\')            