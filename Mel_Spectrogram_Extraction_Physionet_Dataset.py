# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:45:25 2019

@author: jiang

What has been done in this file?

    1.  Try loding wav file and see how the signal looks like
    2.  Try filtering signal with Butterworth filter and see how the result signal looks like
    3.  Try computing mel spectrogram and then log it. Also, plot to see with log and without log results
    4.  Try save plot without white margin
"""

import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from tqdm import tqdm


"""
file_dir:

    source: audio file
    target: mel spectrogram pics
    first_dir: just sub_class file
    sub_dir: for categories

"""
source_dir = r'D:\CVML\Project\Heartchallenge_sound\Peter_dataset'
target_dir = r'D:\CVML\Project\Heartchallenge_sound\Peter_HeartSound_Log_Mel_Spectrogram_py'
first_dir = [r'\training-a',
             r'\training-b',
             r'\training-c',
             r'\training-d',
             r'\training-e',
             r'\training-f',
             ]
sub_dir = [r'\normal',
           r'\abnormal',]


"""
Just configuration:
    May not perfect because this has modified many times, kind of mess.
"""
class Config:
    def __init__(self,  
                 
                 lowcut=10, highcut=500, sr=2000, order=7,
                 pick_file=15,
                 n_fft=300, n_mels=32, fmin=0, fmax=1000, hop_length=100,
                 color='jet',
                 figsize = (10,5), dpi = 100):
        self.lowcut = lowcut       #lower cutoff frequency
        self.highcut = highcut     #higher cutoff frequency
        self.sr = sr               #sampling frequency
        self.order = order         #filter order: in this case [5,7]
        self.pick_file = pick_file # what is that??? I don't know either
        self.n_fft = n_fft         # nfft window lenght
        self.n_mels = n_mels       # how many mels filters?
        self.fmin = fmin           # mini frequency in plot
        self.fmax = fmax            # max frequency in plot
        self.hop_length = hop_length  # overlaping length
        self.color = color              # color style spectrogram
        self.figsize = figsize
        self.dpi = dpi
Config = Config()

"""
Butterworth filter function
"""
def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq= 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, (low, high), btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=50):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
###############################################################################



#Create files
#file_dir = r'D:\CVML\Project\Heartchallenge_sound\Physionet_HeartSound_Log_Mel_Spectrogram_py'
#os.chdir(file_dir)
#if len(os.listdir(file_dir)) == 0:
#    os.mkdir('training-a')
#    os.mkdir('training-b')
#    os.mkdir('training-c')
#    os.mkdir('training-d')
#    os.mkdir('training-e')
#    os.mkdir('training-f')
#    os.mkdir('valid')
#    for subfile in os.listdir(file_dir):
#        os.chdir(file_dir + '\\' + subfile)
#        os.mkdir('normal')
#        os.mkdir('abnormal')
    


# Loop variables
cout_1, cout_2= 0,0   # to count how many first_dir and sub_dir have been gone through
cout_wav = 0   # to count how many wav files have gone through in the loop of each sub_dir
temp_signal = [] # for temporarily store the processed signal
cout_save = 1    #count how many signal chunks are split based on one wav file

for cout_1 in range(0,len(first_dir)):
    
    for cout_2 in tqdm(range(0,len(sub_dir))):


        current_wav_dir = source_dir + first_dir[cout_1] + sub_dir[cout_2]  #current directory
            
        wav_name_list = os.listdir(current_wav_dir)    #get all file name in the current direcory
        
        for cout_wav in tqdm(range(0,len(wav_name_list))):
            # load
            signal , _= librosa.load(current_wav_dir + '\\'+ wav_name_list[cout_wav], sr = Config.sr)
            
            # filting
            signal = butter_bandpass_filter(signal, Config.lowcut, Config.highcut, Config.sr, Config.order)
            
            # normalize 
            signal = signal / np.max(np.abs(signal))
            
            
            while len(signal) > 0:
                
                # one signal is cut into 5s overlapping chunks, step is 4s
                mask = np.ones(len(signal), dtype=bool)
                if len(signal) > 4* Config.sr:
                    temp_signal = signal[:5*Config.sr] #5s
                    mask[:int(4* Config.sr-1)] = False
                    signal = signal[mask] #step:4s
                    
                    #compute mel-spectrogram
                    mel_spec = librosa.feature.melspectrogram(temp_signal, sr = Config.sr, 
                                                            n_fft = Config.n_fft, 
                                                            hop_length = Config.hop_length, 
                                                            n_mels = Config.n_mels, 
                                                            fmin = Config.fmin, fmax = Config.fmax)
                    # compute log mel - spectrogram
                    log_mel_spec = librosa.power_to_db(mel_spec, ref = np.max)
                    
                    # normalization
                    norm_log_mel_spec = librosa.util.normalize(log_mel_spec)
                    
                    ##plot and save plot
                    plt.figure(figsize = Config.figsize, dpi = Config.dpi)
                    
                    librosa.display.specshow(norm_log_mel_spec, 
                                         fmin = Config.fmin, fmax = Config.fmax,
                                         sr = Config.sr, hop_length = Config.hop_length, cmap=Config.color)
                    
                    fig = plt.gcf()
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.margins(0, 0)
                    plt.close()
                    
                    img_name = wav_name_list[cout_wav][:-4]+ '_' + str(cout_save) + '.png'
                    save_dir = target_dir + first_dir[cout_1] + sub_dir[cout_2] + '\\' + img_name 
                    
                    fig.savefig(save_dir, transparent = True, dpi = Config.dpi, 
                                pad_inches = 0, bbox_inches = 'tight')
                    cout_save += 1
                    
                else:
                    temp_signal = []
                    cout_save = 1
                    cout_wav += 1
                    break
        
            
    
    
    
    






