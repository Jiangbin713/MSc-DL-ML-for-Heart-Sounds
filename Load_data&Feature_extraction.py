# -*- coding: utf-8 -*-
"""
Spyder Editor

What has been done in this file?

    1.  Load dataset A and dataset B according to their catergories
    2.  Each wav file is cut into several 3s chunks and then is extracted features
    3.  Convert data in to numpy array type
    4.  Merge the features
    5.  Normalization
    6.  Save Params
    
    
    
Dataset: 
    @misc{pascal-chsc-2011,
       author = "Bentley, P. and Nordehn, G. and Coimbra, M. and Mannor, S.",
       title = "The {PASCAL} {C}lassifying {H}eart {S}ounds {C}hallenge 2011 {(CHSC2011)} {R}esults",
       howpublished = "http://www.peterjbentley.com/heartchallenge/index.html"} 
"""

"""
Bad things needed to be fixed:  Some are lists and Some are numpy array

"""


import os
import numpy as np
import librosa
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from python_speech_features import mfcc
from scipy.signal import find_peaks_cwt
from entropy import *
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from statsmodels.robust import mad
import pywt

source_dir = r'D:\CVML\Project\Heartchallenge_sound\Dataset\Peter_dataset'
first_dir = [
             r'\A_normal',
             r'\A_murmur',
             r'\A_extrahls',
             r'\A_artifact',
                          
             r'\B_normal',
             r'\B_murmur',
             r'\B_extrahls',
             
             r'\A_unlabelledtest',
             r'\B_unlabelledtest']

class Config:
    def __init__(self,                  
                 lowcut=10, highcut=500, sr=2000, order=7,
                 n_fft=256, n_filt=12, n_feat = 6, fmin=0, fmax=1000,
                 mfcc_len_threshold = 50):
        self.lowcut = lowcut        #lower cutoff frequency
        self.highcut = highcut      #higher cutoff frequency
        self.sr = sr            #sampling frequency
        self.order = order     #filter order: in this case [5,7]
        self.n_fft = n_fft    # nfft window lenght
        self.n_filt = n_filt   # how many mels filters?
        self.n_feat = n_feat   # how many mfcc feats
        self.fmin = fmin     # mini frequency in plot
        self.fmax = fmax    # max frequency in plot
        self.mfcc_len_threshold = mfcc_len_threshold  #mfcc lenght threshold if lenght is less than that, just dump the chunk


"""
Define butterworth filter
"""
def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq= 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, (low, high), btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y



"""
Wavelet decomposition and reconstruction
"""
def waveletSmooth( x, wavelet="db4", level=1 ):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # calculate a threshold
    sigma = mad( coeff[-level] )
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec( coeff, wavelet, mode="per" )
    return y



"""
Feature: Extract Zero Crossing Rate
"""
def stZCR(signal):
    """Computes zero crossing rate of frame"""
    count = len(signal)
    countZ = np.sum(np.abs(np.diff(np.sign(signal)))) / 2
    return (np.float64(countZ) / np.float64(count-1.0))




"""
Find peaks indexes of a signal samples
    Use find_peak_cwt function to find all peaks and then use threshold to filter these peak points
""" 
def find_peaks(sample, set_name):
	"""Gets a list of peaks for each sample"""
	if set_name == 'A':
		interval = 200
		r = 5
	else:
		interval = 200
		r = 5


	indexes = find_peaks_cwt(sample, np.arange(1, r))
	peaks = []    #peak indexes storage
	for i in indexes:
		if sample[i] > 0.15:     # first level threshold: global thresholding
			peaks.append(i)

	if len(peaks) > 1:       # if more than one peak
		i = 1
		start = 0
		tmp_array = []
		max_peak = sample[peaks[start]]
		max_ind = start
		while i < len(peaks):                            ###################Find local maximum####################
			if peaks[i] <= (peaks[start] + interval):    #  check if the distance between the current point and 
				max_peak = sample[peaks[i]]           #   the next peak point is within 0.1s length
				if sample[peaks[i]] > max_peak:
					max_ind = i
				if i == len(peaks)-1:
					tmp_array.append(peaks[max_ind])
					break
				i += 1
			else:                                        # if the distance larger than 0.1s length
				tmp_array.append(peaks[max_ind])
				start = i 
				max_ind = start
				max_peak = sample[peaks[start]]
				i += 1
		peaks = tmp_array

	return np.array(peaks)



"""
Roughly find the segmentation bounds index, return 2D array
    Idea to find the maximum distance points pair
"""

def get_S1S2_bounds(data, peaks, set_name):
    #finding difference between all peaks in every file
    all_diffs = np.diff(peaks) 
  
    #finding maximum difference or diastole period
    # and then labelling the first peak as s2 and second peak as s1
    max_index = []
    s1s2_peaks = []
    
    if any(all_diffs):
        max_index= np.argmax(all_diffs)
        s2 = peaks[max_index]
        s1 = peaks[max_index+1]
        s1s2_peaks.append([s1, s2])
    else:
        max_index.append(-1)
        s1s2_peaks.append([-1,-1])
    s1s2_peaks = np.array(s1s2_peaks)
    
    #defining s1 and s2 boundaries    setting fixed bounding box
    s1_bounds = []
    s2_bounds = []
    if set_name == 'A':
        upper_s1 = 140#200*2
        lower_s1 = 140#80*2
        upper_s2 = 100#600*2
        lower_s2 = 100#70*2
    else:
        upper_s1 = 140#25*10 
        lower_s1 = 140#10*10 
        upper_s2 = 100#35*10 
        lower_s2 = 100#10*10 
        

    if s1s2_peaks[0,0] == -1:
        s1_bounds = [-1,-1]
        s2_bounds = [-1,-1]
    else:
        s1_lower = s1s2_peaks[0,0]-lower_s1
        s1_upper = s1s2_peaks[0,0]+upper_s1
        s2_lower = s1s2_peaks[0,1]-lower_s2
        s2_upper = s1s2_peaks[0,1]+upper_s2
        if s1_lower < 0:
            s1_lower = 0
        if s2_lower < 0:
            s2_lower = 0
        if s1_upper >= len(data):
            s1_upper = len(data) - 1
        if s2_upper >= len(data):
            s2_upper = len(data) - 1
        s1_bounds.append(s1_lower)
        s1_bounds.append(s1_upper)
        s2_bounds.append(s2_lower)
        s2_bounds.append(s2_upper)
        
    return s1_bounds, s2_bounds

"""
std deviation of specific interval where
lower is the left most bound of the interval, upper is right most bound
"""
def mean_stdInterval(s1_bounds, s2_bounds, data):
    std = []
    mean = []
    cout = 0
    if cout == 0:
        
        if s1_bounds[0] == -1:
            std.append(0)
            mean.append(0)
        else:
            dev = np.std(data[ s1_bounds[0]:s1_bounds[1] ] )
            avg = np.mean(data[ s1_bounds[0]:s1_bounds[1] ])
            if np.isnan(dev):
                std.append(0)
                mean.append(0)
            else:  
                std.append(dev)
                mean.append(avg)
        cout += 1
    
    if cout == 1:
        
        if s2_bounds[0] == -1:
            std.append(0)
            mean.append(0)
        else:
            dev = np.std(data[ s2_bounds[0]:s2_bounds[1] ] )
            avg = np.mean(data[ s2_bounds[0]:s2_bounds[1] ] )
            if np.isnan(dev):
                std.append(0)
                mean.append(0)
            else:  
                std.append(dev)
                mean.append(avg)
        
    return np.array(mean), np.array(std)


"""
Whole signal frequency energy  (mean and std)
"""
def freq_features(sample):
    freq_mean_std = []

    freq_components = np.abs( np.fft.rfft(sample) )
    freq_mean_std.append(np.mean(freq_components)) #mean
    freq_mean_std.append(np.std(freq_components)) #std
    
    return freq_mean_std


"""
Segment signal frequency energy  (mean and std)
"""
def freq_feature_segments(sample, s1_bounds, s2_bounds):

    cout = 0
    freq_seg_mean_std = []
    
    if cout == 0:
        
        if s1_bounds[0] == -1:
            freq_seg_mean_std.append(0)
            freq_seg_mean_std.append(0)
        else:
            freq_components = np.abs( np.fft.rfft( sample[ s1_bounds[0]: s1_bounds[1] ] ) )
            freq_seg_mean_std.append(np.mean(freq_components))
            freq_seg_mean_std.append(np.std(freq_components))            
        cout +=1
    
    if cout == 1:
        
        if s2_bounds[0] == -1:
            freq_seg_mean_std.append(0)
            freq_seg_mean_std.append(0)
        else:
            freq_components = np.abs( np.fft.rfft( sample[ s2_bounds[0]: s2_bounds[1] ] ) )
            freq_seg_mean_std.append(np.mean(freq_components))
            freq_seg_mean_std.append(np.std(freq_components))
    
    return freq_seg_mean_std

Config = Config()



"""
1. load dataset A and B, training and testing data separateley : 
    
    wav_A   wav_B  
    y_labelA    y_label_B  
    
    normal: 0
    murmur: 1
    extrahls : 2
    artifact : 3
"""
#1.load wave
#2.filter
#3.chunks
#4.Feature cooker

train_wav_A = []     # Save wav chunks file name
train_wav_B = []

y_train_labelA = []   # training labels
y_train_labelB = []

train_mfcc_A = []    # trainingg set features

train_mfcc_A_filter_1 = [] #mfcc
train_mfcc_A_filter_2 = []
train_mfcc_A_filter_3 = []
train_mfcc_A_filter_4 = []
train_mfcc_A_filter_5 = []
train_mfcc_A_filter_6 = []

train_zc_A = []    # zero crossing rate
train_mean_peak_interval_A = []       #segments mean
train_std_peak_interval_A = []        #segments std
train_freq_mean_std_A= []          # chunks frequency engergy mean and std
train_freq_seg_mean_std_A = []     # segment frequency energy mean and std
train_sample_entropy_A = []      # sample entropy
train_spectral_entropy_A = []   # spectral entropy

train_mfcc_B = []

train_mfcc_B_filter_1 = []
train_mfcc_B_filter_2 = []
train_mfcc_B_filter_3 = []
train_mfcc_B_filter_4 = []
train_mfcc_B_filter_5 = []
train_mfcc_B_filter_6 = []

train_zc_B = []
train_mean_peak_interval_B = []
train_std_peak_interval_B = []
train_freq_mean_std_B= []
train_freq_seg_mean_std_B = []
train_sample_entropy_B = []
train_spectral_entropy_B = []

test_wav_A = []     # Save wav chunks name
test_wav_B = []

test_mfcc_A = []    # test set features

test_mfcc_A_filter_1 = []
test_mfcc_A_filter_2 = []
test_mfcc_A_filter_3 = []
test_mfcc_A_filter_4 = []
test_mfcc_A_filter_5 = []
test_mfcc_A_filter_6 = []

test_zc_A = []
test_mean_peak_interval_A = []
test_std_peak_interval_A = []
test_freq_mean_std_A= []
test_freq_seg_mean_std_A = []
test_sample_entropy_A = []
test_spectral_entropy_A = []

test_mfcc_B = []

test_mfcc_B_filter_1 = []
test_mfcc_B_filter_2 = []
test_mfcc_B_filter_3 = []
test_mfcc_B_filter_4 = []
test_mfcc_B_filter_5 = []
test_mfcc_B_filter_6 = []

test_zc_B = []
test_mean_peak_interval_B = []
test_std_peak_interval_B = []
test_freq_mean_std_B= []
test_freq_seg_mean_std_B = []
test_sample_entropy_B = []
test_spectral_entropy_B = []

x_train_A = []  # merged fearures
x_train_B = []
x_test_A = []
x_test_B = []

# LOOPING VARIABLES
cout_1, cout_2= 0,0   # to count how many first_dir have been gone through    cout_2 may not be used
cout_wav = 0    # to count how many wav files have gone through in the loop of each first_dir
temp_signal = [] # for temporarily store the processed signal
cout_save = 1   #count how many signal chunks are split based on one wav file

for cout_1 in range(0,len(first_dir)):
 
    
##### Train_A  #########       
    if cout_1 <= 3:   
        
        current_wav_dir = source_dir + first_dir[cout_1]  #current directory
        wav_name_list = os.listdir(current_wav_dir)     #get all file name in the current direcory
        
        for cout_wav in tqdm(range(0,len(wav_name_list))):
            # load
            signal , _ = librosa.load(current_wav_dir + '\\' + wav_name_list[cout_wav], sr = Config.sr)
            # filting
            signal = butter_bandpass_filter(signal, Config.lowcut, Config.highcut, Config.sr, Config.order)
            # normalize [-1, 1 ]  (强化了在语谱图中低频的能量，稍微强化了弱心音)
            signal = ( signal / np.max( np.abs(signal) ) )
            
            # Padding zero to 3s if the signal is less than 3s
            if len(signal) <3* Config.sr:
                mask_less_3s = np.zeros(3*Config.sr, dtype = float)
                mask_less_3s[:len(signal)] = signal
                signal = mask_less_3s
            
            while len(signal) > 0:
                
                # one signal is cut into 3s non-overlapping chunks
                mask_3s = np.ones(len(signal), dtype=bool)
                if len(signal) >= 3* Config.sr:
                    temp_signal = signal[:3*Config.sr]
                    mask_3s[:int(3*Config.sr-1)] = False
                    signal = signal[mask_3s]
                    
                # Compute features
                    mfcc_temp = mfcc(temp_signal, Config.sr, numcep=Config.n_feat, 
                                     nfilt = Config.n_filt, nfft = Config.n_fft).T #mfcc
                    
                    zc_temp = stZCR(temp_signal)  # zero crossing rate
                    
                    find_peak_idx_temp = find_peaks(temp_signal.tolist(), 'A') #find peaks
                    s1_bounds_temp, s2_bounds_temp = get_S1S2_bounds(temp_signal, find_peak_idx_temp, 'A' ) # S1 S2 interval index
                    train_mean_peak_interval_temp, train_std_peak_interval_temp = mean_stdInterval(s1_bounds_temp, s2_bounds_temp, temp_signal) # std of signal interval
                    
                    train_freq_mean_std_temp = freq_features(temp_signal) # std of frequency energy
                    
                    train_freq_seg_mean_std_temp = freq_feature_segments(temp_signal, s1_bounds_temp, s2_bounds_temp) #std and mean of segment frequency energy
                    
                    train_sample_entropy_temp = sample_entropy(temp_signal)  #sample entropy
                    
                    train_spectral_entropy_temp = spectral_entropy(temp_signal, Config.sr, method = 'fft', nperseg = Config.n_fft, normalize = True) #spectral entropy
                    
                    if len(mfcc_temp[0]) >= Config.mfcc_len_threshold:
                        mfcc_post = np.array( [ np.average(mfcc_temp[0]), np.std(mfcc_temp[0]),
                                                np.average(mfcc_temp[1]), np.std(mfcc_temp[1]),
                                                np.average(mfcc_temp[2]), np.std(mfcc_temp[2]),
                                                np.average(mfcc_temp[3]), np.std(mfcc_temp[3]),
                                                np.average(mfcc_temp[4]), np.std(mfcc_temp[4]), 
                                                np.average(mfcc_temp[5]), np.std(mfcc_temp[5]),
                                                ] )     

                        train_mfcc_A_filter_1.append(mfcc_temp[0])
                        train_mfcc_A_filter_2.append(mfcc_temp[1])
                        train_mfcc_A_filter_3.append(mfcc_temp[2])
                        train_mfcc_A_filter_4.append(mfcc_temp[3])
                        train_mfcc_A_filter_5.append(mfcc_temp[4])
                        train_mfcc_A_filter_6.append(mfcc_temp[5])

                        train_mfcc_A.append(mfcc_post)
                        train_zc_A.append(zc_temp)
                        train_mean_peak_interval_A.append(train_mean_peak_interval_temp)
                        train_std_peak_interval_A.append(train_std_peak_interval_temp)
                        
                        train_freq_mean_std_A.append(train_freq_mean_std_temp)
                        train_freq_seg_mean_std_A.append(train_freq_seg_mean_std_temp)
                        train_sample_entropy_A.append(train_sample_entropy_temp)
                        train_spectral_entropy_A.append(train_spectral_entropy_temp)
                        
                        wav_index = wav_name_list[cout_wav][:-4]+ '_' + str(cout_save)
                        train_wav_A.append(wav_index)
                        y_train_labelA.append(cout_1)
                        cout_save += 1
                else:
                    temp_signal = []
                    cout_save = 1
                    cout_wav += 1
                    break






##### Train_B  #########
    if cout_1 > 3 and cout_1 <= 6:   #train_dataset_B
        
        current_wav_dir = source_dir + first_dir[cout_1]
        wav_name_list = os.listdir(current_wav_dir)
        
        for cout_wav in tqdm(range(0,len(wav_name_list))):
            # load
            signal , _ = librosa.load(current_wav_dir + '\\' + wav_name_list[cout_wav], sr = Config.sr)
            # filting
            signal = butter_bandpass_filter(signal, Config.lowcut, Config.highcut, Config.sr, Config.order)
            # Wavelet
            #signal = waveletSmooth(signal)
            # normalize [-1, 1 ]
            signal = ( signal / np.max( np.abs(signal) ) )
            
            # Padding zero to 3s
            if len(signal) <3* Config.sr:
                mask_less_3s = np.zeros(3*Config.sr, dtype = float)
                mask_less_3s[:len(signal)] = signal
                signal = mask_less_3s
            
            while len(signal) > 0:
                
                # cut 3s chunks non overlapping
                mask_3s = np.ones(len(signal), dtype=bool)
                if len(signal) >= 3* Config.sr:
                    temp_signal = signal[:3*Config.sr]
                    mask_3s[:int(3*Config.sr-1)] = False
                    signal = signal[mask_3s]
                    
                # 计算MFCCs 和 Zero_Crossing
                    mfcc_temp = mfcc(temp_signal, Config.sr, numcep=Config.n_feat, 
                                     nfilt = Config.n_filt, nfft = Config.n_fft).T
                    
                    zc_temp = stZCR(temp_signal)
                    
                    find_peak_idx_temp = find_peaks(temp_signal.tolist(), 'B')
                    s1_bounds_temp, s2_bounds_temp = get_S1S2_bounds(temp_signal, find_peak_idx_temp, 'B' )
                    train_mean_peak_interval_temp, train_std_peak_interval_temp = mean_stdInterval(s1_bounds_temp, s2_bounds_temp, temp_signal)
                    
                    train_freq_mean_std_temp = freq_features(temp_signal)
                    
                    train_freq_seg_mean_std_temp = freq_feature_segments(temp_signal, s1_bounds_temp, s2_bounds_temp)
                    
                    train_sample_entropy_temp = sample_entropy(temp_signal)
                    
                    train_spectral_entropy_temp = spectral_entropy(temp_signal, Config.sr, method = 'fft', nperseg = Config.n_fft, normalize = True)
                    
                    if len(mfcc_temp[0]) >= Config.mfcc_len_threshold:
                        mfcc_post = np.array( [ np.average(mfcc_temp[0]), np.std(mfcc_temp[0]),
                                                np.average(mfcc_temp[1]), np.std(mfcc_temp[1]),
                                                np.average(mfcc_temp[2]), np.std(mfcc_temp[2]),
                                                np.average(mfcc_temp[3]), np.std(mfcc_temp[3]),
                                                np.average(mfcc_temp[4]), np.std(mfcc_temp[4]), 
                                                np.average(mfcc_temp[5]), np.std(mfcc_temp[5]),
                                                ] )
                        
                        train_mfcc_B_filter_1.append(mfcc_temp[0])
                        train_mfcc_B_filter_2.append(mfcc_temp[1])
                        train_mfcc_B_filter_3.append(mfcc_temp[2])
                        train_mfcc_B_filter_4.append(mfcc_temp[3])
                        train_mfcc_B_filter_5.append(mfcc_temp[4])
                        train_mfcc_B_filter_6.append(mfcc_temp[5])

                        train_mfcc_B.append(mfcc_post)
                        train_zc_B.append(zc_temp)
                        train_mean_peak_interval_B.append(train_mean_peak_interval_temp)
                        train_std_peak_interval_B.append(train_std_peak_interval_temp)
                        train_freq_mean_std_B.append(train_freq_mean_std_temp)
                        train_freq_seg_mean_std_B.append(train_freq_seg_mean_std_temp)
                        train_sample_entropy_B.append(train_sample_entropy_temp)
                        train_spectral_entropy_B.append(train_spectral_entropy_temp)
                        
                        wav_index = wav_name_list[cout_wav][:-4]+ '_' + str(cout_save)
                        train_wav_B.append(wav_index)
                        y_train_labelB.append(cout_1-4)
                        cout_save += 1
                else:
                    temp_signal = []
                    cout_save = 1
                    cout_wav += 1
                    break
    

   
##### TestA  #########    
    if cout_1 > 6 and cout_1 <= 7:   #test_dataset_A
        
        current_wav_dir = source_dir + first_dir[cout_1]
        wav_name_list = os.listdir(current_wav_dir)
        
        for cout_wav in tqdm(range(0,len(wav_name_list))):
            # load
            signal , _ = librosa.load(current_wav_dir + '\\' + wav_name_list[cout_wav], sr = Config.sr)
            # filting
            signal = butter_bandpass_filter(signal, Config.lowcut, Config.highcut, Config.sr, Config.order)
            # normalize [-1, 1 ]
            signal = ( signal / np.max( np.abs(signal) ) )
            
            # Padding zero to 3s
            if len(signal) <3* Config.sr:
                mask_less_3s = np.zeros(3*Config.sr, dtype = float)
                mask_less_3s[:len(signal)] = signal
                signal = mask_less_3s
            
            while len(signal) > 0:
                
                # cut 3s chunks non overlapping
                mask_3s = np.ones(len(signal), dtype=bool)
                if len(signal) >= 3* Config.sr:
                    temp_signal = signal[:3*Config.sr]
                    mask_3s[:int(3*Config.sr-1)] = False
                    signal = signal[mask_3s]
                    
                # 计算MFCCs 和 Zero_Crossing
                    mfcc_temp = mfcc(temp_signal, Config.sr, numcep=Config.n_feat, 
                                     nfilt = Config.n_filt, nfft = Config.n_fft).T
                    
                    zc_temp = stZCR(temp_signal)
                    
                    find_peak_idx_temp = find_peaks(temp_signal.tolist(), 'A')
                    s1_bounds_temp, s2_bounds_temp = get_S1S2_bounds(temp_signal, find_peak_idx_temp, 'A' )
                    test_mean_peak_interal_temp, test_std_peak_interval_temp = mean_stdInterval(s1_bounds_temp, s2_bounds_temp, temp_signal)
                    
                    test_freq_mean_std_temp = freq_features(temp_signal)
                    
                    test_freq_seg_mean_std_temp = freq_feature_segments(temp_signal, s1_bounds_temp, s2_bounds_temp)
                    
                    test_sample_entropy_temp = sample_entropy(temp_signal)
                    
                    test_spectral_entropy_temp = spectral_entropy(temp_signal, Config.sr, method = 'fft', nperseg = Config.n_fft, normalize = True)
                    
                    if len(mfcc_temp[0]) >= Config.mfcc_len_threshold:
                        mfcc_post = np.array( [ np.average(mfcc_temp[0]), np.std(mfcc_temp[0]),
                                                np.average(mfcc_temp[1]), np.std(mfcc_temp[1]),
                                                np.average(mfcc_temp[2]), np.std(mfcc_temp[2]),
                                                np.average(mfcc_temp[3]), np.std(mfcc_temp[3]),
                                                np.average(mfcc_temp[4]), np.std(mfcc_temp[4]), 
                                                np.average(mfcc_temp[5]), np.std(mfcc_temp[5]),
                                                ] )    

                        test_mfcc_A_filter_1.append(mfcc_temp[0])
                        test_mfcc_A_filter_2.append(mfcc_temp[1])
                        test_mfcc_A_filter_3.append(mfcc_temp[2])
                        test_mfcc_A_filter_4.append(mfcc_temp[3])
                        test_mfcc_A_filter_5.append(mfcc_temp[4])
                        test_mfcc_A_filter_6.append(mfcc_temp[5])
                   
                        test_mfcc_A.append(mfcc_post)
                        test_zc_A.append(zc_temp)
                        test_mean_peak_interval_A.append(test_mean_peak_interal_temp)
                        test_std_peak_interval_A.append(test_std_peak_interval_temp)
                        test_freq_mean_std_A.append(test_freq_mean_std_temp)
                        test_freq_seg_mean_std_A.append(test_freq_seg_mean_std_temp)
                        test_sample_entropy_A.append(test_sample_entropy_temp)
                        test_spectral_entropy_A.append(test_spectral_entropy_temp)
                        
                        wav_index = wav_name_list[cout_wav][:-4]+ '_' + str(cout_save)
                        test_wav_A.append(wav_index)
                        cout_save += 1
                else:
                    temp_signal = []
                    cout_save = 1
                    cout_wav += 1
                    break                
    

    
##### TestB  #########    
    if cout_1 > 7 and cout_1 <= 8:   #Test_dataset_B
        
        current_wav_dir = source_dir + first_dir[cout_1]
        wav_name_list = os.listdir(current_wav_dir)
        
        for cout_wav in tqdm(range(0,len(wav_name_list))):
            # load
            signal , _ = librosa.load(current_wav_dir + '\\' + wav_name_list[cout_wav], sr = Config.sr)
            # filting
            signal = butter_bandpass_filter(signal, Config.lowcut, Config.highcut, Config.sr, Config.order)
            # Wavelet
            #signal = waveletSmooth(signal)
            # normalize [-1, 1 ]
            signal = ( signal / np.max( np.abs(signal) ) )
            
            # Padding zero to 3s
            if len(signal) <3* Config.sr:
                mask_less_3s = np.zeros(3*Config.sr, dtype = float)
                mask_less_3s[:len(signal)] = signal
                signal = mask_less_3s
            
            while len(signal) > 0:
                
                # cut 3s chunks non overlapping
                mask_3s = np.ones(len(signal), dtype=bool)
                if len(signal) >= 3* Config.sr:
                    temp_signal = signal[:3*Config.sr]
                    mask_3s[:int(3*Config.sr-1)] = False
                    signal = signal[mask_3s]
                    
                # 计算MFCCs 和 Zero_Crossing
                    mfcc_temp = mfcc(temp_signal, Config.sr, numcep=Config.n_feat, 
                                     nfilt = Config.n_filt, nfft = Config.n_fft).T
                    
                    zc_temp = stZCR(temp_signal)
                    
                    find_peak_idx_temp = find_peaks(temp_signal.tolist(), 'B')
                    s1_bounds_temp, s2_bounds_temp = get_S1S2_bounds(temp_signal, find_peak_idx_temp, 'B' )
                    test_mean_peak_interval_temp, test_std_peak_interval_temp = mean_stdInterval(s1_bounds_temp, s2_bounds_temp, temp_signal)
                    
                    test_freq_mean_std_temp = freq_features(temp_signal)
                    
                    test_freq_seg_mean_std_temp = freq_feature_segments(temp_signal, s1_bounds_temp, s2_bounds_temp)
                    
                    test_sample_entropy_temp = sample_entropy(temp_signal)
                    
                    test_spectral_entropy_temp = spectral_entropy(temp_signal, Config.sr, method = 'fft', nperseg = Config.n_fft, normalize = True)
                    
                    
                    if len(mfcc_temp[0]) >= Config.mfcc_len_threshold:
                        mfcc_post = np.array( [ np.average(mfcc_temp[0]), np.std(mfcc_temp[0]),
                                                np.average(mfcc_temp[1]), np.std(mfcc_temp[1]),
                                                np.average(mfcc_temp[2]), np.std(mfcc_temp[2]),
                                                np.average(mfcc_temp[3]), np.std(mfcc_temp[3]),
                                                np.average(mfcc_temp[4]), np.std(mfcc_temp[4]), 
                                                np.average(mfcc_temp[5]), np.std(mfcc_temp[5]),
                                                ] )
                        
                        test_mfcc_B_filter_1.append(mfcc_temp[0])
                        test_mfcc_B_filter_2.append(mfcc_temp[1])
                        test_mfcc_B_filter_3.append(mfcc_temp[2])
                        test_mfcc_B_filter_4.append(mfcc_temp[3])
                        test_mfcc_B_filter_5.append(mfcc_temp[4])
                        test_mfcc_B_filter_6.append(mfcc_temp[5])    
    
                        test_mfcc_B.append(mfcc_post)
                        test_zc_B.append(zc_temp)
                        test_mean_peak_interval_B.append(test_mean_peak_interval_temp)
                        test_std_peak_interval_B.append(test_std_peak_interval_temp)
                        test_freq_mean_std_B.append(test_freq_mean_std_temp)   
                        test_freq_seg_mean_std_B.append(test_freq_seg_mean_std_temp)
                        test_sample_entropy_B.append(test_sample_entropy_temp)
                        test_spectral_entropy_B.append(test_spectral_entropy_temp)
                        
                        wav_index = wav_name_list[cout_wav][:-4]+ '_' + str(cout_save)
                        test_wav_B.append(wav_index)
                        cout_save += 1
                else:
                    temp_signal = []
                    cout_save = 1
                    cout_wav += 1
                    break         


########Features to array train###############
train_mfcc_A = np.array(train_mfcc_A).reshape(-1,12)
train_mfcc_B = np.array(train_mfcc_B).reshape(-1,12)

train_zc_A = np.array(train_zc_A).reshape(-1,1)
train_zc_B = np.array(train_zc_B).reshape(-1,1)

train_mean_peak_interval_A = np.array(train_mean_peak_interval_A).reshape(-1,2)
train_mean_peak_interval_B = np.array(train_mean_peak_interval_B).reshape(-1,2)

train_std_peak_interval_A = np.array(train_std_peak_interval_A).reshape(-1,2)
train_std_peak_interval_B = np.array(train_std_peak_interval_B).reshape(-1,2)

train_freq_mean_std_A = np.array(train_freq_mean_std_A).reshape(-1,2)
train_freq_mean_std_B = np.array(train_freq_mean_std_B).reshape(-1,2)

train_freq_seg_mean_std_A = np.array(train_freq_seg_mean_std_A).reshape(-1,4)
train_freq_seg_mean_std_B = np.array(train_freq_seg_mean_std_B).reshape(-1,4)

train_sample_entropy_A = np.array(train_sample_entropy_A).reshape(-1,1)
train_sample_entropy_B = np.array(train_sample_entropy_B).reshape(-1,1)

train_spectral_entropy_A = np.array(train_spectral_entropy_A).reshape(-1,1)
train_spectral_entropy_B = np.array(train_spectral_entropy_B).reshape(-1,1)
where_are_nan = np.isnan(train_spectral_entropy_A)
train_spectral_entropy_A[where_are_nan] = np.mean(train_spectral_entropy_A[0:20])#0.87
where_are_nan = np.isnan(train_spectral_entropy_B)
train_spectral_entropy_B[where_are_nan] = np.mean(train_spectral_entropy_B[0:26])#0.87


#######Features to array test##################
test_mfcc_A = np.array(test_mfcc_A).reshape(-1,12)
test_mfcc_B = np.array(test_mfcc_B).reshape(-1,12)

test_zc_A = np.array(test_zc_A).reshape(-1,1)
test_zc_B = np.array(test_zc_B).reshape(-1,1)

test_mean_peak_interval_A = np.array(test_mean_peak_interval_A).reshape(-1,2)
test_mean_peak_interval_B = np.array(test_mean_peak_interval_B).reshape(-1,2)

test_std_peak_interval_A = np.array(test_std_peak_interval_A).reshape(-1,2)
test_std_peak_interval_B = np.array(test_std_peak_interval_B).reshape(-1,2)

test_freq_mean_std_A = np.array(test_freq_mean_std_A).reshape(-1,2)
test_freq_mean_std_B = np.array(test_freq_mean_std_B).reshape(-1,2)

test_freq_seg_mean_std_A = np.array(test_freq_seg_mean_std_A).reshape(-1,4)
test_freq_seg_mean_std_B = np.array(test_freq_seg_mean_std_B).reshape(-1,4)

test_sample_entropy_A = np.array(test_sample_entropy_A).reshape(-1,1)
test_sample_entropy_B = np.array(test_sample_entropy_B).reshape(-1,1)

test_spectral_entropy_A = np.array(test_spectral_entropy_A).reshape(-1,1)
test_spectral_entropy_B = np.array(test_spectral_entropy_B).reshape(-1,1)
where_are_nan = np.isnan(test_spectral_entropy_A)
test_spectral_entropy_A[where_are_nan] = np.mean(test_spectral_entropy_A[0:24])#0.87
where_are_nan = np.isnan(test_spectral_entropy_B)
test_spectral_entropy_B[where_are_nan] = np.mean(test_spectral_entropy_A[0:26])#0.87

######## PCA MFCCS ###########

train_mfcc_A_filter_1 = PCA(n_components=2).fit_transform(train_mfcc_A_filter_1)
train_mfcc_A_filter_2 = PCA(n_components=2).fit_transform(train_mfcc_A_filter_2)
train_mfcc_A_filter_3 = PCA(n_components=2).fit_transform(train_mfcc_A_filter_3)
train_mfcc_A_filter_4 = PCA(n_components=2).fit_transform(train_mfcc_A_filter_4)
train_mfcc_A_filter_5 = PCA(n_components=2).fit_transform(train_mfcc_A_filter_5)
train_mfcc_A_filter_6 = PCA(n_components=2).fit_transform(train_mfcc_A_filter_6)

train_mfcc_B_filter_1 = PCA(n_components=2).fit_transform(train_mfcc_B_filter_1)
train_mfcc_B_filter_2 = PCA(n_components=2).fit_transform(train_mfcc_B_filter_2)
train_mfcc_B_filter_3 = PCA(n_components=2).fit_transform(train_mfcc_B_filter_3)
train_mfcc_B_filter_4 = PCA(n_components=2).fit_transform(train_mfcc_B_filter_4)
train_mfcc_B_filter_5 = PCA(n_components=2).fit_transform(train_mfcc_B_filter_5)
train_mfcc_B_filter_6 = PCA(n_components=2).fit_transform(train_mfcc_B_filter_6)

test_mfcc_A_filter_1 = PCA(n_components=2).fit_transform(test_mfcc_A_filter_1)
test_mfcc_A_filter_2 = PCA(n_components=2).fit_transform(test_mfcc_A_filter_2)
test_mfcc_A_filter_3 = PCA(n_components=2).fit_transform(test_mfcc_A_filter_3)
test_mfcc_A_filter_4 = PCA(n_components=2).fit_transform(test_mfcc_A_filter_4)
test_mfcc_A_filter_5 = PCA(n_components=2).fit_transform(test_mfcc_A_filter_5)
test_mfcc_A_filter_6 = PCA(n_components=2).fit_transform(test_mfcc_A_filter_6)

test_mfcc_B_filter_1 = PCA(n_components=2).fit_transform(test_mfcc_B_filter_1)
test_mfcc_B_filter_2 = PCA(n_components=2).fit_transform(test_mfcc_B_filter_2)
test_mfcc_B_filter_3 = PCA(n_components=2).fit_transform(test_mfcc_B_filter_3)
test_mfcc_B_filter_4 = PCA(n_components=2).fit_transform(test_mfcc_B_filter_4)
test_mfcc_B_filter_5 = PCA(n_components=2).fit_transform(test_mfcc_B_filter_5)
test_mfcc_B_filter_6 = PCA(n_components=2).fit_transform(test_mfcc_B_filter_6)

########Feature merging########
x_train_A = np.hstack((train_zc_A, train_freq_mean_std_A ,
                       train_freq_seg_mean_std_A,
                       train_sample_entropy_A, train_spectral_entropy_A,
                       train_mean_peak_interval_A,train_std_peak_interval_A, #train_mfcc_A
                       train_mfcc_A_filter_1, train_mfcc_A_filter_2, train_mfcc_A_filter_3,
                       train_mfcc_A_filter_4, train_mfcc_A_filter_5, train_mfcc_A_filter_6))

x_train_B = np.hstack((train_zc_B, train_freq_mean_std_B ,
                       train_freq_seg_mean_std_B,
                       train_sample_entropy_B, train_spectral_entropy_B,
                       train_mean_peak_interval_B,train_std_peak_interval_B, #train_mfcc_B
                       train_mfcc_B_filter_1, train_mfcc_B_filter_2, train_mfcc_B_filter_3,
                       train_mfcc_B_filter_4, train_mfcc_B_filter_5, train_mfcc_B_filter_6))

x_test_A = np.hstack((test_zc_A, test_freq_mean_std_A ,
                      test_freq_seg_mean_std_A,
                      test_sample_entropy_A, test_spectral_entropy_A,
                      test_mean_peak_interval_A, test_std_peak_interval_A, #test_mfcc_A
                      test_mfcc_A_filter_1, test_mfcc_A_filter_2, test_mfcc_A_filter_3,
                      test_mfcc_A_filter_4, test_mfcc_A_filter_5, test_mfcc_A_filter_6))

x_test_B = np.hstack((test_zc_B, test_freq_mean_std_B ,
                      test_freq_seg_mean_std_B,
                      test_sample_entropy_B, test_spectral_entropy_B,
                      test_mean_peak_interval_B, test_std_peak_interval_B, #test_mfcc_B,
                      test_mfcc_B_filter_1, test_mfcc_B_filter_2, test_mfcc_B_filter_3,
                      test_mfcc_B_filter_4, test_mfcc_B_filter_5, test_mfcc_B_filter_6))


########Normalize Feature###############
for i in range(0:np.size(x_train_A,1))
    x_train_A = x_train_A[:,i]/ np.max(np.abs(x_train_A[:,i]))
for i in range(0:np.size(x_train_B,1))
    x_train_B = x_train_B[:,i]/ np.max(np.abs(x_train_B[:,i]))
for i in range(0:np.size(x_test_A,1))
    x_test_A = x_test_A[:,i]/ np.max(np.abs(x_test_A[:,i]))
for i in range(0:np.size(x_test_B,1))
    x_test_B = x_test_B[:,i]/ np.max(np.abs(x_test_AB[:,i]))
    
    #x_train_A = normalize(x_train_A, axis=0, norm='max')
    #x_train_B = normalize(x_train_B, axis=0, norm='max')
    #x_test_A = normalize(x_test_A, axis=0, norm='max')
    #x_test_B = normalize(x_test_B, axis=0, norm='max')


########Save##############
pickle.dump(x_train_A,open(r'Saved_params\x_train_A.txt', 'wb') )                     
pickle.dump(x_train_B,open(r'Saved_params\x_train_B.txt', 'wb') ) 

pickle.dump(x_test_A, open(r'Saved_params\x_test_A.txt', 'wb') )        
pickle.dump(x_test_B, open(r'Saved_params\x_test_B.txt', 'wb') )  

########Save trained sample features and labels for plot analysis##############
pickle.dump(train_mfcc_A,open(r'Saved_params\train_mfcc_A.txt', 'wb') )
pickle.dump(train_mfcc_B,open(r'Saved_params\train_mfcc_B.txt', 'wb') )

pickle.dump(train_mfcc_A_filter_1,open(r'Saved_params\train_mfcc_A_filter_1.txt', 'wb') )
pickle.dump(train_mfcc_A_filter_2,open(r'Saved_params\train_mfcc_A_filter_2.txt', 'wb') )
pickle.dump(train_mfcc_A_filter_3,open(r'Saved_params\train_mfcc_A_filter_3.txt', 'wb') )
pickle.dump(train_mfcc_A_filter_4,open(r'Saved_params\train_mfcc_A_filter_4.txt', 'wb') )
pickle.dump(train_mfcc_A_filter_5,open(r'Saved_params\train_mfcc_A_filter_5.txt', 'wb') )
pickle.dump(train_mfcc_A_filter_6,open(r'Saved_params\train_mfcc_A_filter_6.txt', 'wb') )

pickle.dump(train_mfcc_B_filter_1,open(r'Saved_params\train_mfcc_B_filter_1.txt', 'wb') )
pickle.dump(train_mfcc_B_filter_2,open(r'Saved_params\train_mfcc_B_filter_2.txt', 'wb') )
pickle.dump(train_mfcc_B_filter_3,open(r'Saved_params\train_mfcc_B_filter_3.txt', 'wb') )
pickle.dump(train_mfcc_B_filter_4,open(r'Saved_params\train_mfcc_B_filter_4.txt', 'wb') )
pickle.dump(train_mfcc_B_filter_5,open(r'Saved_params\train_mfcc_B_filter_5.txt', 'wb') )
pickle.dump(train_mfcc_B_filter_6,open(r'Saved_params\train_mfcc_B_filter_6.txt', 'wb') )

pickle.dump(train_zc_A,open(r'Saved_params\train_zc_A.txt', 'wb') )                     
pickle.dump(train_zc_B,open(r'Saved_params\train_zc_B.txt', 'wb') )                 
        
pickle.dump(train_mean_peak_interval_A,open(r'Saved_params\train_mean_peak_interval_A.txt', 'wb') )                     
pickle.dump(train_mean_peak_interval_B,open(r'Saved_params\train_mean_peak_interval_B.txt', 'wb') )  
        
pickle.dump(train_std_peak_interval_A,open(r'Saved_params\train_std_peak_interval_A.txt', 'wb') )                     
pickle.dump(train_std_peak_interval_B,open(r'Saved_params\train_std_peak_interval_B.txt', 'wb') )                 

pickle.dump(train_freq_mean_std_A,open(r'Saved_params\train_freq_mean_std_A.txt', 'wb') )                     
pickle.dump(train_freq_mean_std_B,open(r'Saved_params\train_freq_mean_std_B.txt', 'wb') )    

pickle.dump(train_freq_seg_mean_std_A,open(r'Saved_params\train_freq_seg_mean_std_A.txt', 'wb') )                     
pickle.dump(train_freq_seg_mean_std_B,open(r'Saved_params\train_freq_seg_mean_std_B.txt', 'wb') )  

pickle.dump(train_sample_entropy_A,open(r'Saved_params\train_sample_entropy_A.txt', 'wb') )                     
pickle.dump(train_sample_entropy_B,open(r'Saved_params\train_sample_entropy_B.txt', 'wb') )  

pickle.dump(train_spectral_entropy_A,open(r'Saved_params\train_spectral_entropy_A.txt', 'wb') )                     
pickle.dump(train_spectral_entropy_B,open(r'Saved_params\train_spectral_entropy_B.txt', 'wb') ) 

pickle.dump(y_train_labelA, open(r'Saved_params\y_train_labelA.txt', 'wb') )        
pickle.dump(y_train_labelB, open(r'Saved_params\y_train_labelB.txt', 'wb') )      
  
pickle.dump(test_wav_A, open(r'Saved_params\test_wav_A.txt', 'wb') )        
pickle.dump(test_wav_B, open(r'Saved_params\test_wav_B.txt', 'wb') )  
