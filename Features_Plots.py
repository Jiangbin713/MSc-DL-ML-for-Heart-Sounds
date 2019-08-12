# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:31:40 2019

@author: jiang

What has been done in this file?

    1.  Plot the features
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


"""
load features
"""
train_mfcc_A = pickle.load(open(r'Saved_params\train_mfcc_A.txt', 'rb') )   #mfcc mean and std of each filter
train_mfcc_B = pickle.load(open(r'Saved_params\train_mfcc_B.txt', 'rb') )

train_mfcc_A_filter_1 = pickle.load(open(r'Saved_params\train_mfcc_A_filter_1.txt', 'rb') )  # mfcc
train_mfcc_A_filter_2 = pickle.load(open(r'Saved_params\train_mfcc_A_filter_2.txt', 'rb') )
train_mfcc_A_filter_3 = pickle.load(open(r'Saved_params\train_mfcc_A_filter_3.txt', 'rb') )
train_mfcc_A_filter_4 = pickle.load(open(r'Saved_params\train_mfcc_A_filter_4.txt', 'rb') )
train_mfcc_A_filter_5 = pickle.load(open(r'Saved_params\train_mfcc_A_filter_5.txt', 'rb') )
train_mfcc_A_filter_6 = pickle.load(open(r'Saved_params\train_mfcc_A_filter_6.txt', 'rb') )

train_mfcc_B_filter_1 = pickle.load(open(r'Saved_params\train_mfcc_B_filter_1.txt', 'rb') )
train_mfcc_B_filter_2 = pickle.load(open(r'Saved_params\train_mfcc_B_filter_2.txt', 'rb') )
train_mfcc_B_filter_3 = pickle.load(open(r'Saved_params\train_mfcc_B_filter_3.txt', 'rb') )
train_mfcc_B_filter_4 = pickle.load(open(r'Saved_params\train_mfcc_B_filter_4.txt', 'rb') )
train_mfcc_B_filter_5 = pickle.load(open(r'Saved_params\train_mfcc_B_filter_5.txt', 'rb') )
train_mfcc_B_filter_6 = pickle.load(open(r'Saved_params\train_mfcc_B_filter_6.txt', 'rb') )

train_zc_A = pickle.load(open(r'Saved_params\train_zc_A.txt', 'rb') ) # zero crossing rate                
train_zc_B = pickle.load(open(r'Saved_params\train_zc_B.txt', 'rb') )                 

train_mean_peak_interval_A = pickle.load(open(r'Saved_params\train_mean_peak_interval_A.txt', 'rb') )  # mean of interval           
train_mean_peak_interval_B = pickle.load(open(r'Saved_params\train_mean_peak_interval_B.txt', 'rb') ) 
               
train_std_peak_interval_A = pickle.load(open(r'Saved_params\train_std_peak_interval_A.txt', 'rb') )  # std of interval           
train_std_peak_interval_B = pickle.load(open(r'Saved_params\train_std_peak_interval_B.txt', 'rb') )                 

train_freq_mean_std_A = pickle.load(open(r'Saved_params\train_freq_mean_std_A.txt', 'rb') )  # mean and std of frequency energy   
train_freq_mean_std_B = pickle.load(open(r'Saved_params\train_freq_mean_std_B.txt', 'rb') )    

train_freq_seg_mean_std_A = pickle.load(open(r'Saved_params\train_freq_seg_mean_std_A.txt', 'rb') )  #mean and std of segment frequency engergy               
train_freq_seg_mean_std_B = pickle.load(open(r'Saved_params\train_freq_seg_mean_std_B.txt', 'rb') )  

train_sample_entropy_A = pickle.load(open(r'Saved_params\train_sample_entropy_A.txt', 'rb') ) #sample entropy
train_sample_entropy_B = pickle.load(open(r'Saved_params\train_sample_entropy_B.txt', 'rb') )  

train_spectral_entropy_A = pickle.load(open(r'Saved_params\train_spectral_entropy_A.txt', 'rb') )  #spectral entropy                   
train_spectral_entropy_B = pickle.load(open(r'Saved_params\train_spectral_entropy_B.txt', 'rb') )  


"""
load labels
"""

y_train_labelA = pickle.load(open(r'Saved_params\y_train_labelA.txt', 'rb') )  #label
y_train_labelB = pickle.load(open(r'Saved_params\y_train_labelB.txt', 'rb') )

x_A_1D = np.arange(0,len(y_train_labelA))  # x axis index Dataset A
x_B_1D = np.arange(0,len(y_train_labelB))  # x axis index Dataset B



"""
Plot features
"""

#Zero crossing
train_zc_A = normalize(train_zc_A, axis=0, norm='max') # normalization
train_zc_B = normalize(train_zc_B, axis=0, norm='max' )

#A
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('Zero crossing rate')

ax[0].scatter(x_A_1D[0:67], train_zc_A[0:67], label = 'normal_A', cmap = 'r', marker = 'o'  )
ax[0].scatter(x_A_1D[68:134], train_zc_A[68:134], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0].scatter(x_A_1D[135:174], train_zc_A[135:174], label ='extrahls_A', cmap = 'k', marker = 's' )
ax[0].scatter(x_A_1D[175:294], train_zc_A[175:294], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0].legend()
ax[0].set_title('Zero crossing rate - Dataset A')

#B
ax[1].scatter(x_B_1D[0:589], train_zc_B[0:589], label = 'normal_B', cmap = 'r', marker = 'o'  )
ax[1].scatter(x_B_1D[590:799], train_zc_B[590:799], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1].scatter(x_B_1D[800:872], train_zc_B[800:872], label ='extrahls_B', cmap = 'k', marker = 's' )
ax[1].legend()
ax[1].set_title('Zero crossing rate - Dataset B')
#######################################################

#mean of S1 S2 interval
train_mean_peak_interval_A = normalize(train_mean_peak_interval_A, axis=0, norm= 'max')
train_mean_peak_interval_B = normalize(train_mean_peak_interval_B, axis=0, norm= 'max')

#A
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('mean of S1 and S2 - Dataset A')

ax[0].scatter(x_A_1D[0:67], train_mean_peak_interval_A[0:67, 0], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0].scatter(x_A_1D[68:134], train_mean_peak_interval_A[68:134, 0], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0].scatter(x_A_1D[135:174], train_mean_peak_interval_A[135:174, 0], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0].scatter(x_A_1D[175:294], train_mean_peak_interval_A[175:294, 0], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0].legend()
ax[0].set_title('S1 segment mean - Dataset A')

ax[1].scatter(x_A_1D[0:67], train_mean_peak_interval_A[0:67, 1], label = 'normal_A', cmap = 'r', marker = 'o'  )
ax[1].scatter(x_A_1D[68:134], train_mean_peak_interval_A[68:134, 1], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1].scatter(x_A_1D[135:174], train_mean_peak_interval_A[135:174, 1], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[1].scatter(x_A_1D[175:294], train_mean_peak_interval_A[175:294, 1], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1].legend()
ax[1].set_title('S2 segment mean - Dataset A')

#B
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('mean of S1 and S2 - Dataset B')

ax[0].scatter(x_B_1D[0:589], train_mean_peak_interval_B[0:589, 0], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[0].scatter(x_B_1D[590:799], train_mean_peak_interval_B[590:799, 0], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0].scatter(x_B_1D[800:872], train_mean_peak_interval_B[800:872, 0], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0].legend()
ax[0].set_title('S1 segment mean - Dataset B')

ax[1].scatter(x_B_1D[0:589], train_mean_peak_interval_B[0:589, 1], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[1].scatter(x_B_1D[590:799], train_mean_peak_interval_B[590:799, 1], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1].scatter(x_B_1D[800:872], train_mean_peak_interval_B[800:872, 1], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1].legend()
ax[1].set_title('S2 segment mean - Dataset B')
#######################################################

#std of S1 S2 interval
train_std_peak_interval_A = normalize(train_std_peak_interval_A, axis=0, norm= 'max')
train_std_peak_interval_B = normalize(train_std_peak_interval_B, axis=0, norm= 'max')

#A
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('Std of S1 and S2 - Dataset A')

ax[0].scatter(x_A_1D[0:67], train_std_peak_interval_A[0:67, 0], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0].scatter(x_A_1D[68:134], train_std_peak_interval_A[68:134, 0], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0].scatter(x_A_1D[135:174], train_std_peak_interval_A[135:174, 0], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0].scatter(x_A_1D[175:294], train_std_peak_interval_A[175:294, 0], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0].legend()
ax[0].set_title('S1 segment std - Dataset A')

ax[1].scatter(x_A_1D[0:67], train_std_peak_interval_A[0:67, 1], label = 'normal_A', cmap = 'r'  , marker = 'o')
ax[1].scatter(x_A_1D[68:134], train_std_peak_interval_A[68:134, 1], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1].scatter(x_A_1D[135:174], train_std_peak_interval_A[135:174, 1], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[1].scatter(x_A_1D[175:294], train_std_peak_interval_A[175:294, 1], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1].legend()
ax[1].set_title('S2 segment std - Dataset A')

#B
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('Std of S1 and S2 - Dataset B')

ax[0].scatter(x_B_1D[0:589], train_std_peak_interval_B[0:589, 0], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[0].scatter(x_B_1D[590:799], train_std_peak_interval_B[590:799, 0], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0].scatter(x_B_1D[800:872], train_std_peak_interval_B[800:872, 0], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0].legend()
ax[0].set_title('S1 segment std - Dataset B')

ax[1].scatter(x_B_1D[0:589], train_std_peak_interval_B[0:589, 1], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[1].scatter(x_B_1D[590:799], train_std_peak_interval_B[590:799, 1], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1].scatter(x_B_1D[800:872], train_std_peak_interval_B[800:872, 1], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1].legend()
ax[1].set_title('S2 segment std - Dataset B')
###########################################################

#Sample entropy
train_sample_entropy_A = normalize(train_sample_entropy_A, axis=0, norm='max')# normalization
train_sample_entropy_B = normalize(train_sample_entropy_B, axis=0, norm='max')

#A
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('Sample entropy')

ax[0].scatter(x_A_1D[0:67], train_sample_entropy_A[0:67], label = 'normal_A', cmap = 'r', marker = 'o'  )
ax[0].scatter(x_A_1D[68:134], train_sample_entropy_A[68:134], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0].scatter(x_A_1D[135:174], train_sample_entropy_A[135:174], label ='extrahls_A', cmap = 'k', marker = 's' )
ax[0].scatter(x_A_1D[175:294], train_sample_entropy_A[175:294], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0].legend()
ax[0].set_title('Sample entropy - Dataset A')

#B
ax[1].scatter(x_B_1D[0:589], train_sample_entropy_B[0:589], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[1].scatter(x_B_1D[590:799], train_sample_entropy_B[590:799], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1].scatter(x_B_1D[800:872], train_sample_entropy_B[800:872], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1].legend()
ax[1].set_title('Sample entropy - Dataset B')
###########################################################

#Spectral entropy
train_spectral_entropy_A = normalize(train_spectral_entropy_A, axis=0, norm='max')# normalization
train_spectral_entropy_B = normalize(train_spectral_entropy_B, axis=0, norm='max')

#A
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('Spectral entropy')

ax[0].scatter(x_A_1D[0:67], train_spectral_entropy_A[0:67], label = 'normal_A', cmap = 'r'  , marker = 'o')
ax[0].scatter(x_A_1D[68:134], train_spectral_entropy_A[68:134], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0].scatter(x_A_1D[135:174], train_spectral_entropy_A[135:174], label ='extrahls_A', cmap = 'k', marker = 's' )
ax[0].scatter(x_A_1D[175:294], train_spectral_entropy_A[175:294], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0].legend()
ax[0].set_title('Spectral entropy - Dataset A')

#B
ax[1].scatter(x_B_1D[0:589], train_spectral_entropy_B[0:589], label = 'normal_B', cmap = 'r', marker = 'o'  )
ax[1].scatter(x_B_1D[590:799], train_spectral_entropy_B[590:799], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1].scatter(x_B_1D[800:872], train_spectral_entropy_B[800:872], label ='extrahls_B', cmap = 'k', marker = 's' )
ax[1].legend()
ax[1].set_title('Spectral entropy - Dataset B')
###########################################################

#Whole wav file std of frequency engery
train_freq_mean_std_A = normalize(train_freq_mean_std_A, axis=0, norm='max')# normalization
train_freq_mean_std_B = normalize(train_freq_mean_std_B, axis=0, norm='max')
#A
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('Mean of frequency amplitude')

ax[0].scatter(x_A_1D[0:67], train_freq_mean_std_A[0:67, 0], label = 'normal_A', cmap = 'r'  , marker = 'o')
ax[0].scatter(x_A_1D[68:134], train_freq_mean_std_A[68:134, 0], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0].scatter(x_A_1D[135:174], train_freq_mean_std_A[135:174, 0], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0].scatter(x_A_1D[175:294], train_freq_mean_std_A[175:294, 0], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0].legend()
ax[0].set_title('Mean of frequency amplitude - Dataset A')

ax[1].scatter(x_B_1D[0:589], train_freq_mean_std_B[0:589, 0], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[1].scatter(x_B_1D[590:799], train_freq_mean_std_B[590:799, 0], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1].scatter(x_B_1D[800:872], train_freq_mean_std_B[800:872, 0], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1].legend()
ax[1].set_title('Mean of frequency amplitude - Dataset B')



#B
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
fig.suptitle('Std of frequency amplitude')

ax[0].scatter(x_A_1D[0:67], train_freq_mean_std_A[0:67, 1], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0].scatter(x_A_1D[68:134], train_freq_mean_std_A[68:134, 1], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0].scatter(x_A_1D[135:174], train_freq_mean_std_A[135:174, 1], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0].scatter(x_A_1D[175:294], train_freq_mean_std_A[175:294, 1], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0].legend()
ax[0].set_title('Std of frequency amplitude - Dataset A')

ax[1].scatter(x_B_1D[0:589], train_freq_mean_std_B[0:589, 1], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[1].scatter(x_B_1D[590:799], train_freq_mean_std_B[590:799, 1], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1].scatter(x_B_1D[800:872], train_freq_mean_std_B[800:872, 1], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1].legend()
ax[1].set_title('Std of frequency amplitude - Dataset B')

###########################################################

#Segmentation mean, std of frequency energy
train_freq_seg_mean_std_A = normalize(train_freq_seg_mean_std_A, axis=0, norm='max')# normalization
train_freq_seg_mean_std_B = normalize(train_freq_seg_mean_std_B, axis=0, norm='max')

#A
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
fig.suptitle('Mean and std of S1 and S2 frequency amplitude  - Dataset A')

ax[0,0].scatter(x_A_1D[0:67], train_freq_seg_mean_std_A[0:67, 0], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0,0].scatter(x_A_1D[68:134], train_freq_seg_mean_std_A[68:134, 0], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0,0].scatter(x_A_1D[135:174], train_freq_seg_mean_std_A[135:174, 0], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0,0].scatter(x_A_1D[175:294], train_freq_seg_mean_std_A[175:294, 0], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0,0].legend()
ax[0,0].set_title('Mean of S1 frequency amplitude  - Dataset A')

ax[0,1].scatter(x_A_1D[0:67], train_freq_seg_mean_std_A[0:67, 1], label = 'normal_A', cmap = 'r'  , marker = 'o')
ax[0,1].scatter(x_A_1D[68:134], train_freq_seg_mean_std_A[68:134, 1], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0,1].scatter(x_A_1D[135:174], train_freq_seg_mean_std_A[135:174, 1], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0,1].scatter(x_A_1D[175:294], train_freq_seg_mean_std_A[175:294, 1], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0,1].legend()
ax[0,1].set_title('Std of S1 frequency amplitude  - Dataset A')

ax[1,0].scatter(x_A_1D[0:67], train_freq_seg_mean_std_A[0:67, 2], label = 'normal_A', cmap = 'r', marker = 'o'  )
ax[1,0].scatter(x_A_1D[68:134], train_freq_seg_mean_std_A[68:134, 2], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1,0].scatter(x_A_1D[135:174], train_freq_seg_mean_std_A[135:174, 2], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[1,0].scatter(x_A_1D[175:294], train_freq_seg_mean_std_A[175:294, 2], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1,0].legend()
ax[1,0].set_title('Mean of S2 frequency amplitude  - Dataset A')

ax[1,1].scatter(x_A_1D[0:67], train_freq_seg_mean_std_A[0:67, 3], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[1,1].scatter(x_A_1D[68:134], train_freq_seg_mean_std_A[68:134, 3], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1,1].scatter(x_A_1D[135:174], train_freq_seg_mean_std_A[135:174, 3], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[1,1].scatter(x_A_1D[175:294], train_freq_seg_mean_std_A[175:294, 3], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1,1].legend()
ax[1,1].set_title('Std of S1 frequency amplitude  - Dataset A')
#B
fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
fig.suptitle('Mean and std of S1 and S2 frequency amplitude  - Dataset B')

ax[0,0].scatter(x_B_1D[0:589], train_freq_seg_mean_std_B[0:589, 0], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[0,0].scatter(x_B_1D[590:799], train_freq_seg_mean_std_B[590:799, 0], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0,0].scatter(x_B_1D[800:872], train_freq_seg_mean_std_B[800:872, 0], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0,0].legend()
ax[0,0].set_title('Mean of S1 frequency amplitude  - Dataset B')

ax[0,1].scatter(x_B_1D[0:589], train_freq_seg_mean_std_B[0:589, 1], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[0,1].scatter(x_B_1D[590:799], train_freq_seg_mean_std_B[590:799, 1], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0,1].scatter(x_B_1D[800:872], train_freq_seg_mean_std_B[800:872, 1], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0,1].legend()
ax[0,1].set_title('Std of S1 frequency amplitude  - Dataset B')

ax[1,0].scatter(x_B_1D[0:589], train_freq_seg_mean_std_B[0:589, 2], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[1,0].scatter(x_B_1D[590:799], train_freq_seg_mean_std_B[590:799, 2], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1,0].scatter(x_B_1D[800:872], train_freq_seg_mean_std_B[800:872, 2], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1,0].legend()
ax[1,0].set_title('Mean of S2 frequency amplitude  - Dataset B')

ax[1,1].scatter(x_B_1D[0:589], train_freq_seg_mean_std_B[0:589, 3], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[1,1].scatter(x_B_1D[590:799], train_freq_seg_mean_std_B[590:799, 3], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1,1].scatter(x_B_1D[800:872], train_freq_seg_mean_std_B[800:872, 3], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1,1].legend()
ax[1,1].set_title('Std of S1 frequency amplitude - Dataset B')

###########################################################

#mfccs mean,std of each filter
train_mfcc_A = normalize(train_mfcc_A, axis=0, norm='max')# normalization
train_mfcc_B = normalize(train_mfcc_B, axis=0, norm='max')

#A
fig,ax = plt.subplots(nrows=3, ncols=4, figsize=(20,15))
fig.suptitle('Mean and std MFCCs of each filter - Dataset A')

ax[0,0].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 0], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0,0].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 0], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0,0].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 0], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0,0].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 0], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0,0].legend()
ax[0,0].set_title('Mean of the first filter MFCCs - Dataset A')

ax[0,1].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 1], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0,1].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 1], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0,1].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 1], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0,1].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 1], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0,1].legend()
ax[0,1].set_title('Std of the firsh filter MFCCs - Dataset A')

ax[0,2].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 2], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0,2].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 2], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0,2].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 2], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[0,2].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 2], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0,2].legend()
ax[0,2].set_title('Mean of the second filter MFCCs - Dataset A')

ax[0,3].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 3], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[0,3].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 3], label = 'murmur_A', cmap = 'b', marker = '^')
ax[0,3].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 3], label ='extrahls_A', cmap = 'k', marker = 's' )
ax[0,3].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 3], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[0,3].legend()
ax[0,3].set_title('Std of the second filter MFCCs - Dataset A')

ax[1,0].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 4], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[1,0].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 4], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1,0].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 4], label ='extrahls_A', cmap = 'k', marker = 's' )
ax[1,0].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 4], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1,0].legend()
ax[1,0].set_title('Mean of the third filter MFCCs - Dataset A')

ax[1,1].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 5], label = 'normal_A', cmap = 'r', marker = 'o'  )
ax[1,1].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 5], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1,1].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 5], label ='extrahls_A', cmap = 'k', marker = 's' )
ax[1,1].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 5], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1,1].legend()
ax[1,1].set_title('Std of the third filter MFCCs - Dataset A')

ax[1,2].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 6], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[1,2].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 6], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1,2].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 6], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[1,2].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 6], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1,2].legend()
ax[1,2].set_title('Mean of the fourth filter MFCCs - Dataset A')

ax[1,3].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 7], label = 'normal_A', cmap = 'r', marker = 'o'  )
ax[1,3].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 7], label = 'murmur_A', cmap = 'b', marker = '^')
ax[1,3].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 7], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[1,3].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 7], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[1,3].legend()
ax[1,3].set_title('Std of the fourth filter MFCCs - Dataset A')

ax[2,0].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 8], label = 'normal_A', cmap = 'r'  , marker = 'o')
ax[2,0].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 8], label = 'murmur_A', cmap = 'b', marker = '^')
ax[2,0].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 8], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[2,0].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 8], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[2,0].legend()
ax[2,0].set_title('Mean of the fifth filter MFCCs - Dataset A')

ax[2,1].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 9], label = 'normal_A', cmap = 'r'  , marker = 'o')
ax[2,1].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 9], label = 'murmur_A', cmap = 'b', marker = '^')
ax[2,1].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 9], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[2,1].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 9], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[2,1].legend()
ax[2,1].set_title('Std of the fifth filter MFCCs - Dataset A')

ax[2,2].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 10], label = 'normal_A', cmap = 'r' , marker = 'o' )
ax[2,2].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 10], label = 'murmur_A', cmap = 'b', marker = '^')
ax[2,2].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 10], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[2,2].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 10], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[2,2].legend()
ax[2,2].set_title('Mean of the sixth filter MFCCs - Dataset A')

ax[2,3].scatter(x_A_1D[0:67], train_mfcc_A[0:67, 11], label = 'normal_A', cmap = 'r'  , marker = 'o')
ax[2,3].scatter(x_A_1D[68:134], train_mfcc_A[68:134, 11], label = 'murmur_A', cmap = 'b', marker = '^')
ax[2,3].scatter(x_A_1D[135:174], train_mfcc_A[135:174, 11], label ='extrahls_A', cmap = 'k' , marker = 's')
ax[2,3].scatter(x_A_1D[175:294], train_mfcc_A[175:294, 11], label = 'artifact_A', cmap = 'g', marker = 'x')
ax[2,3].legend()
ax[2,3].set_title('Std of the sixth filter MFCCs - Dataset A')

#B
fig,ax = plt.subplots(nrows=3, ncols=4, figsize=(20,15))
fig.suptitle('Mean and std MFCCs of each filter - Dataset - B')

ax[0,0].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 0], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[0,0].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 0], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0,0].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 0], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0,0].legend()
ax[0,0].set_title('Mean of the first filter MFCCs - Dataset B')

ax[0,1].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 1], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[0,1].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 1], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0,1].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 1], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0,1].legend()
ax[0,1].set_title('Std of the first filter MFCCs - Dataset B')

ax[0,2].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 2], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[0,2].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 2], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0,2].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 2], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0,2].legend()
ax[0,2].set_title('Mean of the second filter MFCCs - Dataset B')

ax[0,3].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 3], label = 'normal_B', cmap = 'r', marker = 'o'  )
ax[0,3].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 3], label = 'murmur_B', cmap = 'b', marker = '^')
ax[0,3].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 3], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[0,3].legend()
ax[0,3].set_title('Std of the second filter MFCCs - Dataset B')

ax[1,0].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 4], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[1,0].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 4], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1,0].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 4], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1,0].legend()
ax[1,0].set_title('Mean of the third filter MFCCs - Dataset B')

ax[1,1].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 5], label = 'normal_B', cmap = 'r', marker = 'o'  )
ax[1,1].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 5], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1,1].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 5], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1,1].legend()
ax[1,1].set_title('Std of the third filter MFCCs - Dataset B')

ax[1,2].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 6], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[1,2].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 6], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1,2].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 6], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1,2].legend()
ax[1,2].set_title('Mean of the fourth filter MFCCs - Dataset B')

ax[1,3].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 7], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[1,3].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 7], label = 'murmur_B', cmap = 'b', marker = '^')
ax[1,3].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 7], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[1,3].legend()
ax[1,3].set_title('Std of the fourth filter MFCCs - Dataset B')

ax[2,0].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 8], label = 'normal_B', cmap = 'r', marker = 'o'  )
ax[2,0].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 8], label = 'murmur_B', cmap = 'b', marker = '^')
ax[2,0].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 8], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[2,0].legend()
ax[2,0].set_title('Mean of the fifth filter MFCCs - Dataset B')


ax[2,1].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 9], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[2,1].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 9], label = 'murmur_B', cmap = 'b', marker = '^')
ax[2,1].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 9], label ='extrahls_B', cmap = 'k', marker = 's' )
ax[2,1].legend()
ax[2,1].set_title('Std of the fifth filter MFCCs - Dataset B')

ax[2,2].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 10], label = 'normal_B', cmap = 'r'  , marker = 'o')
ax[2,2].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 10], label = 'murmur_B', cmap = 'b', marker = '^')
ax[2,2].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 10], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[2,2].legend()
ax[2,2].set_title('Mean of the sixth filter MFCCs - Dataset B')

ax[2,3].scatter(x_B_1D[0:589], train_mfcc_B[0:589, 11], label = 'normal_B', cmap = 'r' , marker = 'o' )
ax[2,3].scatter(x_B_1D[590:799], train_mfcc_B[590:799, 11], label = 'murmur_B', cmap = 'b', marker = '^')
ax[2,3].scatter(x_B_1D[800:872], train_mfcc_B[800:872, 11], label ='extrahls_B', cmap = 'k' , marker = 's')
ax[2,3].legend()
ax[2,3].set_title('Std of the sixth filter MFCCs - Dataset B')
###########################################################

#mfcc of each filter after reduce diamensions to 2D

train_mfcc_A_filter_1 = np.array(train_mfcc_A_filter_1)
train_mfcc_A_filter_2 = np.array(train_mfcc_A_filter_2)
train_mfcc_A_filter_3 = np.array(train_mfcc_A_filter_3)
train_mfcc_A_filter_4 = np.array(train_mfcc_A_filter_4)
train_mfcc_A_filter_5 = np.array(train_mfcc_A_filter_5)
train_mfcc_A_filter_6 = np.array(train_mfcc_A_filter_6)

train_mfcc_B_filter_1 = np.array(train_mfcc_B_filter_1)
train_mfcc_B_filter_2 = np.array(train_mfcc_B_filter_2)
train_mfcc_B_filter_3 = np.array(train_mfcc_B_filter_3)
train_mfcc_B_filter_4 = np.array(train_mfcc_B_filter_4)
train_mfcc_B_filter_5 = np.array(train_mfcc_B_filter_5)
train_mfcc_B_filter_6 = np.array(train_mfcc_B_filter_6)

train_mfcc_A_filter_1 = normalize(train_mfcc_A_filter_1, axis=0, norm='max')# normalization
train_mfcc_A_filter_2 = normalize(train_mfcc_A_filter_2, axis=0, norm='max')
train_mfcc_A_filter_3 = normalize(train_mfcc_A_filter_3, axis=0, norm='max')
train_mfcc_A_filter_4 = normalize(train_mfcc_A_filter_4, axis=0, norm='max')
train_mfcc_A_filter_5 = normalize(train_mfcc_A_filter_5, axis=0, norm='max')
train_mfcc_A_filter_6 = normalize(train_mfcc_A_filter_6, axis=0, norm='max')

train_mfcc_B_filter_1 = normalize(train_mfcc_B_filter_1, axis=0, norm='max')# normalization
train_mfcc_B_filter_2 = normalize(train_mfcc_B_filter_2, axis=0, norm='max')
train_mfcc_B_filter_3 = normalize(train_mfcc_B_filter_3, axis=0, norm='max')
train_mfcc_B_filter_4 = normalize(train_mfcc_B_filter_4, axis=0, norm='max')
train_mfcc_B_filter_5 = normalize(train_mfcc_B_filter_5, axis=0, norm='max')
train_mfcc_B_filter_6 = normalize(train_mfcc_B_filter_6, axis=0, norm='max')

#Reduce diamensions to 2D
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

#A
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
fig.suptitle('Mfccs - Dataset A')

ax[0,0].scatter(train_mfcc_A_filter_1[0:67, 0], train_mfcc_A_filter_1[0:67, 1], label = 'normal_A', cmap = 'r', alpha = 0.7  )
ax[0,0].scatter(train_mfcc_A_filter_1[68:134, 0], train_mfcc_A_filter_1[68:134, 1], label = 'murmur_A', cmap = 'b', alpha = 0.7)
ax[0,0].scatter(train_mfcc_A_filter_1[135:174, 0], train_mfcc_A_filter_1[135:174, 1], label ='extrahls_A', cmap = 'k', alpha = 0.7 )
ax[0,0].scatter(train_mfcc_A_filter_1[175:294, 0], train_mfcc_A_filter_1[175:294, 1], label = 'artifact_A', cmap = 'g', alpha = 0.7)
ax[0,0].legend()
ax[0,0].set_title('The first filter mfccs - Dataset A')

ax[0,1].scatter(train_mfcc_A_filter_2[0:67, 0], train_mfcc_A_filter_2[0:67, 1], label = 'normal_A', cmap = 'r', alpha = 0.7  )
ax[0,1].scatter(train_mfcc_A_filter_2[68:134, 0], train_mfcc_A_filter_2[68:134, 1], label = 'murmur_A', cmap = 'b', alpha = 0.7)
ax[0,1].scatter(train_mfcc_A_filter_2[135:174, 0], train_mfcc_A_filter_2[135:174, 1], label ='extrahls_A', cmap = 'k', alpha = 0.7 )
ax[0,1].scatter(train_mfcc_A_filter_2[175:294, 0], train_mfcc_A_filter_2[175:294, 1], label = 'artifact_A', cmap = 'g', alpha = 0.7)
ax[0,1].legend()
ax[0,1].set_title('The second filter mfccs - Dataset A')

ax[0,2].scatter(train_mfcc_A_filter_3[0:67, 0], train_mfcc_A_filter_3[0:67, 1], label = 'normal_A', cmap = 'r', alpha = 0.7  )
ax[0,2].scatter(train_mfcc_A_filter_3[68:134, 0], train_mfcc_A_filter_3[68:134, 1], label = 'murmur_A', cmap = 'b', alpha = 0.7)
ax[0,2].scatter(train_mfcc_A_filter_3[135:174, 0], train_mfcc_A_filter_3[135:174, 1], label ='extrahls_A', cmap = 'k', alpha = 0.7 )
ax[0,2].scatter(train_mfcc_A_filter_3[175:294, 0], train_mfcc_A_filter_3[175:294, 1], label = 'artifact_A', cmap = 'g', alpha = 0.7)
ax[0,2].legend()
ax[0,2].set_title('The third filter mfccs - Dataset A')

ax[1,0].scatter(train_mfcc_A_filter_4[0:67, 0], train_mfcc_A_filter_4[0:67, 1], label = 'normal_A', cmap = 'r', alpha = 0.7  )
ax[1,0].scatter(train_mfcc_A_filter_4[68:134, 0], train_mfcc_A_filter_4[68:134, 1], label = 'murmur_A', cmap = 'b', alpha = 0.7)
ax[1,0].scatter(train_mfcc_A_filter_4[135:174, 0], train_mfcc_A_filter_4[135:174, 1], label ='extrahls_A', cmap = 'k', alpha = 0.7 )
ax[1,0].scatter(train_mfcc_A_filter_4[175:294, 0], train_mfcc_A_filter_4[175:294, 1], label = 'artifact_A', cmap = 'g', alpha = 0.7)
ax[1,0].legend()
ax[1,0].set_title('The fourth filter mfccs - Dataset A')

ax[1,1].scatter(train_mfcc_A_filter_5[0:67, 0], train_mfcc_A_filter_5[0:67, 1], label = 'normal_A', cmap = 'r', alpha = 0.7  )
ax[1,1].scatter(train_mfcc_A_filter_5[68:134, 0], train_mfcc_A_filter_5[68:134, 1], label = 'murmur_A', cmap = 'b', alpha = 0.7)
ax[1,1].scatter(train_mfcc_A_filter_5[135:174, 0], train_mfcc_A_filter_5[135:174, 1], label ='extrahls_A', cmap = 'k', alpha = 0.7 )
ax[1,1].scatter(train_mfcc_A_filter_5[175:294, 0], train_mfcc_A_filter_5[175:294, 1], label = 'artifact_A', cmap = 'g', alpha = 0.7)
ax[1,1].legend()
ax[1,1].set_title('The fifth filter mfccs - Dataset A')

ax[1,2].scatter(train_mfcc_A_filter_6[0:67, 0], train_mfcc_A_filter_6[0:67, 1], label = 'normal_A', cmap = 'r', alpha = 0.7  )
ax[1,2].scatter(train_mfcc_A_filter_6[68:134, 0], train_mfcc_A_filter_6[68:134, 1], label = 'murmur_A', cmap = 'b', alpha = 0.7)
ax[1,2].scatter(train_mfcc_A_filter_6[135:174, 0], train_mfcc_A_filter_6[135:174, 1], label ='extrahls_A', cmap = 'k', alpha = 0.7 )
ax[1,2].scatter(train_mfcc_A_filter_6[175:294, 0], train_mfcc_A_filter_6[175:294, 1], label = 'artifact_A', cmap = 'g', alpha = 0.7)
ax[1,2].legend()
ax[1,2].set_title('The sixth filter mfccs - Dataset A')

#B
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
fig.suptitle('Mfccs - Dataset B')

ax[0,0].scatter(train_mfcc_B_filter_1[0:589, 0], train_mfcc_B_filter_1[0:589, 1], label = 'normal_B', cmap = 'r'  )
ax[0,0].scatter(train_mfcc_B_filter_1[590:799, 0], train_mfcc_B_filter_1[590:799, 1], label = 'murmur_B', cmap = 'b')
ax[0,0].scatter(train_mfcc_B_filter_1[800:872, 0], train_mfcc_B_filter_1[800:872, 1], label ='extrahls_B', cmap = 'k' )
ax[0,0].legend()
ax[0,0].set_title('The first filter mfccs - Dataset B')

ax[0,1].scatter(train_mfcc_B_filter_2[0:589, 0], train_mfcc_B_filter_2[0:589, 1], label = 'normal_B', cmap = 'r'  )
ax[0,1].scatter(train_mfcc_B_filter_2[590:799, 0], train_mfcc_B_filter_2[590:799, 1], label = 'murmur_B', cmap = 'b')
ax[0,1].scatter(train_mfcc_B_filter_2[800:872, 0], train_mfcc_B_filter_2[800:872, 1], label ='extrahls_B', cmap = 'k' )
ax[0,1].legend()
ax[0,1].set_title('The second filter mfccs - Dataset B')

ax[0,2].scatter(train_mfcc_B_filter_3[0:589, 0], train_mfcc_B_filter_3[0:589, 1], label = 'normal_B', cmap = 'r'  )
ax[0,2].scatter(train_mfcc_B_filter_3[590:799, 0], train_mfcc_B_filter_3[590:799, 1], label = 'murmur_B', cmap = 'b')
ax[0,2].scatter(train_mfcc_B_filter_3[800:872, 0], train_mfcc_B_filter_3[800:872, 1], label ='extrahls_B', cmap = 'k' )
ax[0,2].legend()
ax[0,2].set_title('The third filter mfccs - Dataset B')

ax[1,0].scatter(train_mfcc_B_filter_4[0:589, 0], train_mfcc_B_filter_4[0:589, 1], label = 'normal_B', cmap = 'r'  )
ax[1,0].scatter(train_mfcc_B_filter_4[590:799, 0], train_mfcc_B_filter_4[590:799, 1], label = 'murmur_B', cmap = 'b')
ax[1,0].scatter(train_mfcc_B_filter_4[800:872, 0], train_mfcc_B_filter_4[800:872, 1], label ='extrahls_B', cmap = 'k' )
ax[1,0].legend()
ax[1,0].set_title('The fourth filter mfccs - Dataset B')

ax[1,1].scatter(train_mfcc_B_filter_5[0:589, 0], train_mfcc_B_filter_5[0:589, 1], label = 'normal_B', cmap = 'r'  )
ax[1,1].scatter(train_mfcc_B_filter_5[590:799, 0], train_mfcc_B_filter_5[590:799, 1], label = 'murmur_B', cmap = 'b')
ax[1,1].scatter(train_mfcc_B_filter_5[800:872, 0], train_mfcc_B_filter_5[800:872, 1], label ='extrahls_B', cmap = 'k' )
ax[1,1].legend()
ax[1,1].set_title('The fifth filter mfccs - Dataset B')

ax[1,2].scatter(train_mfcc_B_filter_6[0:589, 0], train_mfcc_B_filter_6[0:589, 1], label = 'normal_B', cmap = 'r'  )
ax[1,2].scatter(train_mfcc_B_filter_6[590:799, 0], train_mfcc_B_filter_6[590:799, 1], label = 'murmur_B', cmap = 'b')
ax[1,2].scatter(train_mfcc_B_filter_6[800:872, 0], train_mfcc_B_filter_6[800:872, 1], label ='extrahls_B', cmap = 'k' )
ax[1,2].legend()
ax[1,2].set_title('The sixth filter mfccs - Dataset B')
########################################################################







    























