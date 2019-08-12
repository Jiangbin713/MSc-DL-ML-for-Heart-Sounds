# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:57:14 2019

@author: jiang

What has been done in this file?

    1.  output results into a excel file and the order is the same as the evaluation file
        
        Param: 
            which_dataset   :  A or B
"""
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import pickle
class Config():
    def __init__(self,
                 Evaluation_excel = r'D:\CVML\Project\Heartchallenge_sound\Peter_dataset\Evaluation_Files\Challenge2_evaluation_sheet.xlsx', 
                 which_dataset = 'A'
                 
                 
                 ):
        self.Evaluation_excel = Evaluation_excel
        self.dataset = which_dataset
config = Config()

path = r'D:\CVML\Project\Heartchallenge_sound\Py_code\MachineLearn\Self\Saved_params'

#Results
results = results



test_wav_A = pickle.load(open(path +r'\test_wav_A.txt', 'rb') )   #file name

test_wav_B = pickle.load(open(path +r'\test_wav_B.txt', 'rb') )



#载入csv file 的文件名列
if config.dataset == 'A':
    Evaluation_list = pd.read_excel(config.Evaluation_excel, usecols = [0], sheet_name = 'Dataset A')
    label_order = test_wav_A 
    postfix_num = -6
else:
    Evaluation_list = pd.read_excel(config.Evaluation_excel, usecols = [0], sheet_name = 'Dataset B')        
    label_order = test_wav_B
    postfix_num = -6
    
#Voting
Voting_result_dict = {}
voting_0, voting_1, voting_2, voting_3 = 0,0,0,0
count = 0
new_index = 0


for num in range(0,len(label_order)-1):
    label_flag_up = label_order[num][:-2]  #current label filename
    label_flag_down = label_order[num+1][:-2] # next label filename
    
    if label_flag_up == label_flag_down:
        if results[num] == 0:
            voting_0 += 1
        elif results[num] == 1:
            voting_1 += 1
        elif results[num] == 2:
            voting_2 += 1
        elif results[num] == 3:
            voting_3 += 1
    else:
        if results[num] == 0:
            voting_0 += 1
        elif results[num] == 1:
            voting_1 += 1
        elif results[num] == 2:
            voting_2 += 1
        elif results[num] == 3:
            voting_3 += 1
        temp = [voting_0,voting_1,voting_2,voting_3]
        temp_max = max(temp)
        temp_max_index = [i for i,x in enumerate(temp) if x == temp_max]
        Voting_result_dict[new_index] = [label_order[num][:-2],temp_max_index]
        new_index += 1
        voting_0, voting_1, voting_2, voting_3 = 0, 0, 0, 0
    if num == len(label_order)-2 and label_order[num][:-2] ==  label_order[num+1][:-2]:
        if results[num] == 0:
            voting_0 += 1
        elif results[num] == 1:
            voting_1 += 1
        elif results[num] == 2:
            voting_2 += 1
        elif results[num] == 3:
            voting_3 += 1
        temp = [voting_0,voting_1,voting_2,voting_3]
        temp_max = max(temp)
        temp_max_index = [i for i,x in enumerate(temp) if x == temp_max]
        Voting_result_dict[new_index] = [label_order[num][:-2],temp_max_index]
        new_index += 1
        voting_0, voting_1, voting_2, voting_3 = 0, 0, 0, 0
    
##### Match the Evaluation excel file order ####   
        
for num in range(0, len(Voting_result_dict)):
    if config.dataset == 'A':
        Voting_result_dict[num][0] = Voting_result_dict[num][0] + '.aif'
    else:
        Voting_result_dict[num][0] = Voting_result_dict[num][0] + '.aiff'
        
excel_result = {}

for num in range(0, len(Voting_result_dict)):
    if config.dataset == 'A':
        if Voting_result_dict[num][0] in Evaluation_list['Dataset A'].tolist():
            excel_result[  np.where(Evaluation_list == Voting_result_dict[num][0])[0][0]   ] = [    Voting_result_dict[num][0] ,  Voting_result_dict[num][1]    ]
    else:
        if Voting_result_dict[num][0] in Evaluation_list['Dataset B'].tolist():
            excel_result[  np.where(Evaluation_list == Voting_result_dict[num][0])[0][0]   ] = [    Voting_result_dict[num][0] ,  Voting_result_dict[num][1]    ]

###### convert to csv ##########
excel_file = pd.DataFrame(columns=['filename','normal','murmur','extrahls','artifact']) 

for n in excel_result.keys():
    excel_file.loc[n, 'filename'] = excel_result[n][0]
    if excel_result[n][1] == [0]:
        excel_file.loc[n, 'normal'] = 1
    if excel_result[n][1] == [1]:
        excel_file.loc[n, 'murmur'] = 1
    if excel_result[n][1] == [2]:
        excel_file.loc[n, 'extrahls'] = 1
    if excel_result[n][1] == [3]:
        excel_file.loc[n, 'artifact'] = 1

excel_file.sort_index(inplace=True)    
excel_file.fillna(0, inplace=True)  


 
if config.dataset == 'A':
        
    excel_file.to_excel('A.xls')

else:
    insertRow = excel_file.loc[97]
    above = excel_file.loc[:96]
    below = excel_file.loc[97:]    
    excel_file = above.append(insertRow,ignore_index=True).append(below,ignore_index=False)    
    
    insertRow = pd.DataFrame([['181_1308052613891_A.aiff',1.,0.,0.,0.]],columns=['filename','normal','murmur','extrahls','artifact'])
    above = excel_file.loc[:111]
    below = excel_file.loc[112:]  
    excel_file = above.append(insertRow,ignore_index=True).append(below,ignore_index=False)  
    
    excel_file.to_excel('B.xls')

