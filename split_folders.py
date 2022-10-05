# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:06:07 2020

@author: carlo
"""


import os
import json
import pandas as pd
import numpy as np
import shutil
curr_dir = os.getcwd()                                               #current directory
dataset_dir = os.path.join(curr_dir, 'MaskDataset')                  #dataset directory
json_file_dir = os.path.join(dataset_dir, 'train_gt.json')           #json file directory
myDataset_dir = os.path.join(curr_dir, 'myDataset')                  #myDataset directory
if not os.path.exists(myDataset_dir):                                #I create folder for myDataset
    os.makedirs(myDataset_dir)
myTrainDataset_dir = os.path.join(myDataset_dir, 'train')            #train directory
if not os.path.exists(myTrainDataset_dir):                           #I create folder for train dataset
    os.makedirs(myTrainDataset_dir)

with open(json_file_dir) as json_file:
    labels = json.load(json_file)
df = pd.DataFrame(list(labels.items()), columns=['name', 'label'])   #I create pandas dataframe


dir_0 = os.path.join(myTrainDataset_dir,'trainClass0')               #directory for 0 class images
dir_1 = os.path.join(myTrainDataset_dir,'trainClass1')               #directory for 1 class images
dir_2 = os.path.join(myTrainDataset_dir,'trainClass2')               #directory for 2 class images
train_dir = os.path.join(dataset_dir,'training')                     #directory of training dataset
if not os.path.exists(dir_0):                                        #I create folders for class0, class1, class2
    os.makedirs(dir_0)
if not os.path.exists(dir_1):
    os.makedirs(dir_1)
if not os.path.exists(dir_2):
    os.makedirs(dir_2)    


zeros_pos = np.array(np.where(df['label']==0))                       #positions of 0 class images in the dataframe
ones_pos = np.array(np.where(df['label']==1))                        #positions of 1 class images in the dataframe
twos_pos = np.array(np.where(df['label']==2))                        #positions of 2 class images in the dataframe

for i in range(zeros_pos.shape[1]):                                  #loop over indeces of zeros_pos
    curr_name = str(df.iloc[zeros_pos[0][i]]['name'])                #get the name of the i-th image of zero class list
    curr_photo_source = os.path.join(train_dir,curr_name)            #get the address of i-th image of zero class
    shutil.copy2(curr_photo_source, dir_0)                           #copy the i-th image of zero class in class0 folder 
    
for i in range(ones_pos.shape[1]):                                   #same for class 1
    curr_name = str(df.iloc[ones_pos[0][i]]['name'])
    curr_photo_source = os.path.join(train_dir,curr_name)
    shutil.copy2(curr_photo_source, dir_1)
    
for i in range(twos_pos.shape[1]):                                   #same for class 2
    curr_name = str(df.iloc[twos_pos[0][i]]['name'])
    curr_photo_source = os.path.join(train_dir,curr_name)
    shutil.copy2(curr_photo_source, dir_2)

testDataset_dir = os.path.join(dataset_dir, 'test')
shutil.copytree(testDataset_dir, os.path.join(myDataset_dir,'test'))
        
        
        