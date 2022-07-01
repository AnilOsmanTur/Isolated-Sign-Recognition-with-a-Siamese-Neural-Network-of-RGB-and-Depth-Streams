#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:31:00 2017

@author: anilosmantur
"""
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
# pre-trained net import
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

import cv2
import numpy as np
import os

#SRC_DATASET_PATH = "/home/anilosmantur/Documents/datasets/created_trainset/"
STD_DATASET_PATH = "D:/anil_dl/datasets/standardized_trainset_40/"
PRO_DATASET_PATH = "D:/anil_dl/datasets/processed_trainset_40/"

#PRO_DATASET_PATH = "/home/anilosmantur/Documents/datasets/processed_varied_trainset/"

def load_model():
    pre_model = ResNet50(include_top=False, weights='imagenet', pooling='max')
    return pre_model

def load_video(fileName):
    sampleVideo = []
    reader = cv2.VideoCapture(fileName)
#    print (reader.isOpened())
    if(reader.isOpened()):
        framesN = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
#        print (fileName,' frame: ',framesN)
        for i in range(framesN):
            _,x = reader.read()
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            sampleVideo.append(x)
        reader.release()
    else:
        print('error: ', fileName)
    
    return sampleVideo

def load_video_inputs(src, fileName):
    # src example: STD_DATASET_PATH   
    # file = '/Sample0002/5_color.mp4' # file name example

    colorFile = src + fileName + '_color.mp4'
    sampleVideoColor = load_video(colorFile)
    
    depthFile = src + fileName + '_depth.mp4'
    sampleVideoDepth = load_video(depthFile)    
   
    sampleVideoColor = np.array(sampleVideoColor)
    sampleVideoDepth = np.array(sampleVideoDepth)
#    print ('test color shape', sampleVideoColor.shape)
#    print ('test depth shape', sampleVideoDepth.shape)
    test_data_color = preprocess_input(sampleVideoColor)
    test_data_depth = preprocess_input(sampleVideoDepth)
#    test_data_color = np.array([test_data_color])
#    test_data_depth = np.array([test_data_depth])
#    print ('test color shape', test_data_color.shape)
#    print ('test depth shape', test_data_depth.shape)
    return test_data_color, test_data_depth

model = load_model()

info = np.load(STD_DATASET_PATH+'/dataset_header.npy')
dirNames = [i for i in os.listdir(STD_DATASET_PATH) if not 'dataset' in i]
fileNames = info[:,0]

for dirname in dirNames:
    dst_path = PRO_DATASET_PATH + dirname
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

#data_color, data_depth = load_video_inputs('Sample0002', 'Sample0002_4_')
#fileNames = fileNames[3840:]
fileCount = 1
for fileName in fileNames:
    
#    source_path = STD_DATASET_PATH + fileName
    fileName = fileName[:-10]

    print("\r%3d / %d" % (fileCount, len(fileNames)), end='') # for tracking the progress

    data_color, data_depth = load_video_inputs(STD_DATASET_PATH, fileName)
    predict = model.predict(data_color)
    np.save(PRO_DATASET_PATH + fileName +'_color.npy', predict)
    
    predict = model.predict(data_depth)
    np.save(PRO_DATASET_PATH + fileName +'_depth.npy', predict)
    fileCount += 1
    
np.save(PRO_DATASET_PATH+'/dataset_header.npy', info)


#prediction = model.predict(data_color)
#np.save('data/test_2.npy', prediction)
#reloaded = np.load('data/test_2.npy')



