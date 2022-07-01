#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 22:54:52 2018

@author: anilosmantur
"""


# pre-trained net import
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

import cv2
import numpy as np
import os

#SRC_DATASET_PATH = "/home/anilosmantur/Documents/datasets/created_trainset/"
STD_DATASET_PATH = "/home/anilosman/Documents/datasets/standardized_trainset_40/"
PRO_DATASET_PATH = "/home/anilosman/Documents/datasets/vgg_processed_trainset_40_avg/"

#PRO_DATASET_PATH = "/home/anilosmantur/Documents/datasets/processed_varied_trainset/"

def load_model():
    pre_model = VGG16(include_top=False, weights='imagenet', pooling='avg')
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

def load_video_inputs(src, fileName, choose=2):
    # src example: STD_DATASET_PATH   
    # file = '/Sample0002/5_color.mp4' # file name example
    if choose != 1:
        colorFile = src + fileName + '_color.mp4'
        sampleVideoColor = load_video(colorFile)
        sampleVideoColor = np.array(sampleVideoColor)
        test_data_color = preprocess_input(sampleVideoColor)

    if choose != 0:
        depthFile = src + fileName + '_depth.mp4'
        sampleVideoDepth = load_video(depthFile)    
        sampleVideoDepth = np.array(sampleVideoDepth)
        test_data_depth = preprocess_input(sampleVideoDepth)
    
    if choose == 0:
        return test_data_color
    elif choose == 1:
        return test_data_depth
    return test_data_color, test_data_depth

model = load_model()

info = np.load(STD_DATASET_PATH+'/dataset_header.npy')
dirNames = [i for i in os.listdir(STD_DATASET_PATH) if not 'dataset' in i]
fileNames = info[:,0]

if not os.path.isdir(PRO_DATASET_PATH):
    os.mkdir(PRO_DATASET_PATH)
for dirname in dirNames:
    dst_path = PRO_DATASET_PATH + dirname
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

#data_color, data_depth = load_video_inputs('Sample0002', 'Sample0002_4_')

fileCount = 1
for fileName in fileNames:
#    source_path = STD_DATASET_PATH + fileName
    fileName = fileName[:-10]

    print("\rcolor %3d / %d" % (fileCount, len(fileNames)), end='') # for tracking the progress

    data_color = load_video_inputs(STD_DATASET_PATH, fileName, 0)
    predict = model.predict(data_color)
    np.save(PRO_DATASET_PATH + fileName +'_color.npy', predict)
    fileCount += 1
    
fileCount = 1
for fileName in fileNames:
#    source_path = STD_DATASET_PATH + fileName
    fileName = fileName[:-10]

    print("\rdepth %3d / %d" % (fileCount, len(fileNames)), end='') # for tracking the progress
    data_depth = load_video_inputs(STD_DATASET_PATH, fileName, 1)
    predict = model.predict(data_depth)
    np.save(PRO_DATASET_PATH + fileName +'_depth.npy', predict)
    fileCount += 1
    
np.save(PRO_DATASET_PATH+'/dataset_header.npy', info)


#prediction = model.predict(data_color)
#np.save('data/test_2.npy', prediction)
#reloaded = np.load('data/test_2.npy')
