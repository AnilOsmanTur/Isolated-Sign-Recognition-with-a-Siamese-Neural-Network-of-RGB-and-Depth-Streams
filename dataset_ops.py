#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:52:39 2018

@author: anilosmantur
"""

import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.preprocessing import image
from tempfile import mkdtemp


STD_DATASET_PATH = "/home/anilosmantur/Documents/datasets/standardized_trainset_40/"

def load_video(fileName):
    w = h = 224
    sampleVideo = []
    reader = cv2.VideoCapture(fileName)
#    print (reader.isOpened())
    if(reader.isOpened()):
        framesN = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        print (fileName,' frame: ',framesN)
        for i in range(framesN):
            _,img = reader.read()
            img = cv2.resize(img, (w,h), cv2.INTER_CUBIC)
            x = image.img_to_array(img)
            #x = np.expand_dims(x, axis=0)
            sampleVideo.append(x)
        reader.release()
    else:
        print('error: ', fileName)
    
    return sampleVideo

def load_video_inputs(src, fileName, preprocess):
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
    data_color = preprocess(sampleVideoColor)
    data_depth = preprocess(sampleVideoDepth)
#    test_data_color = np.array([test_data_color])
#    test_data_depth = np.array([test_data_depth])
#    print ('test color shape', test_data_color.shape)
#    print ('test depth shape', test_data_depth.shape)
    return data_color, data_depth


def load_videos(fileName, gestureID, preprocess):
    w = h = 224
    # dir example: PRO_DATASET_PATH      
    # file = 'Sample0003/5_color.mp4' # file name example
    
#    print(gestureID, ' ', inFileDir)
    colorFile = STD_DATASET_PATH + fileName + '_color.mp4'
    videoColor = []
    reader = cv2.VideoCapture(colorFile)
    if(reader.isOpened()):
        framesN = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(framesN):
            _,img = reader.read()
            img = cv2.resize(img, (w,h), cv2.INTER_CUBIC)
            x = image.img_to_array(img)
            videoColor.append(x)
        reader.release()
    videoColor = np.array(videoColor)
    videoColor = preprocess(videoColor)
            
    
    depthFile = STD_DATASET_PATH + fileName + '_depth.mp4'
    videoDepth = []
    reader = cv2.VideoCapture(depthFile)
    if(reader.isOpened()):
        framesN = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(framesN):
            _,img = reader.read()
            img = cv2.resize(img, (w,h), cv2.INTER_CUBIC)
            x = image.img_to_array(img)
            videoDepth.append(x)
        reader.release()
    videoDepth = np.array(videoDepth)
    videoDepth = preprocess(videoDepth)   
   
    label = np.zeros(20)
    label[gestureID-1] = 1
    
#    print ('test color shape', dataColor.shape)
#    print ('test depth shape', dataDepth.shape)
    return videoColor, videoDepth, label
#    return np.array([dataColor]), np.array([dataDepth]), np.array([label])

def load_dataset(preprocess, num=0, path=''):
    print('[info] loading dataset...')
    info = np.load(STD_DATASET_PATH+'/dataset_header.npy')
    
    dirNames = [i for i in os.listdir(STD_DATASET_PATH) if not 'dataset_header' in i]
    files = []
    
    if num>0:
        dirNames = dirNames[:num]
        
    for dirName in dirNames:
        path = STD_DATASET_PATH + dirName
        x = [dirName +'/'+ i for i in os.listdir(path) if 'color' in i]
        files = files + x
    
    header = []
    for i in info:
        if str(i[0]) in files:
            header.append(i)
    header = np.array(header)

    trainFileNames = []
    testFileNames = []
    for i in range(1,21):
        gestureFileNames = []
        for j in header:
            if int(j[1]) == i:
                gestureFileNames.append([str(j[0][:-10]), i])
        shuffle(gestureFileNames)
        x = len(gestureFileNames)
        x = int(x * 0.8 + 0.2)
        trainFileNames += gestureFileNames[:x] # %80 train
        testFileNames += gestureFileNames[x:] # %20 test
        
    shuffle(trainFileNames) # additional randomness
    shuffle(testFileNames) # additional randomness
    np.save(path+'/dataset_header_train.npy', trainFileNames)
    np.save(path+'/dataset_header_test.npy', testFileNames)
    
    # gether the train dataset
    file1 = os.path.join(mkdtemp(), 'trainC.dat')
    file2 = os.path.join(mkdtemp(), 'trainD.dat')
    file3 = os.path.join(mkdtemp(), 'trainY.dat')
    n_sample = len(trainFileNames)
    print('train ',n_sample)
    fileCount = 0
    first = True
    train_cVideos = None
    train_dVideos = None
    train_labels = None
    for name in trainFileNames:
        color, depth, label = load_videos(name[0], name[1], preprocess)
        if first:
            # instead of empty memmap recomended 
            train_cVideos = np.memmap(file1, mode='w+', shape=(n_sample, color.shape[0], color.shape[1], color.shape[2], color.shape[3]), dtype='float32')
            train_dVideos = np.memmap(file2, mode='w+', shape=(n_sample, depth.shape[0], depth.shape[1], depth.shape[2], depth.shape[3]), dtype='float32')
            train_labels = np.memmap(file3, mode='w+', shape=(n_sample, label.shape[0]), dtype='float32')
            first = False
            
            train_cVideos[fileCount] = color
            train_dVideos[fileCount] = depth
            train_labels[fileCount] = label
        else:
            train_cVideos[fileCount] = color
            train_dVideos[fileCount] = depth
            train_labels[fileCount] = label
        print("\rtrainset: %d / %d" % (fileCount+1, n_sample), end='\r')
        fileCount += 1
    print('')
        
#    train_cVideos = np.array(train_cVideos)
#    train_dVideos = np.array(train_dVideos)
#    train_labels = np.array(train_labels, dtype='float32')
    

    # gether the test dataset
    file1 = os.path.join(mkdtemp(), 'testC.dat')
    file2 = os.path.join(mkdtemp(), 'testD.dat')
    file3 = os.path.join(mkdtemp(), 'testY.dat')
    n_sample = len(testFileNames)
    print('test ',n_sample)
    fileCount = 0
    first = True
    test_cVideos = None
    test_dVideos = None
    test_labels = None
    for name in testFileNames:
        color, depth, label = load_videos(name[0], name[1], preprocess)
        if first:
            # instead of empty memmap recomended 
            test_cVideos = np.memmap(file1, mode='w+', shape=(n_sample, color.shape[0], color.shape[1], color.shape[2], color.shape[3]), dtype='float32')
            test_dVideos = np.memmap(file2, mode='w+', shape=(n_sample, depth.shape[0], depth.shape[1], depth.shape[2], depth.shape[3]), dtype='float32')
            test_labels = np.memmap(file3, mode='w+', shape=(n_sample, label.shape[0]), dtype='float32')
            first = False
            
            test_cVideos[fileCount] = color
            test_dVideos[fileCount] = depth
            test_labels[fileCount] = label
        else:
            test_cVideos[fileCount] = color
            test_dVideos[fileCount] = depth
            test_labels[fileCount] = label
        print("\rtestset: %d / %d" % (fileCount+1, n_sample), end='\r')
        fileCount += 1
    print('')
    
    print('[info] dataset loaded.')
    
    return [train_cVideos, train_dVideos, train_labels], [test_cVideos, test_dVideos, test_labels]

def load_dataset_ges(preprocess, ges=0, path=''):
    print('[info] loading dataset...')
    info = np.load(STD_DATASET_PATH+'/dataset_header.npy')
    
    ges=3
    gestures = [i for i in range(20)]
    shuffle(gestures)
    if ges > 0:
        gestures = gestures[:ges]
    print('gestures will be used are: ', gestures)
    
    header = []
    for i in info:
        if int(i[1]) in gestures:
            header.append(i)
    header = np.array(header)

    trainFileNames = []
    testFileNames = []
    for i in range(1,21):
        gestureFileNames = []
        for j in header:
            if int(j[1]) == i:
                gestureFileNames.append([str(j[0][:-10]), i])
        shuffle(gestureFileNames)
        x = len(gestureFileNames)
        x = int(x * 0.8 + 0.2)
        trainFileNames += gestureFileNames[:x] # %80 train
        testFileNames += gestureFileNames[x:] # %20 test
        
    shuffle(trainFileNames) # additional randomness
    shuffle(testFileNames) # additional randomness
    np.save(path+'/dataset_header_train.npy', trainFileNames)
    np.save(path+'/dataset_header_test.npy', testFileNames)
    
    # gether the train dataset
    file1 = os.path.join(mkdtemp(), 'trainC.dat')
    file2 = os.path.join(mkdtemp(), 'trainD.dat')
    file3 = os.path.join(mkdtemp(), 'trainY.dat')
    n_sample = len(trainFileNames)
    print('train ',n_sample)
    fileCount = 0
    first = True
    train_cVideos = None
    train_dVideos = None
    train_labels = None
    for name in trainFileNames:
        color, depth, label = load_videos(name[0], name[1], preprocess)
        if first:
            # instead of empty memmap recomended 
            train_cVideos = np.memmap(file1, mode='w+', shape=(n_sample, color.shape[0], color.shape[1], color.shape[2], color.shape[3]), dtype='float32')
            train_dVideos = np.memmap(file2, mode='w+', shape=(n_sample, depth.shape[0], depth.shape[1], depth.shape[2], depth.shape[3]), dtype='float32')
            train_labels = np.memmap(file3, mode='w+', shape=(n_sample, label.shape[0]), dtype='float32')
            first = False
            
            train_cVideos[fileCount] = color
            train_dVideos[fileCount] = depth
            train_labels[fileCount] = label
        else:
            train_cVideos[fileCount] = color
            train_dVideos[fileCount] = depth
            train_labels[fileCount] = label
        print("\rtrainset: %d / %d" % (fileCount+1, n_sample), end='\r')
        fileCount += 1
    print('')
        
#    train_cVideos = np.array(train_cVideos)
#    train_dVideos = np.array(train_dVideos)
#    train_labels = np.array(train_labels, dtype='float32')
    

    # gether the test dataset
    file1 = os.path.join(mkdtemp(), 'testC.dat')
    file2 = os.path.join(mkdtemp(), 'testD.dat')
    file3 = os.path.join(mkdtemp(), 'testY.dat')
    n_sample = len(testFileNames)
    print('test ',n_sample)
    fileCount = 0
    first = True
    test_cVideos = None
    test_dVideos = None
    test_labels = None
    for name in testFileNames:
        color, depth, label = load_videos(name[0], name[1], preprocess)
        if first:
            # instead of empty memmap recomended 
            test_cVideos = np.memmap(file1, mode='w+', shape=(n_sample, color.shape[0], color.shape[1], color.shape[2], color.shape[3]), dtype='float32')
            test_dVideos = np.memmap(file2, mode='w+', shape=(n_sample, depth.shape[0], depth.shape[1], depth.shape[2], depth.shape[3]), dtype='float32')
            test_labels = np.memmap(file3, mode='w+', shape=(n_sample, label.shape[0]), dtype='float32')
            first = False
            
            test_cVideos[fileCount] = color
            test_dVideos[fileCount] = depth
            test_labels[fileCount] = label
        else:
            test_cVideos[fileCount] = color
            test_dVideos[fileCount] = depth
            test_labels[fileCount] = label
        print("\rtestset: %d / %d" % (fileCount+1, n_sample), end='\r')
        fileCount += 1
    print('')
    
    print('[info] dataset loaded.')
    
    return [train_cVideos, train_dVideos, train_labels], [test_cVideos, test_dVideos, test_labels]
