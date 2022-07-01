#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:21:32 2017

@author: anilosmantur
"""

import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import os

# keras layer imports
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import Concatenate

PRO_DATASET_PATH = "/home/anilosmantur/Documents/datasets/processed_trainset/"

def load_model(input_lenght=40, dense=128, lstm=128, lr=0.001, decay=0.0):
    print('[info] loading model...')
    # color input
    x_color = Input(shape=(2048,))
    feat_color = Dense(dense, activation='relu')(x_color)
    model_color = Model(inputs=x_color, outputs=feat_color)
    
    # depth input
    x_depth = Input(shape=(2048,))
    feat_depth = Dense(dense, activation='relu')(x_depth)
    model_depth = Model(inputs=x_depth, outputs=feat_depth)
    
    # color model input shape definition
    in_color = Input(shape=(input_lenght, 2048))
    # sequentialization the network
    color_seq = TimeDistributed(model_color)(in_color)
    
    # color model input shape definition
    in_depth = Input(shape=(input_lenght, 2048))
    # sequentialization the network
    depth_seq = TimeDistributed(model_depth)(in_depth)
     
    # merging these two features from depth and color
    merged_feat = Concatenate(axis=2)([color_seq, depth_seq])

    sequence = LSTM(lstm)(merged_feat)
    
    predictions = Dense(20, activation='softmax')(sequence) # output to 20 categories
    
    model = Model(inputs=[in_color, in_depth], outputs=predictions) # model to train
    
    # we need costum optimizer    
    opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc']) 
    print('[info] model ready to train.')
    return model


def load_video_inputs(inFileDir, gestureID):
    # dir example: 'Sample0002'      
    # file = '/Sample0002_5_color.mp4' # file name example
    
#    print(gestureID, ' ', inFileDir)
    
    inFileDir = '/' + inFileDir
    colorFile = inFileDir + inFileDir + '_' + str(gestureID) + '_color.npy'
    dataColor = np.load(PRO_DATASET_PATH + colorFile)
    
    depthFile = inFileDir + inFileDir + '_' + str(gestureID) + '_depth.npy'
    dataDepth = np.load(PRO_DATASET_PATH + depthFile)    
   
    label = np.zeros(20)
    label[gestureID-1] = 1
    
#    print ('test color shape', dataColor.shape)
#    print ('test depth shape', dataDepth.shape)
    return dataColor, dataDepth, label

def load_dataset(sampleCount):
    print('[info] loading dataset...')
    dirNames = [i for i in os.listdir(PRO_DATASET_PATH)]
    dirNames.sort()
    
    totalSampleTypes = []
    for dirName in dirNames:
        path = PRO_DATASET_PATH + dirName
        files = [int(i[11:-10]) for i in os.listdir(path) if 'color' in i]
        totalSampleTypes.append(files)
    
    totalGestureFileNames = []
    for i in range(1,21):
            gestureFileNames = []
            for j, sampleTypes in enumerate(totalSampleTypes):
                for sType in sampleTypes:
                    if sType == i:
                        gestureFileNames.append(dirNames[j])
#            print("Gesture size: ",len(gestureFileNames))
            shuffle(gestureFileNames)
            totalGestureFileNames.append(gestureFileNames[:sampleCount])
            
    # gether the dataset
    color_videos = []
    depth_videos = []
    labels = []
    x = np.arange(20)
    for i in range(sampleCount):
        np.random.shuffle(x)
#        j = x[2]
        for j in x:
            color, depth, label = load_video_inputs(totalGestureFileNames[j][i],j+1)
            color = np.array(color)
            depth = np.array(depth)
            label = np.array(label)
            depth_videos.append(depth)
            color_videos.append(color)
            labels.append(label)
    print('[info] dataset loaded.')
    return color_videos, depth_videos, labels
            
#model = load_model()

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

dataColor, dataDepth, labels = load_dataset(150)
dataColor = np.array(dataColor)
dataDepth = np.array(dataDepth)
labels = np.array(labels, dtype='float32')

#hist = model.fit(x=[dataColor, dataDepth], y=labels, epochs=100, validation_split=0.1, shuffle=True)

def plotHistory(history, i):
    print('[info] saving the best results.')
    acc = np.array(history.history['acc'])
    val_acc = np.array(history.history['val_acc'])
    loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    best_acc = np.argmax(acc)
    best_valacc = np.argmax(val_acc)
    best_loss = np.argmin(loss)
    best_valloss = np.argmin(val_loss)
    with open('stats/best_acc_loss_test_'+str(i)+'.txt', 'w') as f:
        f.write('epoch: '+str(best_acc)+' acc: ' + str(acc[best_acc])+'\n')
        f.write('epoch: '+str(best_valacc)+' val_acc: ' + str(val_acc[best_valacc])+'\n')
        f.write('epoch: '+str(best_loss)+' loss: ' + str(loss[best_loss])+'\n')
        f.write('epoch: '+str(best_valloss)+' val_loss: ' + str(val_loss[best_valloss])+'\n')
    # print(history.history.keys())
    # summarize history for accuracy
    plt.plot(acc, label='acc')
    plt.plot(best_acc, acc[best_acc], 'ro', label='_nolegend_')
    plt.plot(val_acc, label='val_acc')
    plt.plot(best_valacc, val_acc[best_valacc], 'ro', label='_nolegend_')
    plt.axis([0,acc.size-1, 0, acc.max()])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('stats/trainAccHist_'+ str(i) +'.png')
    plt.show()
    plt.close()
    # summarize history for loss
    plt.plot(loss, label='loss')
    plt.plot(best_loss, loss[best_loss], 'ro', label='_nolegend_')
    plt.plot(val_loss, label='val_loss')
    plt.plot(best_valloss, val_loss[best_valloss], 'ro', label='_nolegend_')
    plt.axis([0,loss.size-1, loss.min(), loss.max()])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('stats/trainLossHist_'+ str(i) +'.png')
    plt.show()
    plt.close()
    
    # one together
    plt.plot(acc, label='acc')
    plt.plot(best_acc, acc[best_acc], 'ro', label='_nolegend_')
    plt.plot(val_acc, label='val_acc')
    plt.plot(best_valacc, val_acc[best_valacc], 'ro', label='_nolegend_')
    # summarize history for loss
    plt.plot(loss, label='loss')
    plt.plot(best_loss, loss[best_loss], 'ro', label='_nolegend_')
    plt.plot(val_loss, label='val_loss')
    plt.plot(best_valloss, val_loss[best_valloss], 'ro', label='_nolegend_')
    plt.title('model accuracy loss')
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('stats/acc_loss_plot_'+ str(i) +'.png')
    plt.show()
    plt.close()

#plotHistory(hist, 1)

lrs = [pow(10,random.uniform(-6, -4)) for i in range(50)]

#print(lrs)

histories = []
for i, lr in enumerate(lrs):
    print('LR Test: '+str(i)+'/'+str(len(lrs)) + 'lr = ', lr)
    model = load_model(input_lenght=40, dense=128, lstm=128, lr=lr, decay=0.0)
    hist = model.fit(x=[dataColor, dataDepth], y=labels, batch_size=16, epochs=3, validation_split=0.1, shuffle=True)
    histories.append([lr, hist])
    del model
    
    
acc=[]
loss=[]
for hist in histories:
    acc.append([hist[0],hist[1].history['acc'][2],hist[1].history['val_acc'][2]])
    loss.append([hist[0],hist[1].history['loss'][2],hist[1].history['val_loss'][2]])

acc.sort()
loss.sort()

with open('lr_fineTune_stats/lr_acc_cases_3.txt','w') as f:
    for a in acc:
        f.write('lr: '+str(a[0])+' acc: ' + str(a[1])+' val_acc: ' + str(a[2])+'\n')

with open('lr_fineTune_stats/lr_loss_cases_3.txt','w') as f:
    for l in loss:
        f.write('lr: '+str(l[0])+' loss: ' + str(l[1])+' val_loss: ' + str(l[2])+'\n')

acc2 = np.array(acc)
        
plt.plot(acc2[:,0],acc2[:,1], label='acc')
plt.plot(acc2[:,0],acc2[:,2], label='val_acc')
plt.title('lr model accuracy')
plt.ylabel('accuracy')
plt.xlabel('lr')
plt.legend()
plt.show()
plt.savefig('lrAcc_'+ str(3) +'.png')
plt.close()

loss2 = np.array(loss)
plt.plot(loss2[:,0],loss2[:,1], label='loss')
plt.plot(loss2[:,0],loss2[:,2], label='val_loss')
plt.title('lr model loss')
plt.ylabel('Loss')
plt.xlabel('lr')
plt.legend()
plt.show()
plt.savefig('lrLoss_'+ str(3) +'.png')
plt.close()


dense_sizes = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024]
lstm_sizes =  [32, 64,  64, 128,  64, 128, 256, 128, 256, 512,  128,  256,  512, 1024]
lr = 9.532019405681943e-05
for i, (dense, lstm) in enumerate(zip(dense_sizes, lstm_sizes), start=1):
    model = load_model(input_lenght=40, dense=1024, lstm=lstm, lr=lr, decay=0.0)
    hist = model.fit(x=[dataColor, dataDepth], y=labels, batch_size=16, epochs=100, validation_split=0.1, shuffle=True)
    plotHistory(hist, i)
    del(model, hist)

