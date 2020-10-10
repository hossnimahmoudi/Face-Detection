#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:28:35 2018

@author: hosni
"""
#header = np.fromfile('/YALE/faces/subject01.centerlight.pgm' )
#print(header)
import numpy as np
import glob
from PIL import Image
import os
from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import multiclass
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import imghdr, struct


liste=[]
for i in os.listdir("/Downloads/YALE/faces/"):
#    if imghdr.what(i)!='.pgm':
    if i != '*.pgm':
#        print(i)
        im=Image.open("/Downloads/YALE/faces/"+i)
        print(im)
        Data=list(im.getdata())
        transposer=np.transpose(Data)
        liste.append(transposer)
print(len(liste))
pca=decomposition.PCA(n_components=500)
y=pca.fit_transform(liste)
print(y)
label1=np.zeros((11,1))
label2=np.ones((11,1))
label3=np.ones((11,1))*2
label4=np.ones((11,1))*3
label5=np.ones((11,1))*4
label6=np.ones((11,1))*5
label7=np.ones((11,1))*6
label8=np.ones((11,1))*7
label9=np.ones((11,1))*8
label10=np.ones((11,1))*9
label11=np.ones((11,1))*10
label12=np.ones((11,1))*11
label13=np.ones((11,1))*12
label14=np.ones((11,1))*13
label15=np.ones((11,1))*14
label=np.concatenate((label1,label2,label3,label4,label5,label6,label7,label8,label9,
                      label10,label11,label12,label13,label14,label15),axis=0)
print("longueur de y est:",len(y))
data1_train,data1_test,label_train,label_test=train_test_split(y,label,test_size=0.33)
print("data1_train",len(data1_train.shape))
print("data1_test",len(data1_test.shape))
#clf = OneVsRestClassifier(estimator,n_jobs=1)
clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)

#clf = OneVsRestClassifier(LinearSVC(random_state=0))

#clf = OneVsRestClassifier(SVC(kernel='rbf'))
clf.fit(data1_train,np.ravel(label_train))
predicted=clf.predict(data1_test)
print("Predicted =",predicted)
accuracy=accuracy_score(label_test,predicted)
print("accuracy ",accuracy)
