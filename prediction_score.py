import numpy as np
import matplotlib.pyplot as plt

from time import time

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

from pystruct.models import GraphCRF, LatentGraphCRF
from pystruct.learners import NSlackSSVM, LatentSSVM

import csv
from itertools import chain

### Load the scikit-learn digits classification dataset.
##digits = load_digits()
##X, y_org = digits.data, digits.target

def list2features(data_list, ratio):
    return [num2features(data_list, i+1) for i in range(int(ratio*len(data_list)-1))]

def list2labels_sleep(data_list, ratio):
    return [1 if int(data_list[i+1][2])==3004 else 0 for i in range(int(ratio*len(data_list)-1))]
#    return [str(int(data_list[i+1][2])==3004) for i in range(len(data_list)-1)]

def list2labels_tv(data_list, ratio):
    return [1 if int(data_list[i+1][2])==5102 else 0 for i in range(int(ratio*len(data_list)-1))]

def num2features(sent, i):
    try:
        features = [
                float(sent[i][4]),
                float(sent[i][5]),
                float(sent[i][6]),
                float(sent[i][7]),
                float(sent[i][8]),
                float(sent[i][9]),
                float(sent[i][10]),
                float(sent[i][11]),
                float(sent[i][12]),
                ]
    except ValueError:
        features = [0]*9
        
    return features


with open('/Users/Zhanghongzhuo/Desktop/EECS6893/TrainingSet/TrainingSet_copy.csv', 'rb') as f:
    reader = csv.reader(f)
    train_list = list(reader)

X_org = list2features(train_list, i)
X = np.array(X_org)
y = list2labels_sleep(train_list, i)
y_org = np.array(y)
Y = y_org.reshape(-1, 1)
X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]

X_train, X_test, y_train, y_test = train_test_split(X_, Y)

def train_crf(X_train, y_train):
    pbl = GraphCRF(inference_method='unary')
    svm = NSlackSSVM(pbl, C=100)
    svm.fit(X_train, y_train)
    return svm

count = 2001
X_train_tmp = X_train
y_train_tmp = y_train

    
with open('/Users/Zhanghongzhuo/Desktop/EECS6893/TrainingSet/TrainingSet_copy.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        X_in = np.array(list2features(row, 1))
        Y_in = np.array(list2labels_sleep(row,1)).reshape(-1,1)
        if count > 2000:
            y_pred = np.vstack(svm.predict(X_train_tmp))
            print("Score with crf in the last period: %f "
                  % np.mean(y_pred == y_train_tmp))
            svm = train_crf(X_train_tmp, y_train_tmp)
            count = 0
            X_train_tmp = X_in
            y_train_tmp = Y_in
        
        else:
            X_train_tmp = np.vstack((X_train_tmp,X_in))
            y_train_tmp = np.vstack((y_train_tmp,Y_in))
            count = count + 1
