import numpy as np
import matplotlib.pyplot as plt

from time import time

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

from pystruct.models import GraphCRF, LatentGraphCRF
from pystruct.learners import NSlackSSVM, LatentSSVM

import csv
import math
from itertools import chain

with open('/Users/Zhanghongzhuo/Desktop/EECS6893/TrainingSet/TrainingSet.csv','rb') as f:
	 reader = csv.reader(f)
	 train_list = list(reader)

def list2features(data_list):
    #num = int(ratio*len(data_list))
    return [num2features(data_list, i+1) for i in range(len(data_list)-1)]

def list2labels_sleep(data_list):
    #num = int(ratio*len(data_list))
    return [1 if int(data_list[i+1][2])==3004 else 0 for i in range(len(data_list)-1)]
    #return [str(int(data_list[i+1][2])==3004) for i in range(len(data_list)-1)]

def list2labels_tv(data_list):
    #num = int(ratio*len(data_list))
    return [1 if int(data_list[i+1][2])==5102 else 0 for i in range(len(data_list)-1)]

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

def runIt(train_list):
    X_org = list2features(train_list)
    X = np.array(X_org)
    y = list2labels_sleep(train_list)
    y_org = np.array(y)
    Y = y_org.reshape(-1, 1)

    X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
    X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size=.5)

    pbl = GraphCRF(inference_method='unary')
    svm = NSlackSSVM(pbl, C=100)

    start = time()
    svm.fit(X_train, y_train)
    time_svm = time() - start
    y_pred = np.vstack(svm.predict(X_test))
    print("Score with pystruct crf svm: %f (took %f seconds)"
          % (np.mean(y_pred == y_test), time_svm))

# ratio = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# for i in ratio:
#     numTuple
#     runIt(i)

numSlice = 20000
accuracy = 0
total_time = 0
total_accuracy = 0
# we take 20 segments of 20000 tuples from the original physiological data, which is more than half of the available data
iterations = 20
for i in range(iterations):
    # we can update the crf model either on every iteration or on selected rounds based on its previous performance
    if accuracy < 0.95:
        try:
            #runIt(train_list[i*numSlice+1:(i+1)*numSlice])
            X_org = list2features(train_list[i*numSlice+1:(i+1)*numSlice])
            X = np.array(X_org)
            y = list2labels_sleep(train_list[i*numSlice+1:(i+1)*numSlice])
            y_org = np.array(y)
            Y = y_org.reshape(-1, 1)
        
            X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
            X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size=.5)

            pbl = GraphCRF(inference_method='unary')
            svm = NSlackSSVM(pbl, C=100)
    
            start = time()
            svm.fit(X_train, y_train)
            y_pred = np.vstack(svm.predict(X_test))
            time_svm = time() - start
            accuracy = np.mean(y_pred == y_test)
            total_time = total_time + time_svm
            total_accuracy = total_accuracy + accuracy
            print("Score on this round with crf: %f (took %f seconds)"
                    % (accuracy, time_svm))
        except UnboundLocalError:
            print("pass (incomplete data)")
    else:
        X_org = list2features(train_list[i*numSlice+1:(i+1)*numSlice])
        X = np.array(X_org)
        y = list2labels_sleep(train_list[i*numSlice+1:(i+1)*numSlice])
        y_org = np.array(y)
        Y = y_org.reshape(-1, 1)
        X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
        start = time()
        y_pred = np.vstack(svm.predict(X_))
        time_svm = time() - start
        accuracy = np.mean(y_pred == Y)
        total_time = total_time + time_svm
        total_accuracy = total_accuracy + accuracy
        print("Score on this round with crf: %f (took %f seconds)"
              % (accuracy, time_svm))
print("the average accuracy is: %f , the total time taken is %f seconds"
      % (total_accuracy/iterations, total_time))
