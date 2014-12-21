import numpy as np
import matplotlib.pyplot as plt

from time import time

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

from pystruct.models import GraphCRF, LatentGraphCRF
from pystruct.learners import NSlackSSVM, LatentSSVM

import csv
from itertools import chain

# Load the scikit-learn digits classification dataset.
digits = load_digits()
X, y_org = digits.data, digits.target


with open('/Users/Zhanghongzhuo/Desktop/EECS6893/TrainingSet/TrainingSet.csv','rb') as f:
	 reader = csv.reader(f)
	 train_list = list(reader)

def list2features(data_list):
    return [num2features(data_list, i+1) for i in range(len(data_list)-1)]

def list2labels_sleep(data_list):
    return [1 if int(data_list[i+1][2])==3004 else 0 for i in range(len(data_list)-1)]
#    return [str(int(data_list[i+1][2])==3004) for i in range(len(data_list)-1)]

def list2labels_tv(data_list):
    return [1 if int(data_list[i+1][2])==5102 else -1 for i in range(len(data_list)-1)]

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

X_org = list2features(train_list)
X = np.array(X_org)
y = list2labels_sleep(train_list)
y_org = np.array(y)
Y = y_org.reshape(-1, 1) 

X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]

X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size=.3)

pbl = GraphCRF(inference_method='unary')
svm = NSlackSSVM(pbl, C=100)


start = time()
svm.fit(X_train, y_train)
time_svm = time() - start
y_pred = np.vstack(svm.predict(X_test))
print("Score with pystruct crf svm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_svm))
