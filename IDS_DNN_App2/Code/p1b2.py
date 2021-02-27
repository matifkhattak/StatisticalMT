from __future__ import absolute_import

import os
import sys
import gzip

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

from data_utils import get_file


seed = 2016


def get_p1_file(link):
    fname = os.path.basename(link)
    return get_file(fname, origin=link, cache_subdir='Pilot1')


def load_dataDNN3(trainingRowsShuffle=False,testRowsShuffle=False, n_cols=None):
    train_path = '/Data/researchPaperTrain.csv'  # train.csv
    test_path = '/Data/researchPaperTest.csv'  # test.csv

    traindata = pd.read_csv(train_path, header=None, engine='c')
    testdata = pd.read_csv(test_path, header=None, engine='c')

    X = traindata.iloc[:, 1:42]  # traindata.iloc[:,1:42]
    T = testdata.iloc[:, 1:42]  # testdata.iloc[:,1:42]

    scaler = Normalizer().fit(X)
    trainX = scaler.transform(X)

    scaler = Normalizer().fit(T)
    testT = scaler.transform(T)

    y_train = pd.get_dummies(traindata.iloc[:, 0]).as_matrix()
    y_test = pd.get_dummies(testdata.iloc[:, 0]).as_matrix()

    X_train = np.array(trainX)
    X_test = np.array(testT)

    return (X_train, y_train), (X_test, y_test)


def evaluateAccuracy(y_pred, y_test):
    def map_max_indices(nparray):
        maxi = lambda a: a.argmax()
        iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    # print('Final accuracy of best model: {}%'.format(100 * accuracy))
    return accuracy

def evaluate(y_pred, y_test):
    def map_max_indices(nparray):
        maxi = lambda a: a.argmax()
        iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    # print('Final accuracy of best model: {}%'.format(100 * accuracy))
    return {'accuracy': accuracy}
