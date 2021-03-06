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


def load_data(trainingRowsShuffle=False,testRowsShuffle=False, n_cols=None):
    #train_path = get_p1_file('http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/P1B2.train.csv')
    #test_path = get_p1_file('http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P1B2/P1B2.test.csv')
    #test_path = get_p1_file('C:/Users/faqeerrehman/MSU/Research/CancerPrediction/ScientificSWTesting/Data/Pilot1/P1B2.testSmallDS1Row.csv')


    #dataFilePath = get_p1_file('C:/Users/faqeerrehman/MSU/CourseWork/SecondSemester/MachineLearning/CourseProject/IntrustionDetectionSystem/Pilot1/P1B2Tests/Data/TrainCIDDS_week1Processed.csv') #test.csv
    train_path = 'C:/Users/faqeerrehman/MSU/OneDrive - Montana State University/CourseWork/SecondSemester/MachineLearning/CourseProject/IntrustionDetectionSystem/Pilot1/P1B2Tests/Data/TrainCIDDS_week1Processed.csv'  # train.csv
    test_path = 'C:/Users/faqeerrehman/MSU/OneDrive - Montana State University/CourseWork/SecondSemester/MachineLearning/CourseProject/IntrustionDetectionSystem/Pilot1/P1B2Tests/Data/TestCIDDS_week1Processed.csv'  # test.csv

    usecols = list(range(n_cols)) if n_cols else None

    df_train = pd.read_csv(train_path, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_path, engine='c', usecols=usecols)

    if trainingRowsShuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
    if testRowsShuffle:
        df_test = df_test.sample(frac=1, random_state=seed)
    numberOfIndepdendentFeatures = 32
    X_train = df_train.iloc[:, :numberOfIndepdendentFeatures].as_matrix()  # Faqeer df_train.iloc[:, :32].as_matrix()
    X_test = df_test.iloc[:, :numberOfIndepdendentFeatures].as_matrix()  # Faqeer df_test.iloc[:, :32].as_matrix()

    y_train = pd.get_dummies(df_train[['classLabel']]).as_matrix()
    y_test = pd.get_dummies(df_test[['classLabel']]).as_matrix()

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
