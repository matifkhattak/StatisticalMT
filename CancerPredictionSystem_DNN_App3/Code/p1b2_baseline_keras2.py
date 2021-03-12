from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

import tensorflow as tf
import random as rn

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint
from keras.regularizers import l2
import csv
import p1b2 as p1b2

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
# Printing complete marix / full numpy array
import sys
np.set_printoptions(threshold=sys.maxsize)

#Writing in Excel
import xlwt
from xlwt import Workbook
BATCH_SIZE = 64 #2400
NB_EPOCH = 20#100#20                 # number of training epochs
PENALTY = 0.00001             # L2 regularization penalty
ACTIVATION = 'sigmoid'
outputLayerActivation = 'softmax'
FEATURE_SUBSAMPLE = None
DROP = None

L1 = 1024
L2 = 512
L3 = 256
L4 = 0
LAYERS = [L1, L2, L3, L4]

class BestLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.val_losses = ''
        self.val_accuracies = ''
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_accuracy', 0)), self.best_val_acc)

        self.val_losses = self.val_losses + ',' + str(round(float(logs.get('val_loss', 0)),4))
        self.val_accuracies = self.val_accuracies + ',' + str(round(float(logs.get('val_accuracy', 0)),4))


def extension_from_parameters():
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(ACTIVATION)
    ext += '.B={}'.format(BATCH_SIZE)
    ext += '.D={}'.format(DROP)
    ext += '.E={}'.format(NB_EPOCH)
    if FEATURE_SUBSAMPLE:
        ext += '.F={}'.format(FEATURE_SUBSAMPLE)
    for i, n in enumerate(LAYERS):
        if n:
            ext += '.L{}={}'.format(i+1, n)
    ext += '.P={}'.format(PENALTY)
    return ext

def recordSoftmaxProbabilities(X_train = None, y_train = None, X_test = None, y_test = None, DeterministicResults = False,fileName=None,classLabel=''):
    if(DeterministicResults):
        __setSession()
        # Workbook is created
    wb = Workbook()

    # =====create sheet1 and add headers====
    sheetToRecordTrainValidTestLossAndAccuracy = wb.add_sheet('Sheet 1')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 0, 'Class Label = ' + classLabel)
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 1, 'ValidationLoss')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 2, 'TestLoss')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 3, 'Accuracy')


    for x in range(1,101):
        print('mainToRecordTrainValidateTestLosses:Run===>',x)
        if X_train is None:
            (X_train, y_train), (X_test, y_test) = p1b2.load_data(False, False)
            k = 1; #np.random.randint(1, 100)
            b = 10; #np.random.randint(100)
            for i in range(X_train.shape[1]):
               X_train[:, i] = X_train[:, i] * k + b
               X_test[:, i] = X_test[:, i] * k + b

            #(X_train, y_train), (X_test, y_test) = p1b2.load_dataWithImportantFeatures(False, False)
            #(X_train, y_train), (X_test, y_test) = p1b2.load_dataWithImportantFeatures(n_cols=FEATURE_SUBSAMPLE)

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        model = Sequential()
        model.add(Dense(LAYERS[0], input_dim=input_dim,
                        activation=ACTIVATION,
                        kernel_regularizer=l2(PENALTY),
                        activity_regularizer=l2(PENALTY)))

        for layer in LAYERS[1:]:
            if layer:
                if DROP:
                    model.add(Dropout(DROP))
                model.add(Dense(layer, activation=ACTIVATION,
                                kernel_regularizer=l2(PENALTY),
                                activity_regularizer=l2(PENALTY)))

        #model.add(Dense(output_dim, activation=ACTIVATION))
        model.add(Dense(output_dim, activation=outputLayerActivation)) #Added by Faqeer
        #Next the model would be compiled. Compiling the model takes two parameters: optimizer and loss
        #https: // towardsdatascience.com / building - a - deep - learning - model - using - keras - 1548ca149d37
        #https://towardsdatascience.com/sequence-models-by-andrew-ng-11-lessons-learned-c62fb1d3485b
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print("Model Summary:", model.summary())

        ext = extension_from_parameters()
        checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)
        history = BestLossHistory()

        trainingResults = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NB_EPOCH,
                  validation_split=0.2,
                  callbacks=[history, checkpointer])

        y_pred = history.best_model.predict(X_test)
        predictedOutputs = model.predict_classes(X_test)

        #print("Y_Pred = " , y_pred)
        #print("PredictedOutputs = ", predictedOutputs)
        scores = p1b2.evaluate(y_pred, y_test)


        testResults = model.evaluate(X_test,y_test,batch_size=BATCH_SIZE)
        print('Evaluation on test data:', scores)
        print('Loss: ', np.amin(trainingResults.history['loss']),'Accuracy: ',np.amin(trainingResults.history['accuracy']),'Val_Loss: ',np.amin(trainingResults.history['val_loss']),'Val_Accuracy :',np.amin(trainingResults.history['val_accuracy']))
        print('best_val_loss={:.5f} best_val_acc={:.5f}'.format(history.best_val_loss, history.best_val_acc))
        print('Test Scores [Test Loss] = ', testResults[0])

        print('Evaluation on test data:', scores)

        #print('Best model saved to: {}'.format('model'+ext+'.h5'))

        #======Save Training loss,Validation(Best model) loss, test loss and Accuracy
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 0, str(round(np.amin(trainingResults.history['loss']),3)))
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 1, str(round(history.best_val_loss,3)))
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 2, str(round(testResults[0],3)))
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 3, str(scores))
        #===========================================================================
        #=====Save Instance level outputs against each experiment/iteration over for Each Class=====
        # =====create sheet2 and add headers====
        sheetToRecordInstanceLevelOutput = wb.add_sheet('IterationNo'+str(x)+'_ClassLabel='+classLabel)
        sheetToRecordInstanceLevelOutput.write(0, 0, 'Test Data Outputs for Class Label = ' + classLabel)
        sheetToRecordInstanceLevelOutput.write(1, 0, 'InputFeatures')
        sheetToRecordInstanceLevelOutput.write(1, 1, 'Expected_OR_ActualOutput')
        sheetToRecordInstanceLevelOutput.write(1, 2, 'PredictedOutput')
        sheetToRecordInstanceLevelOutput.write(1, 3, 'Probabilities')
        sheetToRecordInstanceLevelOutput.write(1, 4, 'MaxProbability')
        startRowToBeInserted = 2
        for x in range(X_test.shape[0]):
            # print("ddd = ", X_test[x])
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 0, 'Test Data Input Features')# str(X_test[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 1, str(y_test[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 2, str(predictedOutputs[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 3, str(y_pred[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 4, str(np.amax(y_pred[x])))
            startRowToBeInserted = startRowToBeInserted + 1
        #==============================================================================
        submission = {'scores': scores,
                      'model': model.summary(),
                      'submitter': 'Developer Name' }

    if fileName!=None:
        wb.save(fileName) #.xls
    else:
        wb.save("Default.xls")  # .xls
    # print('Submitting to leaderboard...')
    # leaderboard.submit(submission)
    __resetSeed()
    #return history.best_model
    return None# history.best_model#scores


def __resetSeed():
    np.random.seed()
    rn.seed()

def __setSession():
    # Sets session for deterministic results
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development


    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    import os
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    from keras import backend as K
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    # tf.global_variables_initializer()
    tf.compat.v1.set_random_seed(1234)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

    # Fixed by Faqeer ur Rehman on 24 Nov 2019
    #K.set_session(sess)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == '__main__':
    #mainOrigional()
    recordSoftmaxProbabilities(None,None,None,None,DeterministicResults = False, fileName= "Source.xls")
