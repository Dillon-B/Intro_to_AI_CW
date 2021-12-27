#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:46:11 2021

@author: dillonbhanderi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.utils
import io
import csv
import os

from tensorflow import keras
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping  
from tensorflow.keras.layers import Dropout 

fortune = pd.read_csv('datasetF1000.csv')
fortune

X = fortune[['market value'].replace({'\$':'', ',': ''}, regex=True)]
Y = fortune[['Revenue'].replace({'\$':'', ',': ''}, regex=True)]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# training_data, testing_data = train_test_split(fortune, test_size= 0.25, random_state=(30))

# print(f"No. of training examples: {training_data.shape[0]}")
# print(f"No. of testing examples: {testing_data.shape[0]}")
print(f"No. of training examples: {X_train.shape[0]}")
print(f"No. of testing examples: {X_test.shape[0]}")
print(f"No. of training examples: {Y_train.shape[0]}")
print(f"No. of testing examples: {Y_test.shape[0]}")
# print(training_data.shape)
# print(testing_data.shape)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#perceptronFortune = Perceptron(max_iter= 60, tol=(0.001), eta0=1)
#perceptronFortune.fit(X_train,y_train)
#pred_perceptron = perceptronFortune.predict(X_train)

trainingModel = linear_model.LinearRegression()

trainingModel.fit(X_train, Y_train)

linear_model.LinearRegression(copy_X=True, fit_itercept=True, n_jobs=None, normalize=False)

Y_pred = trainingModel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names=Y))


