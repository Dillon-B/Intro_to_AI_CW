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

fortune = pd.read_csv('datasetF1000.csv', nrows= 1000)
fortune

X = fortune[['market value']]
y = fortune[['Revenue']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=(45))

training_data, testing_data = train_test_split(fortune, test_size= 0.25, random_state=(30))

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")
print(f"No. of training examples: {X_train.shape[0]}")
print(f"No. of testing examples: {X_test.shape[0]}")
print(f"No. of training examples: {y_train.shape[0]}")
print(f"No. of testing examples: {y_test.shape[0]}")
print(training_data.shape)
print(testing_data.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)