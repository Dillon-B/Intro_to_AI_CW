#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:46:11 2021

@author: dillonbhanderi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import csv
import os
import seaborn as sns
import statsmodels.api as sm

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


# Load the dataset
fortune = pd.read_csv('datasetF1000.csv')
fortune
# Print the first 5 elements of the dataset
print(fortune.head())
# Print the count of the elements of the dataset
print(fortune.describe())


# Function to show a matrix using the pearson correlation
matrix = fortune.corr(
    method = 'pearson',  
    min_periods = 1      
)
# Print and plot the matrix
matrix = fortune.corr().round(2)
print(matrix)
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()

# Show which datapoints are being analysed
X = fortune[['market value']]
Y = fortune[['Revenue']]


# Split the dataset into testing and training data with a split of 80/20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Plot the training data
plt.xlabel('Market Value ($)')
plt.ylabel('Revenue ($)');
plt.plot(X_train, Y_train, 'o', color='blue');

# Plot the test data
plt.xlabel('Market Value ($)')
plt.ylabel('Revenue ($)')
plt.plot(X_test, Y_test, 'o', color='black')
plt.show()

# training_data, testing_data = train_test_split(fortune, test_size= 0.25, random_state=(30))

# print(f"No. of training examples: {training_data.shape[0]}")
# print(f"No. of testing examples: {testing_data.shape[0]}")

# Print the number of training and testing examples
print(f"No. of training examples: {X_train.shape[0]}")
print(f"No. of testing examples: {X_test.shape[0]}")
print(f"No. of training examples: {Y_train.shape[0]}")
print(f"No. of testing examples: {Y_test.shape[0]}")
# print(training_data.shape)
# print(testing_data.shape)

# Print the shape of the testing data
print(X_test.shape, Y_test.shape)

#perceptronFortune = Perceptron(max_iter= 60, tol=(0.001), eta0=1)
#perceptronFortune.fit(X_train,y_train)
#pred_perceptron = perceptronFortune.predict(X_train)

# Create a linear regression model
trainingModel = linear_model.LinearRegression()

# Fit the data into the model
trainingModel.fit(X_train, Y_train)

# Create a linear regression model with the data 
linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

# Predict the outcomes based on the inputted data
Y_pred = trainingModel.predict(X_test)

# Plot the model
plt.title('Linear Regression')
plt.xlabel('Market Value ($)')
plt.ylabel('Revenue ($)')
plt.plot(X_test, Y_pred)
plt.plot(X_test, Y_pred, 'o', color='black')

# Print the coeefficient
print('Coefficient: ' , trainingModel.coef_)
# Print the intercept
print('Intercept: ' , trainingModel.intercept_)
# Calculate and print the mean squared error
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
# Calculate and print the R-squared value
print('Coefficient of determination: (R^2): %.2f'
      % r2_score(Y_test, Y_pred))

