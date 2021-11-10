# -*- coding:utf-8 -*-
import sys
import numpy as np
from sklearn.datasets import load_digits  # Load handwritten digit recognition data
from sklearn.model_selection import train_test_split  # Training test data split
from sklearn.preprocessing import StandardScaler  # Standardized tools
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report  # Forecast result analysis tool
from sklearn.metrics import confusion_matrix  # Confusion matrix
from test import *


def small_train_eval(X_train, X_test, Y_train, Y_test):
    ss = StandardScaler()  # Invoke standardized tools
    # fit is an instance method and must be called by the instance
    X_train = ss.fit_transform(X_train)
    # The same applies to the testset
    X_test = ss.transform(X_test)

    lsvc = SVC(kernel='sigmoid')  # Use the SVC function to define the SVM model and set the kernel function at the same time
    lsvc.fit(X_train, Y_train)  # Classifier training
    Y_predict = lsvc.predict(X_test)  # predict
    # result analysis tool
    print(classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str)))
    # confusion_matrix
    confusion = confusion_matrix(Y_test, Y_predict)
    confusion_plot(confusion)


digits = load_digits()  # Call the load_digits() method
# Data dimension, 1797 pictures, 8*8
print(digits.data.shape)
# One-dimensional vector of length 64
print(digits.data[0])
# Split data
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

# print(X_train.shape)  # (1347,64)
# print(Y_test.shape)  # (450,)
# print(Y_test)
small_train_eval(X_train, X_test, Y_train, Y_test)
