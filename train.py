# Fetch imports, 2 different classifiers
import csv
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Set filenames
filenameTrain = 'C:/Users/Huzaifa/Documents/ML/TitanicSurvivalPredictor/train.csv'
filenameTest = 'C:/Users/Huzaifa/Documents/ML/TitanicSurvivalPredictor/test.csv'

# Initialize and set y column
y = []
y = pd.read_csv(filenameTrain,usecols=[1])


def get_data(filename, boolean):
    df = pd.read_csv(filename, header = 0)

    # Remove non numbers
    df = df._get_numeric_data()

    # Put the numeric column names in a python list
    numeric_headers = list(df.columns.values)

    # Drop unncessary columns from dataframe
    df = df.drop(columns=['Age'])
    df = df.drop(columns=['PassengerId'])

    # If training set, also remove the y column
    if(boolean == True):
        df = df.drop(columns=['Survived'])

    # Create numpty_array
    df = df.reset_index()
    numpy_array = df.as_matrix()

    return numpy_array

# Set data in X and test set
X = get_data(filenameTrain, True)
Xtest = get_data(filenameTest, False)

"""
print(X)
print(y)
print(Xtest)
print(y.values.ravel())
"""
# Test SVM classifier
clf = svm.SVC()
clf.fit(X, y.values.ravel())
print(clf.predict(Xtest))

# Test neural network classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y.values.ravel())
print(clf.predict(Xtest))                         