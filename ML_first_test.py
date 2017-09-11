# Initial tests on data set to see baseline effectiveness

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Get filenames of csv files
filenames = glob.glob("stats_team/*.csv")

# Select only a few years for initial testing (last three years)
filenames = filenames[-3:]

# Load data and concatenate into one large DataFrame
df = pd.concat((pd.read_csv(f, index_col=None) for f in filenames), ignore_index=True)

# for f in filenames:
#     print(f)
#
# print(df.shape)
# print(df.columns.values)

# Set brownlow votes to be boolean
df.Brownlow = df.Brownlow > 0

# Split the data into train (including cv) and validation sets
array = df.values
X = array[:, 2:42].astype(dtype='float64')
Y = array[:, 42].astype(dtype='bool')
validation_size = 0.20
seed = 42796
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# Normalize and shift the mean of each column from training data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# Test options and evaluation metric. Use f1 score, as there are relatively few vote winners
seed = 1203812
n_splits = 10
scoring = 'f1'

# First test algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
