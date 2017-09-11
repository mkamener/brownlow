# Run pca on dataset as practice and see if there are any visual groupings

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

# Get filenames of csv files
filenames = glob.glob("stats_team/*.csv")

# Load data and concatenate into one large DataFrame
df = pd.concat((pd.read_csv(f, index_col=None) for f in filenames), ignore_index=True)

# Split the data into data and target values. Better visual separation is achived when team data is not included. This
# makes sense, as the individual performance is given more weight
array = df.values
X = array[:, 2:22].astype(dtype='float64')
Y = array[:, 42].astype(dtype='int')


# Normalize and shift the mean of each column from training data
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)


# Run pca algorithm
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

# Select number of values to plot at random
n_vals = 1000
idx = np.random.randint(Y.size, size=n_vals)
X = X[idx, :]
Y = Y[idx]

# Create figure
fig = plt.figure(1)
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

# Add data to the plot
ax.scatter(X[(Y == 0), 0], X[(Y == 0), 1], X[(Y == 0), 2], c='grey')    # 0 votes
ax.scatter(X[(Y == 1), 0], X[(Y == 1), 1], X[(Y == 1), 2], c='blue')    # 1 vote
ax.scatter(X[(Y == 2), 0], X[(Y == 2), 1], X[(Y == 2), 2], c='green')   # 2 votes
ax.scatter(X[(Y == 3), 0], X[(Y == 3), 1], X[(Y == 3), 2], c='red')     # 3 votes

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


