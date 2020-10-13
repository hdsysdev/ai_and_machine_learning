import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

data = pandas.read_csv("data.csv")

x, y = make_classification(n_samples=5000, n_features=10,
                           n_classes=2,
                           n_clusters_per_class=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33)
lsvc = LinearSVC(verbose=0)
print(lsvc)

LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=10000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)

lsvc.fit(xtrain, ytrain)
score = lsvc.score(xtrain, ytrain)
print("Score: ", score)

