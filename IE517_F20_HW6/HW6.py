# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:47:36 2020

@author: 倪文卿
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


fault=pd.read_csv('C:\ccdefault.csv')

#Imbalance Data
from sklearn.utils import resample
X= fault.iloc[:,:-1].values
y=fault.iloc[:,24].values

print('Number of class 1 samples before:',X[y == 1].shape[0])

X_upsampled, y_upsampled = resample(X[y == 1],
 y[y == 1],
 replace=True,
 n_samples=X[y == 0].shape[0],
 random_state=1)
print('Number of class 1 samples after:',X_upsampled.shape[0])
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))
X=X_bal
y=y_bal



test_score = []
for n in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=n,stratify=y)
    pipe_dt = make_pipeline(StandardScaler(),
                            DecisionTreeClassifier(criterion='entropy', 
                              max_depth=3, random_state=n))

    pipe_dt.fit(X_train, y_train)
    y_pred = pipe_dt.predict(X_test)
    test_score.append(accuracy_score(y_test, y_pred))

train_score = []
for n in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=n,stratify=y)
    pipe_dt = make_pipeline(StandardScaler(),
                            DecisionTreeClassifier(criterion='entropy', 
                              max_depth=3, random_state=n))

    pipe_dt.fit(X_train, y_train)
    y_pred = pipe_dt.predict(X_train)
    train_score.append(accuracy_score(y_train, y_pred))

print('test_mean=',np.mean(test_score))
print('test_sd=',np.std(test_score))
print('train_mean=',np.mean(train_score))
print('train_sd=',np.std(train_score))

from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1,stratify=y)
pipe_dt = make_pipeline(StandardScaler(),
                            DecisionTreeClassifier(criterion='entropy', 
                              max_depth=3, random_state=1))

scores = cross_val_score(estimator=pipe_dt,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
 np.std(scores)))


#out_of_sample
scores_test = cross_val_score(estimator=pipe_dt,X=X_test,y=y_test,cv=10, n_jobs=1)
print('CV test accuracy scores: %s' % scores_test)
print('mean of test score: %.3f' %np.mean(scores_test))
print('standard deviation of test score: %.3f' %np.std(scores_test))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1,stratify=y)
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
pipe_dt = make_pipeline(StandardScaler(),
                            DecisionTreeClassifier(criterion='entropy', 
                              max_depth=3, random_state=1))
train_sizes, train_scores, test_scores =learning_curve(estimator=pipe_dt,
                                X=X_train,
                                y=y_train,
                                train_sizes=np.linspace(0.1, 1.0, 10),
                                cv=10,
                                n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean,color='blue', 
         marker='o',markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
 color='green', linestyle='--',
 marker='s', markersize=5,
 label='validation accuracy')
plt.fill_between(train_sizes,
 test_mean + test_std,
 test_mean - test_std,
 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()

print("My name is {wenqing ni}")
print("My NetID is: {wn5}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
