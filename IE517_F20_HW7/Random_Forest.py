# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:37:58 2020

@author: 倪文卿
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
fault=pd.read_csv('C:\ccdefault.csv')
X= fault.iloc[:,:-1].values
y=fault.iloc[:,24].values
feat_labels = fault.columns[:-1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1,stratify=y)

In_sample_score = []
for n in range(10,31):
    forest=RandomForestClassifier(n_estimators=n,random_state=1)
#in-sample accuracy
    scores = cross_val_score(estimator=forest, 
                         X=X_train, 
                         y=y_train, 
                         cv=10, 
                         scoring='roc_auc')
    print("In-sample ROC AUC: %0.2f (+/- %0.2f) " 
          % (scores.mean(), scores.std()))    
    In_sample_score.append(scores.mean())



forest=RandomForestClassifier(n_estimators=30,random_state=1)
forest.fit(X_train,y_train)

feat_labels = fault.columns[:-1]
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                        feat_labels[indices[f]],
                        importances[indices[f]]))
    
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is {wenqing ni}")
print("My NetID is: {wn5}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
