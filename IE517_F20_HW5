# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:19:40 2020

@author: 倪文卿
"""

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from pandas import DataFrame

Yield= pd.read_csv('C:\hw5_treasury yield curve data.csv')
#preprocessing
Yield1 = Yield.drop(['Date'],axis=1)


from sklearn.impute import SimpleImputer
imr = SimpleImputer(missing_values=np.nan,strategy='mean')
imr=imr.fit(Yield1)
impute_data = imr.transform(Yield1)
yield1=pd.DataFrame(impute_data,columns=Yield1.columns)


#plot the boxplot
data1=Yield.iloc[:,:30]
data1.boxplot(figsize=(30,20))
plt.xlabel('Attributes',fontsize=20)
plt.ylabel('Observations',fontsize=20)
plt.show()

#Time Series plot
import matplotlib as mpl

date_str = Yield.iloc[:,0]
type(date_str)

date = pd.to_datetime(date_str, format='%m/%d/%Y') # warn:it is capital Y
plt.figure(figsize=[50,20])
SVENF01=Yield['SVENF01']
ax1 = plt.subplot(211)
ax1.plot(date, SVENF01, 'o-')
date_format=mpl.dates.DateFormatter('%Y-%m-%d')
ax=plt.gca()
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(300))
plt.ylabel('SVENF01',fontsize=50)
plt.xticks(rotation=270)

ax2 = plt.subplot(212)
ax2.plot(date, Yield['SVENF02'], 'r')
date_format=mpl.dates.DateFormatter('%Y-%m-%d')
ax=plt.gca()
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(300))
plt.xlabel('Time',fontsize=50)
plt.ylabel('SVENF02',fontsize=50)
plt.xticks(rotation=270)
plt.show()

#Heatmap
plt.figure(figsize=(20,10))
corMat = DataFrame(yield1.corr())
sns.heatmap(corMat,annot=True)

#Pairplot
sns.pairplot(yield1.iloc[:,14:27], height=5,kind='reg',diag_kind='hist')
plt.show()


# SGDregressor for orgianal data
X=Yield1.iloc[:,0:30].values
y = Yield1[['Adj_Close']].values
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42)

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler().fit(X_train)

scalery = StandardScaler().fit(y_train)
X_train = scalerX.transform(X_train)
y_train = scalery.transform(y_train)
X_test = scalerX.transform(X_test)
y_test = scalery.transform(y_test)


from sklearn import linear_model

#SGD with peanlty None
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', 
                                    penalty=None, random_state=42)
clf_sgd.fit(X_train, y_train)
from sklearn import metrics
y_test_pred = clf_sgd.predict(X_test)
y_train_pred = clf_sgd.predict(X_train)
#accuracy R2 score
print(metrics.r2_score(y_test, y_test_pred))
print(metrics.r2_score(y_train, y_train_pred))

#using CV score
cv = KFold(5, shuffle=True, random_state=42)

sgdscores_test = cross_val_score(clf_sgd, y_test, y_test_pred, cv=cv)
sgdscores_train = cross_val_score(clf_sgd, y_train, y_train_pred, cv=cv)
print ("Average SGD Accuracy of train set, test set:"
           ,[np.mean(sgdscores_train), np.mean(sgdscores_test)])

#RMSE
RMSE_Train= np.sqrt(mean_squared_error(y_train, y_train_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test, y_test_pred))
print('RMSE of SGDregressor train and test:', 
[RMSE_Train,RMSE_Test])           
        
#SGD with peanlty L2
clf_sgd1 = linear_model.SGDRegressor(loss='squared_loss', 
                                     penalty='l2', random_state=42)
clf_sgd1.fit(X_train, y_train)
y_test_pred = clf_sgd1.predict(X_test)
y_train_pred = clf_sgd1.predict(X_train)

#accuracy R2 score
print(metrics.r2_score(y_test, y_test_pred))
print(metrics.r2_score(y_train, y_train_pred))

#using CV score
cv = KFold(5, shuffle=True, random_state=42)
sgd1scores_test = cross_val_score(clf_sgd1, y_test, y_test_pred, cv=cv)
sgd1scores_train = cross_val_score(clf_sgd1, y_train, y_train_pred, cv=cv)
print ("Average SGD1 Accuracy of train set, test set using 5-fold crossvalidation:"
           ,[np.mean(sgd1scores_train), np.mean(sgd1scores_test)])


#RMSE of L2SGD
RMSE_Train= np.sqrt(mean_squared_error(y_train, y_train_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test, y_test_pred))
print('RMSE of SGDregressor1 train and test:', 
[RMSE_Train,RMSE_Test])


#SVR Linear model for Original Data
from sklearn import svm
clf_svr = svm.SVR(kernel='linear')
clf_svr.fit(X_train, y_train)
y_test_pred = clf_svr.predict(X_test)
y_train_pred = clf_svr.predict(X_train)
from sklearn import metrics
#accuracy R2 score
print('svr _R2_score of Test:', metrics.r2_score(y_test, y_test_pred))
print('svr _R2_score of Train:',metrics.r2_score(y_train, y_train_pred))

#Using CV_Score
cv = KFold(5, shuffle=True, random_state=42)
svrlinear_scores_test = cross_val_score(clf_svr, y_test, y_test_pred, cv=cv)
svrlinear_scores_train = cross_val_score(clf_svr, X_train, y_train_pred, cv=cv)
print ("Average svrlinear Accuracy of train set, test set using 5-fold crossvalidation:"
           ,[np.mean(svrlinear_scores_test), np.mean(svrlinear_scores_train)])

#RMSE of LinearSVR
RMSE_Train= np.sqrt(mean_squared_error(y_train, y_train_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test, y_test_pred))
print('RMSE of LinearSVR train and test:', 
[RMSE_Train,RMSE_Test])


#SVR Poly model
clf_svr_poly = svm.SVR(kernel='poly')
clf_svr_poly.fit(X_train, y_train)
y_test_pred = clf_svr_poly.predict(X_test)
y_train_pred = clf_svr_poly.predict(X_train)
from sklearn import metrics
#accuracy R2 score
print('svr_poly _R2_score of Test:', metrics.r2_score(y_test, y_test_pred))
print('svr_poly _R2_score of Train:',metrics.r2_score(y_train, y_train_pred))
#Use CV_Score
cv = KFold(5, shuffle=True, random_state=42)
svrpoly_scores_test = cross_val_score(clf_svr_poly, y_test, y_test_pred, cv=cv)
svrpoly_scores_train = cross_val_score(clf_svr_poly, y_train, y_train_pred, cv=cv)
print ("Average svrpoly Accuracy of train set, test set using 5-fold crossvalidation:"
           ,[np.mean(svrpoly_scores_test), np.mean(svrpoly_scores_train)])

#RMSE of PolySVR
RMSE_Train= np.sqrt(mean_squared_error(y_train, y_train_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test, y_test_pred))
print('RMSE of PolySVR train and test:', 
[RMSE_Train,RMSE_Test])


#SVR Rbf model
clf_svr_rbf = svm.SVR(kernel='rbf')
clf_svr_rbf.fit(X_train, y_train)
y_test_pred = clf_svr_rbf.predict(X_test)
y_train_pred = clf_svr_rbf.predict(X_train)
from sklearn import metrics
#accuracy R2 score
print('svr_rbf _R2_score of Test:', metrics.r2_score(y_test, y_test_pred))
print('svr_rbf _R2_score of Train:',metrics.r2_score(y_train, y_train_pred))

cv = KFold(5, shuffle=True, random_state=42)
svrrbf_scores_test = cross_val_score(clf_svr_rbf, y_test, y_test_pred, cv=cv)
svrrbf_scores_train = cross_val_score(clf_svr_rbf, y_train, y_train_pred, cv=cv)
print ("Average svrrbf Accuracy of train set, test set using 5-fold crossvalidation:"
           ,[np.mean(svrrbf_scores_test), np.mean(svrrbf_scores_train)])

#RMSE of RbfSVR
RMSE_Train= np.sqrt(mean_squared_error(y_train, y_train_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test, y_test_pred))
print('RMSE of RbfSVR train and test:', 
[RMSE_Train,RMSE_Test])
'''
#As we can see, use the Radial Basis Function (RBF) kernel, the model have much
#better performance than other two kernels, therefore,we select RBF.
'''

#PCA
X, y = Yield1.iloc[:,0:30].values, yield1[['Adj_Close']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                 random_state=42)
# standardize the features

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler().fit(X_train)

scalery = StandardScaler().fit(y_train)
X_train_std = scalerX.transform(X_train)
y_train_std = scalery.transform(y_train)
X_test_std = scalerX.transform(X_test)
y_test_std = scalery.transform(y_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,4), var_exp[0:3], alpha=0.5, align='center',
        label='individual explained variance')

plt.step(range(1,4), cum_var_exp[0:3], where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Three principal component index')
plt.legend(loc='best')
plt.show()

from sklearn.decomposition import PCA

pca =PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
All_components_EVR=pca.explained_variance_ratio_

pca = PCA(n_components=3)

X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)

#Fit SGD linear Model
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

clf_sgd = linear_model.SGDRegressor(loss='squared_loss', 
                                    penalty=None, random_state=42)
clf_sgd.fit(X_train_pca, y_train_std)


from sklearn import metrics
y_test_pca_pred = clf_sgd.predict(X_test_pca)
y_train_pca_pred = clf_sgd.predict(X_train_pca)

#accuracy R2 score
print('r2_SGD_PCA is:',metrics.r2_score(y_test_std, y_test_pca_pred))
print('r2_SGD_PCA is:',metrics.r2_score(y_train_std, y_train_pca_pred))

#using CV score
cv = KFold(5, shuffle=True, random_state=42)

sgdscores_pca_test = cross_val_score(clf_sgd, y_test_std, y_test_pca_pred, cv=cv)
sgdscores_pca_train = cross_val_score(clf_sgd,y_train_std, y_train_pca_pred, cv=cv)
print ("Average SGD Accuracy of train set, test set:"
           ,[np.mean(sgdscores_pca_train), np.mean(sgdscores_pca_test)])

#RMSE
RMSE_pca_Train= np.sqrt(mean_squared_error(y_train_std, y_train_pca_pred))
RMSE_pca_Test=np.sqrt(mean_squared_error(y_test_std, y_test_pca_pred))
print('RMSE of SGDregressor train and test:', 
[RMSE_pca_Train,RMSE_pca_Test])

#Fit SGD_L2
clf_sgd1 = linear_model.SGDRegressor(loss='squared_loss', 
                                     penalty='l2', random_state=42)
clf_sgd1.fit(X_train_pca, y_train_std)
y_test_pca_pred = clf_sgd1.predict(X_test_pca)
y_train_pca_pred = clf_sgd1.predict(X_train_pca)

#accuracy R2 score
print('r2_SGD1_PCA is:',metrics.r2_score(y_test_std, y_test_pca_pred))
print('r2_SGD1_PCA is:',metrics.r2_score(y_train_std, y_train_pca_pred))

#using CV score
cv = KFold(5, shuffle=True, random_state=42)

sgd1scores_pca_test = cross_val_score(clf_sgd1, y_test_std, y_test_pca_pred, cv=cv)
sgd1scores_pca_train = cross_val_score(clf_sgd1,y_train_std, y_train_pca_pred, cv=cv)
print ("Average SGD1 Accuracy of train set, test set:"
           ,[np.mean(sgd1scores_pca_train), np.mean(sgd1scores_pca_test)])

#RMSE
RMSE_pca_Train= np.sqrt(mean_squared_error(y_train_std, y_train_pca_pred))
RMSE_pca_Test=np.sqrt(mean_squared_error(y_test_std, y_test_pca_pred))
print('RMSE of SGDregressor train and test:', 
[RMSE_pca_Train,RMSE_pca_Test])


#Fit a SVM model to PCA
#SVR Linear model
from sklearn import svm
clf_svr = svm.SVR(kernel='linear')
clf_svr.fit(X_train_pca, y_train_std)
y_test_pca_pred = clf_svr.predict(X_test_pca)
y_train_pca_pred = clf_svr.predict(X_train_pca)
from sklearn import metrics
#accuracy R2 score
print('svr_R2_score of pca_Test:', metrics.r2_score(y_test_std, y_test_pca_pred))
print('svr_R2_score of pca_Train:',metrics.r2_score(y_train_std, y_train_pca_pred))

#Using CV_Score
cv = KFold(5, shuffle=True, random_state=42)
svrlinear_scores_pca_test = cross_val_score(clf_svr, y_test_std, y_test_pca_pred, cv=cv)
svrlinear_scores_pca_train = cross_val_score(clf_svr, y_train_std, y_train_pca_pred, cv=cv)
print ("Average svrlinear Accuracy of train set, test set using 5-fold crossvalidation:"
           ,[np.mean(svrlinear_scores_pca_test), np.mean(svrlinear_scores_pca_train)])

#RMSE of LinearSVR
RMSE_Train= np.sqrt(mean_squared_error(y_train_std, y_train_pca_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test_std, y_test_pca_pred))
print('RMSE of LinearSVR train and test:', 
[RMSE_Train,RMSE_Test])

#SVR Poly model
clf_svr_poly = svm.SVR(kernel='poly')
clf_svr_poly.fit(X_train_pca, y_train_std)
y_test_pca_pred = clf_svr_poly.predict(X_test_pca)
y_train_pca_pred = clf_svr_poly.predict(X_train_pca)
from sklearn import metrics
#accuracy R2 score
print('svr_poly _R2_score of Test:', metrics.r2_score(y_test_std, y_test_pca_pred))
print('svr_poly _R2_score of Train:',metrics.r2_score(y_train_std, y_train_pca_pred))
#Use CV_Score
cv = KFold(5, shuffle=True, random_state=42)
svrpoly_scores_test = cross_val_score(clf_svr_poly, y_test_std, y_test_pca_pred, cv=cv)
svrpoly_scores_train = cross_val_score(clf_svr_poly, y_train_std, y_train_pca_pred, cv=cv)
print ("Average svrpoly Accuracy of train set, test set using 5-fold crossvalidation:"
           ,[np.mean(svrpoly_scores_test), np.mean(svrpoly_scores_train)])

#RMSE of PolySVR
RMSE_Train= np.sqrt(mean_squared_error(y_train_std, y_train_pca_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test_std, y_test_pca_pred))
print('RMSE of PolySVR train and test:', 
[RMSE_Train,RMSE_Test])


#SVR Rbf model
clf_svr_rbf = svm.SVR(kernel='rbf')
clf_svr_rbf.fit(X_train_pca, y_train_std)
y_test_pca_pred = clf_svr_rbf.predict(X_test_pca)
y_train_pca_pred = clf_svr_rbf.predict(X_train_pca)
from sklearn import metrics
#accuracy R2 score
print('svr_rbf _R2_score of Test:', metrics.r2_score(y_test_std, y_test_pca_pred))
print('svr_rbf _R2_score of Train:',metrics.r2_score(y_train_std, y_train_pca_pred))

cv = KFold(5, shuffle=True, random_state=42)
svrrbf_scores_test = cross_val_score(clf_svr_rbf, y_test_std, y_test_pca_pred, cv=cv)
svrrbf_scores_train = cross_val_score(clf_svr_rbf, y_train_std, y_train_pca_pred, cv=cv)
print ("Average svrrbf Accuracy of train set, test set using 5-fold crossvalidation:"
           ,[np.mean(svrrbf_scores_test), np.mean(svrrbf_scores_train)])

#RMSE of RbfSVR
RMSE_Train= np.sqrt(mean_squared_error(y_train_std, y_train_pca_pred))
RMSE_Test=np.sqrt(mean_squared_error(y_test_std, y_test_pca_pred))
print('RMSE of RbfSVR train and test:', 
[RMSE_Train,RMSE_Test])

print("My name is {wenqing ni}")
print("My NetID is: {wn5}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

