# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:40:52 2020

@author: wenqing ni
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pylab
import scipy.stats as stats
from pandas import DataFrame
from sklearn.metrics import r2_score
House =pd.read_csv('C:\housing2.csv')

House.fillna(0,inplace=True)

summary = House.describe()
print(summary)
# we can see from the summary that first fourteen columns have means 
#rather smaller than later columns,so we can separate them into two parts
#plot the boxplot
data1=House.iloc[:,:14]
data1.boxplot(figsize=(10,10))
plt.xlabel('Attributes',fontsize=20)
plt.ylabel('Observations',fontsize=20)
plt.show()

data2=House.iloc[:,14:27]
data2.boxplot(figsize=(10,10))
plt.xlabel('Attributes',fontsize=20)
plt.ylabel('Observations',fontsize=20)
plt.show()

#visualize correlations using heatmap
plt.figure(figsize=(20,10))
corMat = DataFrame(House.corr())
sns.heatmap(corMat,annot=True)

#visualization using pairplot
sns.pairplot(House.iloc[:,14:27], height=5,kind='reg',diag_kind='hist')
plt.show()


#Solving regression for regression parameters with gradient descent
class LinearRegressionGD(object):
    
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self


    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return self.net_input(X)

X = House[['RM']].values
y = House['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)


sns.reset_orig() # resets matplotlib style
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

array=np.array([5.0])
num_rooms_std = sc_x.transform(array.reshape(-1,1))
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % \
sc_y.inverse_transform(price_std))


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
lin_regplot(X, y, slr)

plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()


#1. Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = House.iloc[:, :-1].values
y = House['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


#plot the Linear residuals
plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals using Linear Regression')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()
#Slope: 2.449 Intercept: 41.088 adding noise
print('The Slope of Linear Regression is: %.3f' % slr.coef_[0])
print(slr.coef_)
print('The Intercept of Linear Regression is: %.3f' % slr.intercept_)
# Compute the MSE
from sklearn.metrics import mean_squared_error
print('MSE of Linear (train: %.3f, test: %.3f)' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))
# the R^2 after adding R^2 train: 0.766, test: 0.777

print('R^2 of Linear (train: %.3f, test: %.3f)' % 
(r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))
#Mean Squared Error (MSE), which is simply the averaged value 
#of the SSE cost that we minimized to fit the linear regression model


#2. Ridge regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
X = House.iloc[:, :-1].values
y = House['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

#plot the residuals of Ridge
plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#Compute the slope and Intercept of Ridge
print('Slope of Ridge: %.3f' % reg.coef_[0])
print(reg.coef_)
print('Intercept of Ridge: %.3f' % reg.intercept_)
#Compute the MSE of Ridge
print('MSE of Ridge (train: %.3f, test: %.3f)' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))

#Compute the R^2 of Ridge
print('R^2 of Ridge (train: %.3f, test: %.3f)' % 
(r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))

# Find the best alpha of Ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
X = House.iloc[:, :-1].values
y = House['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

#Plot Ridge tuning parameter alpha
ridge = Ridge(max_iter=10000, normalize=True)
coefs = []
alphas = 10**np.linspace(6,-2,50)*0.5
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    
np.shape(coefs)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha of Ridge Regression')
plt.ylabel('weights')
#On the right hand side we can see the null model, 
#containing only the intercept. This is caused by the very high penalty

from sklearn import linear_model
reg = linear_model.RidgeCV(alphas = 10**np.linspace(6,-2,50)*0.5)
reg.fit(X_train, y_train)
print('the optimal Alpha is:' ,reg.alpha_)
regbest=Ridge(alpha=reg.alpha_)
regbest.fit(X_train, y_train)
print("Optimal Ridge mse = ",mean_squared_error(y_test, regbest.predict(X_test)))
print("Optimal Ridge coefficients:")
best_model_coe=pd.Series(regbest.coef_, 
 index=pd.DataFrame(X,columns=House.columns[0:26]).columns)
print(best_model_coe)
print('Optimal R^2 of Ridge:%.3f' % 
r2_score(y_test, regbest.predict(X_test)))

      
#3.Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
X = House.iloc[:, :-1].values
y = House['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

#plot the residuals
plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#Compute the Slope,Intrecept,MSE and R^2
print('Slope of Lasso: %.3f' % lasso.coef_[0])
print(lasso.coef_)
print('Intercept of Lasso: %.3f' % lasso.intercept_)
print('MSE of Lasso (train: %.3f, test: %.3f)' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))
print('R^2 of Lasso (train: %.3f, test: %.3f)' % 
(r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))

#Find the Optimal α of Lasso
from sklearn.linear_model import LassoCV
from sklearn import linear_model
X = House.iloc[:, :-1].values
y = House['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

#Plot Ridge tuning parameter alpha
lasso = Lasso(max_iter=10000, normalize=True)
coefs = []
alphas = 10**np.linspace(6,-2,50)*0.5
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    
np.shape(coefs)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha of Lasso')
plt.ylabel('weights')

lasso = LassoCV(alphas = 10**np.linspace(6,-2,50)*0.5)
lasso.fit(X_train, y_train)
print('the optimal Alpha of Lasso is:' ,lasso.alpha_)
lassobest=Lasso(alpha=lasso.alpha_)
lassobest.fit(X_train, y_train)
print("Optimal Lasso mse = ",mean_squared_error(y_test, lassobest.predict(X_test)))
print("Optimal Lasso coefficients:")
best_model_coe=pd.Series(lassobest.coef_, 
 index=pd.DataFrame(X,columns=House.columns[0:26]).columns)
print(best_model_coe)
print('Optimal R^2 of Ridge:%.3f' % 
r2_score(y_test, lassobest.predict(X_test)))

print("My name is {wenqing ni}")
print("My NetID is: {wn5}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
