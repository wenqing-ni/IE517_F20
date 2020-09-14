# -*- coding: utf-8 -*-

#Created on Wed Sep  9 11:13:35 2020

#@author: wenqing ni
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from random import uniform
import pylab
import scipy.stats as stats

Corporatebond = pd.read_csv('C:\HY_Universe_corporate bond.csv')

bonddata = np.array(Corporatebond)
bonddata= bonddata.tolist()
nrow = len(bonddata)
ncol = len(bonddata[1])

#count the number of data in different categories in each column
type = [0]*3
colCounts = []
for col in range(ncol):
    for row in bonddata:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
                     str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1
#we can see which kind of data in each column and choose suitbale visualization

#plot boxplot using column weekly_mean_volume,weekly_median_volume
plt.boxplot(x=[Corporatebond.iloc[1:,30].astype(float),
               Corporatebond.iloc[1:,31].astype(float)])
plt.show()

#The 18th and 10th column contains binary categorical variables,
#so we can count the number of data in each category 

col = 18
colData = []
for row in bonddata:
 colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*2
for elt in colData:
 catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

col = 10
colData = []
for row in bonddata:
 colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*12
for elt in colData:
 catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

#Plot the histogram for column 19 and 10
plt.figure()
sns.countplot(x='IN_ETF', data=Corporatebond, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes']) # [0,1] denotes location,[2,3]denotes label
plt.show()

plt.figure(figsize=(15,5))
sns.countplot(y='Maturity Type', data=Corporatebond)
 # [0,1] denotes location,[2,3]denotes label
plt.xticks()
plt.show()

# QQPlot for trading volume
col = 20
colData = []
for row in bonddata:
 colData.append(float(row[col]))
stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()
#we can see from the plot, the normal distribtion does not fit the 
#trading volume well.


# Read and Summarize the numeric data
Corporatebond = pd.read_csv('C:\HY_Universe_corporate bond.csv',header=None, 
                            prefix="V")
#print head and tail of data frame
print(Corporatebond.head())
print(Corporatebond.tail())
Corbond = Corporatebond.iloc[1:,19:36].astype(float)
#print summary of column 19 to 35
summary = Corbond.describe()
print(summary)

#Parallel Coordinates Graph
Corporatebond = pd.read_csv('C:\HY_Universe_corporate bond.csv',
                            header=None, prefix="V")

for i in range(1,2722):
 #assign color based on "Yes" or "No" labels
 if Corporatebond.iat[i,18] == "Yes": #get the value from i row, 18 column
     pcolor = "red"
 else:
     pcolor = "blue"
 #plot rows of data as if they were series data
 dataRow = Corporatebond.iloc[i,19:36].astype(float)
 dataRow.plot(color=pcolor)
plt.xlabel("Attribute Index")
plt.ylabel(("Attribute Values"))
plt.show()

# plot the scatterplot to measure the correlation of LIQ SCORE with
#Client_Trade_Percentage and percent_intra_dealer
Corporatebond = pd.read_csv('C:\HY_Universe_corporate bond.csv',
                            header=None, prefix="V")
dataCol19 = Corporatebond.iloc[1:1000,19].astype(float)

dataCol26 = Corporatebond.iloc[1:1000,26].astype(float)
plt.scatter(dataCol19, dataCol26)
plt.xlabel("LIQ SCORE") 
plt.ylabel("percent_intra_dealer")
plt.show()

dataCol29 = Corporatebond.iloc[1:1000,29].astype(float)
plt.scatter(dataCol19, dataCol29)
plt.xlabel("LIQ SCORE") 
plt.ylabel(("Client_Trade_Percentage"))
plt.show()
# the plot shows that perhaps the LIQ SCORE and Client_Trade_Percentageare
# have approximately postive linear relationship, while LIQ SCORE 
# and Client_Trade_Percentageare have negative linear relationship


# Correlation between variables, To improve the visualization, 
#this version dithers the points a little and makes them somewhat transparent
target = []
for i in range(1,200):
# assign 0 or 1 target value based on "Yes" or "No" labels
# and add some dither
 if Corporatebond.iat[i,18] == "Yes":
     target.append(1.0 + uniform(-0.1, 0.1))
 else:
     target.append(0.0 + uniform(-0.1, 0.1))
 #plot 23th attribute with semi-opaque points
dataRow = Corporatebond.iloc[1:200,25].astype(float)
plt.scatter(dataRow, target, alpha=0.5, s=120)# the points are 
#plotted with alpha=0.5 in order that the points are only partially opaque
plt.xlabel("total_mean_size")
plt.ylabel("In_ETF")
plt.show()

#Heatmap plot selecting last seven numeric data column
Corporatebond = pd.read_csv('C:\HY_Universe_corporate bond.csv',
                            header=None, prefix="V")
bondnew = Corporatebond.iloc[1:,29:36].astype(float)
corMat = DataFrame(bondnew.corr())
#visualize correlations using heatmap
sns_plot = sns.heatmap(corMat,annot=True)
plt.pcolor(corMat)
plt.show()

print("My name is {wenqing ni}")
print("My NetID is: {wn5}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
