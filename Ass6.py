#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 07:15:51 2018

@author: ericzheng
"""
#Part1
#Revisit the decision tree model you built for Module 2.
#Using Iris dataset, with 90% for training and 10% for test, 
#Run in-sample and out-of-sample accuracy for 10 different samples by changing random_state from 1 to 10 in sequence.  
#Display the individual scores, calculate the mean and standard deviation of the set.
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#import iris data set

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


train_scores_set=[]
test_scores_set=[]


r = range(1,11)         #for loop range inclusive for front half, exclusive for back half
for i in r:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    
    dectree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    dectree.fit(X_train_std, y_train)

    y_train_pred = dectree.predict(X_train_std)
    y_test_pred = dectree.predict(X_test_std)
    
    train_scores_set.append(metrics.accuracy_score(y_train_pred, y_train))
    test_scores_set.append(metrics.accuracy_score(y_test_pred, y_test))
    

# =============================================================================
#     pd.set_option('display.max_rows', 16)
#     data = pd.Series(lr.coef_,index=[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']])
#     print("\n" + "Coefficients:")
#     print (data)
# =============================================================================
    
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 16)
data_train_scores_set = pd.Series(train_scores_set, index=[["1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " 10"]])
data_test_scores_set = pd.Series(test_scores_set, index=[["1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " 10"]])
    
print("Train accuracy scores:\n", data_train_scores_set)
print("\n")
print("Mean of train accuracy scores:\n{} ".format(np.mean(train_scores_set)))
print("\n")
print("Std of train accuracy scores:\n{}".format(np.std(train_scores_set)))
print("\n")

print("Test accuracy scores:\n", data_test_scores_set)
print("\n")
print("Mean of test accuracy scores:\n{} ".format(np.mean(test_scores_set)))
print("\n")
print("Std of test accuracy scores:\n{}".format(np.std(test_scores_set)))
print("\n")


#Part2
#Now rerun your performance using cross_val_scores with k-fold CV (k=10).  Evaluate on the holdout set.
#Report experiment results using template format provided and evaluate.
from sklearn.model_selection import cross_val_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


dectree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
#dectree.fit(X_train, y_train)

CV_scores = cross_val_score(estimator=dectree, X=X_train_std, y=y_train, cv=10, n_jobs=-1)

data_CV_accuracy_scores_set = pd.Series(CV_scores, index=[["1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " 10"]])

print("CV accuracy scores: \n", data_CV_accuracy_scores_set)
print("\n")
print("Mean of CV accuracy scores:\n{} ".format(np.mean(CV_scores)))
print("\n")
print("Std of CV accuracy scores:\n{}".format(np.std(CV_scores)))
print("\n")
# =============================================================================
# print('CV(cv=10) accuracy: %.5f +/- %.5f' % (np.mean(CV_scores),np.std(CV_scores)))
# print("\n")
# =============================================================================

dectree.fit(X_train_std, y_train)
y_test_pred = dectree.predict(X_test_std)
accuracy_score_out_sample = metrics.accuracy_score(y_test_pred, y_test)
print("Out-of-sample accuracy score:\n%.16f" % (accuracy_score_out_sample))
print("\n")

print("My name is Hao Zheng")
print("My NetID is: haoz7")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")