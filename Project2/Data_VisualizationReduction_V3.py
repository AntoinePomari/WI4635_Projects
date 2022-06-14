# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:33:05 2022

@author: Antoine
"""

import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

names_list = ["age", "sex", "chest pain", "bp", "cholesterol", "blood sugar", "resting ST", "max hr", "ex-ind angina", "ex-ind ST depression", "ex ST slope", "fluoroscopy", "thalassemia", "disease" ]
heart_data = pd.read_csv("heart.csv", names = names_list)
feat_list = names_list[0:13]
response_list = [names_list[-1]]
#DEFINE A HELPER FUNCTION TO TREAT THE DATA
def ModifyData(heart):
    #FIRST: DEFINE THE CATEGORICAL VARIABLES AS CATEGORICAL
    heart["chest pain"] = heart["chest pain"].astype('category')
    heart["resting ST"] = heart["resting ST"].astype('category')
    heart["ex ST slope"] = heart["ex ST slope"].astype('category')
    heart["thalassemia"] = heart["thalassemia"].astype('category')

    # #SECOND: DEFINE THE BINARY VARIABLES AS INTEGERS
    heart["sex"] = heart["sex"].astype(int)
    heart["blood sugar"] = heart["blood sugar"].astype(int)
    heart["ex-ind angina"] = heart["ex-ind angina"].astype(int)
    heart["disease"] = heart["disease"].astype(int)

    # #THIRD: PERFORM MINMAX SCALING ON THE CONTINUOUS / DISCRETE VARIABLES (EXCEPT THE ONE ABOUT # OF VESSELS -> DOES NOT SEEM LIKE A SENSIBLE CHOICE..)
    heart["age"] = ( heart["age"] - heart["age"].min() ) / heart_data["age"].max()
    heart["bp"] = ( heart["bp"] / heart["bp"].min() ) / heart["bp"].max()
    heart["cholesterol"] = ( heart["cholesterol"] / heart["cholesterol"].min() ) / heart["cholesterol"].max()
    heart["max hr"] = ( heart["max hr"] / heart["max hr"].min() ) / heart["max hr"].max()
    
    return heart



heart_data = ModifyData(heart_data)
heart_data2 = heart_data.to_numpy() #WE LOSE THE DISTINCTION..

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


#Plotting the decision surfaces
# Train
clf = tree.DecisionTreeClassifier()
clf = clf.fit(heart_data[["chest pain", "thalassemia"]], heart_data["disease"])

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of Classification Tree')
# Set-up grid for plotting.
xx, yy = make_meshgrid(heart_data["chest pain"], heart_data["thalassemia"])
plt.axis()
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(heart_data["chest pain"], heart_data["thalassemia"], c=heart_data["disease"], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()





