# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:01:18 2022

@author: Antoine
"""
import pandas as pd
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn import svm
import graphviz as gv
import numpy as np
import matplotlib.pyplot as plt

names_list = ["age", "sex", "chest pain", "bp", "cholesterol", "blood sugar", "resting ST", "max hr", "ex-ind angina", "ex-ind ST depression", "ex ST slope", "fluoroscopy", "thalassemia", "disease" ]
heart_data = pd.read_csv("heart.csv", names = names_list)
feat_list = names_list[0:13]
response_list = [names_list[-1]]
heart_data2 = heart_data.to_numpy()

# CURRENT_DIR.parent.mkdir(parents=True, exist_ok=True)  
# heart_data.to_csv(CURRENT_DIR)   #Permission denied error -> can't save the data with the "good" col names.. not such a big thing tho


"""
Dimensionality reduction #1: Classification tree on untouched data

"""
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


# #Export a tree plot and see what happens
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(heart_data2[:,0:13], heart_data2[:,-1])
# #Plotting the tree
# dot_data = tree.export_graphviz(clf,feature_names=feat_list, out_file=None, filled=True, rounded=True,  special_characters=True)  
# graph = gv.Source(dot_data)  
# graph 
# graph.render("Heart disease") 

#Plotting the decision surfaces
# Train
clf = tree.DecisionTreeClassifier()
clf = clf.fit(heart_data2[:,(0,4)], heart_data2[:,-1])

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of Classification Tree')
# Set-up grid for plotting.
xx, yy = make_meshgrid(heart_data2[:,0], heart_data2[:,4])
plt.axis()
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(heart_data2[:,0], heart_data2[:,4], c=heart_data2[:,-1], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()



"""
Dimensionality reduction #2: PCA on untouched data
"""
pca = PCA(n_components=2)
X_test_pca = pca.fit(heart_data2[:,0:13]).transform(heart_data2[:,0:13])

model = svm.SVC(kernel='linear')
clf = model.fit(X_test_pca, heart_data2[:,-1])

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC using PCA')
# Set-up grid for plotting.
xx, yy = make_meshgrid(X_test_pca[:,0], X_test_pca[:,1])
plt.axis()
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X_test_pca[:,0], X_test_pca[:,1], c=heart_data2[:,-1], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('Principal component #2')
ax.set_xlabel('Principal component #1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()









