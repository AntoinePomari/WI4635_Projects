# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:21:10 2022

@author: Antoine
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
import plotly.express as px

#Import the data with numerical values scaled or with numerical values 
scaled_data = pd.read_csv("heart_data_scaled.csv")
dummy_data = pd.read_csv("heart_data_dummies.csv")
scaled_data = scaled_data.to_numpy()
dummy_data = dummy_data.to_numpy()
scaled_data = scaled_data[:,1:]
dummy_data = dummy_data[:,1:]


"""
Dimensionality reduction #1: Classification tree
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
clf = clf.fit(scaled_data[:,(4,7)],scaled_data[:,-1])
#Plot
fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of Classification Tree')
# Set-up grid for plotting.
xx, yy = make_meshgrid(scaled_data[:,4], scaled_data[:,7])
plt.axis()
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(scaled_data[:,4], scaled_data[:,7], c=scaled_data[:,-1], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('Age (scaled)')
ax.set_xlabel('Cholesterol (scaled)')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()

"""
Dimensionality reduction #2: PCA
"""
pca = PCA(n_components=2)
X_test_pca = pca.fit(scaled_data[:,:-1]).transform(scaled_data[:,:-1])
X_std_pca = pca.fit(dummy_data[:,:-1]).transform(dummy_data[:,:-1])

model = svm.SVC(kernel='linear')
clf = model.fit(X_test_pca, scaled_data[:,-1])

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC using PCA')
# Set-up grid for plotting.
xx, yy = make_meshgrid(X_test_pca[:,0], X_test_pca[:,1])
plt.axis()
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X_test_pca[:,0], X_test_pca[:,1], c=scaled_data[:,-1], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('Principal component #2')
ax.set_xlabel('Principal component #1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
fig.grid()
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2)

target_classes = range(0, 2)
colors = ("blue", "red")
markers = ("^", "s")

for target_class, color, marker in zip(target_classes, colors, markers):
    ax1.scatter(
        x=X_test_pca[target_class, 0],
        y=X_test_pca[target_class, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

    ax2.scatter(
        x=X_std_pca[:, 0],
        y=X_std_pca[:, 1],
        color=color,
        label=f"class {target_class}",
        alpha=0.5,
        marker=marker,
    )

ax1.set_title("Stdized dataset after PCA")
ax2.set_title("Stdized + dummy coded dataset after PCA")

for ax in (ax1, ax2):
    ax.set_xlabel("1st principal component")
    ax.set_ylabel("2nd principal component")
    ax.legend(loc="upper right")
    ax.grid()



