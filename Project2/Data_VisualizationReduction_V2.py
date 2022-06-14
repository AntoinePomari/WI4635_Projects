# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:57:38 2022

@author: Antoine
"""

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import tree
# from sklearn.decomposition import PCA
import pandas as pd
# Loading our data
names_list = ["age", "sex", "chest pain", "bp", "cholesterol", "blood sugar", "resting ST", "max hr", "ex-ind angina", "ex-ind ST depression", "ex ST slope", "fluoroscopy", "thalassemia", "disease" ]
heart_data = pd.read_csv("heart.csv", names = names_list)
feat_list = names_list[0:13]
response_list = [names_list[-1]]
heart_data2 = heart_data.to_numpy()



# Training a classifier
# pca = PCA(n_components=2)
# X_test_pca = pca.fit(heart_data2[:,0:13]).transform(heart_data2[:,0:13])
MyTree = tree.DecisionTreeClassifier()
MyTree = MyTree.fit(heart_data2[:,(0,4)], heart_data2[:,-1])


# Plotting decision regions
plot_decision_regions(heart_data2[:,(0,4)], heart_data2[:,-1].astype(int), clf=MyTree, legend=2)
# Adding axes annotations
plt.xlabel('age')
plt.ylabel('cholesterol')
plt.title('disease')
plt.show()
