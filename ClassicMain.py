# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:06:50 2022

@author: Antoine
"""
import numpy as np
import Kmeans_EuclDistance as KmEu
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float64) / 255.0 #To float & normalize: general habit for computations
K = 10


'''
 Classic Kmeans algorithm:
'''
# Initialization options: ++, random (for Kmeans++ and uniform random sampling among data points respectively)
[centers, clust] = KmEu.EuclKmeans(Xarray, K, maxit = 150, initialize = "random")
centers = centers * 255.0

result = KmEu.MostRepresentedInEachCluster(centers,clust, y)
print("Most present number in each cluster:",result)

acc = KmEu.Accuracy(centers,clust, y)
print("Accuracy for this model:", acc)











