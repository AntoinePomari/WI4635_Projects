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
Xarray = Xarray.astype(np.float64) / 255.0 #Normalization: useful for computation
K = 36


#Set seed -> Reproducibility, random initialization of centroids
np.random.seed(K)

[centers, clust] = KmEu.EuclKmeans(Xarray, K, maxit = 75)
centers = centers * 255.0

result = KmEu.MostRepresentedInEachCluster(centers,clust, y)

acc = KmEu.Accuracy(centers,clust, y)

