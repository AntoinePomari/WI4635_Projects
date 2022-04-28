# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:06:50 2022

@author: Antoine
"""
import numpy as np
import Kmeans_EuclDistance as KmEu
import Kmeans_Kernel as KmK
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float16) / 255.0 #Normalization: useful for computation
K = 10


#Set seed -> Reproducibility, random initialization of centroids
np.random.seed(K)

# Choose kernel function
def kernel0(x_i,x_j): # Standard Euclidean distance
    return np.inner(x_i,x_j)

def kernel1(x_i,x_j): # Counts total non-zero elements of the image
    return np.inner(x_i,x_j) + np.count_nonzero(x_i)*np.count_nonzero(x_j)

def kernel2(x_i,x_j): # Counts total non-zero elements in each row of the image
    return np.inner(x_i,x_j) + np.sum([np.count_nonzero(x_i[28*i: 28*(i+1)-1])*np.count_nonzero(x_j[28*i: 28*(i+1)-1]) for i in range(28)])

def kernel3(x_i,x_j): # Polynomial kernel of degree r
    r = 2
    return (1 + np.inner(x_i,x_j)) ** r

def kernel4(x_i,x_j): # Gaussian kernel with parameter gamma
    gamma = 0.5
    return np.exp(- gamma * np.sum( (x_i - x_j) ** 2 ) )

clust = KmK.Kmeans(kernel0, Xarray[:1000,:], K, maxit = 75)

result = KmK.MostRepresentedInEachCluster(clust, y)

acc = KmK.Accuracy(clust, y)

