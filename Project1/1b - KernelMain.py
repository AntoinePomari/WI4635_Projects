# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:06:50 2022

@author: Antoine
"""
import numpy as np
import Kmeans_Kernel as KmK
import Kmeans_BuildKernelMatrix as KmBKM
from sklearn.datasets import fetch_openml


## Load data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float32) / 255.0 #Normalization: useful for computation


## Initialization steps
np.random.seed(12)
# Number of clusters 
K = 10

# Number of data points to be used: ideally 70000
BigN = 3000
indices = np.random.choice(Xarray.shape[0], size = BigN, replace = False)
Data = Xarray[indices,:]
real_val = y[indices] 

# Construct desired Kernel matrix 
# Kernel (and hyperparameter) options: Gauss (with gamma), Poly (with r), Sigmoid (with alpha and C), 
# ..., quadrant_col_sum, nonzerototal, nonzerorows, nonzerorows_cols

KernelMat = KmBKM.BuildKernel(Data, kernel = "Gauss")
KernelMat = KernelMat + KernelMat.T - np.diag( np.diag(KernelMat) )
# np.save("GaussianKernel_FullDataset",KernelMat)

## Run algorithm
clust = KmK.Kmeans(KernelMat, Data, K, maxit = 120)
clust_with_real_indices = [[] for kk in range(K)]
for cluster_id, cluster in enumerate(clust):
    for element in cluster:
        clust_with_real_indices[cluster_id].append(indices[element])
del cluster, cluster_id, element, clust

## Compute and print result
result = KmK.MostRepresentedInEachCluster(clust_with_real_indices, real_val)
acc = KmK.Accuracy(clust_with_real_indices, real_val)

print("Size of resulting clusters: ", result)
print("Accuracy: ", acc)
