# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:06:50 2022

@author: Antoine
"""
import numpy as np
import Kmeans_Kernel as KmK
import Kmeans_BuildKernelMatrix as KmBKM
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float32) / 255.0 #Normalization: useful for computation
K = 10


#Set seed -> Reproducibility, random initialization
np.random.seed(K ** 2)
BigN = 10000
Data = Xarray[:BigN,:] 
real_val = y[:BigN] 

#Construct desired Kernel matrix (options: Gauss, Poly2, Sigmoid, quadrant_col_sum, nnzcount)
KernelMat = KmBKM.BuildKernel(Data, kernel = "Sigmoid")
KernelMat = KernelMat + KernelMat.T - np.diag( np.diag(KernelMat) )
# np.save("GaussianKernel_FullDataset",KernelMat)

clust = KmK.Kmeans(KernelMat, Data, K, maxit = 120)
result = KmK.MostRepresentedInEachCluster(clust, real_val)
acc = KmK.Accuracy(clust, real_val)

print("Size of resulting clusters: ", result)
print("Accuracy: ", acc)