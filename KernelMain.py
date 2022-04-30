# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:06:50 2022

@author: Antoine
"""
import numpy as np
import Kmeans_Kernel_Copy_Modified as KmK
import Kmeans_BuildSimilartyMatrices as KmBSM

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float32) / 255.0 #Normalization: useful for computation
K = 10


#Set seed -> Reproducibility, random initialization
np.random.seed(K ** 2)
BigN = 30000
Data = Xarray[:BigN,:]
real_val = y[:BigN]

#Construct desired Kernel matrix
KernelMat = KmBSM.BuildKernel(Data, kernel = "Gauss")
KernelMat = KernelMat + KernelMat.T - np.diag( np.diag(KernelMat) )
# np.save("GaussianKernel_FullDataset",KernelMat)

#NB Computations for Polynomial Kernel are very slow! -> Better keep maxit below 25 lol
#NB Computations for Gaussian Kernel are significantly faster (not superfast tho)
#NB Computations for Sigmoid Kernel are also very slow
clust = KmK.Kmeans(KernelMat, Data, K, maxit = 120)
result = KmK.MostRepresentedInEachCluster(clust, real_val)
acc = KmK.Accuracy(clust, real_val)
