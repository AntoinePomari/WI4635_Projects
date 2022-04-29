# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:06:50 2022

@author: Antoine
"""
import numpy as np
# import scipy.sparse as sps
# import matplotlib.pyplot as plt
import Kmeans_EuclDistance as KmEu
import Kmeans_Kernel_SumByQuadrant as KmKSQ
import Kmeans_BuildKernel as KmBK
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float64) / 255.0 #To float & normalize: general habit for computations
K = 10


'''
Testing Kernel Computation:

'''

BigN = 35000

Gauss_Kernel_Matrix = KmBK.BuildKernel(Xarray[0:BigN,:], kernel = "Gauss")
# plt.spy(Gauss_Kernel_Matrix)

Polynomial_Kernel_Matrix = KmBK.BuildKernel(Xarray[0:BigN], kernel = "Poly2")
# plt.spy(Polynomial_Kernel_Matrix)

Hyperbolic_Kernel_Matrix = KmBK.BuildKernel(Xarray[0:BigN,:], kernel = "Sigmoid")
# plt.spy(Hyperbolic_Kernel_Matrix)

np.save("PolKerMat_first10k",Polynomial_Kernel_Matrix)
PolKerMat = np.load("PolKerMat_first10k.npy", allow_pickle = True)



'''
Testing Classic Kmeans algorithm:
'''
#Set seed -> Reproducibility, random initialization of centroids
for tryout in range(1):
    
    np.random.seed(tryout** 3)
    
    [centers, clust] = KmEu.EuclKmeans(Xarray, K, maxit = 150, initialize = "random")
    centers = centers * 255.0
    
    result = KmEu.MostRepresentedInEachCluster(centers,clust, y)
    print("most present number in each cluster:",result)
    
    acc = KmEu.Accuracy(centers,clust, y)
    print("accuracy for this model:", acc)



'''
Testing Classic Kmeans on augmented feature vectors:
'''
for tryout in range(1):
    
    np.random.seed(2*tryout** 3)
    
    [centersK,clustK] = KmKSQ.Kernel_eucl_Kmeans(Xarray, K, maxit = 150)
    centersK = centersK* 255.0
    
    resultK = KmKSQ.MostRepresentedInEachCluster(centersK,clustK, y)
    print("For the Kernelized version:", resultK)
    
    accK = KmKSQ.Accuracy(centersK, clustK, y)
    print("For the kernelized version:",accK)









