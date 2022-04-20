# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
# import matplotlib as plt
# import numba
import sklearn
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Turn it into numpy object. For personal preference
Xarray = X.to_numpy()
ind = 15000
image = Xarray[ind,:].reshape((28,28))
number = y[ind]
print(image,number)
image = image.astype(int)
# image = 255 - image Line to invert constrast scale if needed 
# Visualize some of the images
img = Image.fromarray(image)
img.show()
# img.save('first_image_test.png')


# Define number of clusters (=number of digits)
K = 10

#Define function for K-means with standard Euclidean distance
def EuclKmeans(Data = np.ndarray, K = int, maxit = 1000):
    '''
    EuclKmeans: perform K means clustering using classic Euclidean norm
    
    INPUT:
        Data: our set of images (used as vectors: ideally (N_obs, 784) shape)
        clusters: number of desired clusters
        maxit: control value to limit iterations
    
    OUTPUT:
        centroids: ndarray(K,784), dtype = int32, each (1,784) vector correpsonds to the centroid of one cluster.
        clusters: ndarray(Nobs, 1), dtype = int32, each position is the "clustered" number associated to the image
    '''
    #Define a counter and some useful numbers: number of observations, number of pixels / image, "Assign": defines the class of each number
    count = 0
    [Nobs, Npix] = np.shape(Data)
#   Initialization: define centroids and assign each image to a cluster
    centroids = np.empty((K,Npix),dtype = int)
    clusters = AssignCluster(Data, Nobs, centroids, K)
    alpha = EvaluateKmeans(Data, Nobs, centroids, K, clusters)
    # Initialization II: perform one step outside the while loop
    centroids = OneStepKmeans(Data, Nobs, K, clusters)
    beta = EvaluateKmeans(Data, Nobs, centroids, K, clusters)
    count = 1
    
    while beta < alpha and count < maxit:
        alpha = beta
        clusters = AssignCluster(Data, Nobs, centroids, K)
        centroids = OneStepKmeans(Data, Nobs, K, clusters)
        beta = EvaluateKmeans(Data, Nobs, centroids, K, clusters)
        count = count+1
    
    return centroids, clusters





def initialize(Data = np.ndarray, clusters = int):
    
    
    
    
    return centroids



def AssignCluster(Data, Nobs, centroids, K):
    clusters = np.zeros((Nobs,1), dtype = int)
    for ii in range(Nobs):
        value = (Data[ii,:]-centroids[0,:]) @ (Data[ii,:]-centroids[0,:])
        for kk in range(K-1):
            tmp = (Data[ii,:]-centroids[kk+1,:]) @ (Data[ii,:]-centroids[kk+1,:])
            if tmp < value:
                value = tmp
        clusters[ii] = value
    del value
    return clusters

def EvaluateKmeans(Data, Nobs, centroids, K, clusters):
    obj = 0
    for kk in range(K):
        for ii in range(Nobs):
            if clusters[ii] == kk:
                obj = obj + (Data[ii,:]-centroids[kk,:])@(Data[ii,:]-centroids[kk,:])
    return obj

def OneStepKmeans(Data, Nobs, Npix, K, clusters):
    centroids = np.zeros(shape = (K,Npix), dtype = int)
    SizeCluster = np.bincount(clusters).astype(int)
    for kk in range(K):
        for ii in range(Nobs):
            if clusters[ii] == kk:
                centroids[kk,:] = centroids[kk,:] + Data[ii,:]
        centroids[kk,:] = (1 / SizeCluster[kk]) * centroids[kk,:]
    del SizeCluster
    return centroids






