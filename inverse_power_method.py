# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 04:39:49 2022

@author: 31649
"""



import numpy as np
import scipy.linalg as spla
import Kmeans_EuclDistance as KmEu
import Kmeans_BuildSimilartyMatrices as KmBSM


def inverse_power(A, mu, v, niter, tol):
    for i in range(niter)
        Av = np.dot(A,v)
        Av_norm = np.linalg.norm(Av)
        v = Av/Av_norm
        mu_new = np.dot(np.transpose(v),np,dot(A,v))
        if (abs(mu_new - mu)/mu_new) < tol:
            return mu_new, v
        mu = mu_new
    
    return mu_new, v


def find_eigen_vectors(L, k):
    smallest_k = np.zeros((k, L.shape[0]))
    mu = 0
    v = np.ones(L.shape[0]) 
    for i in range(0, k):
        L = L - mu*np.dot(v, np.transpose(v))
        mu, v = inverse_power(L, mu, v, niter, tol)
        smallest_k[i,:] = v
    
    return smallest_k
        


def SpecClust(Data, K, Kmeans_maxit = 100, Kmeans_initialize = "random", SimilMat_kernel = "Gauss", normalize = False):
    '''
    Spectral Clustering, with automatic computation of eigenvalues. 
    Steps: (if normalize = True, same but on normalized Laplacian)
    Compute Similarity Matrix W and Laplacian L.
    Construct H: columns correspond to K minimal eigenvalues of L.
    Use Kmeans algorithm on the rows of matrix H.
    
    Parameters
    ----------
    Data : Our usual set of images
    K : # of clusters
    Kmeans_maxit : The default is 100. Max number of iterations for Kmeans routine called by this function.
    Kmeans_initialize : The default is "random". Init type for Kmeans routine called by this function. 
    SimilMat_kernel : The default is "Gausss". Init type for similarity matrix routine called by this function. 
    normalize : The defalut is False. If False Algorithm 24 from Lect Notes, if True Algorithm 25 from Lect Notes
    Returns
    -------
    centers: centroids of Kmeans ran on rows of matrix H
    clusters: clusters corresponding to each centroid
        
    '''
    if not normalize:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN
        L = KmBSM.BuildKernel(Data, kernel = SimilMat_kernel) #upper triangular W
        L = L + L.T - np.diag(np.diah(L)) # correct shape W
        L = np.diag( np.sum(L, axis = 1) ) - L #actual Laplacian L
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS WITH OUR OWN SOLVER
        L_inv = np.linalg.pinv(L)
        H = find_eigen_vectors(L_inv, K)
        #RUN Kmeans ON ROWS OF MATRIX H
        [centers, clusters] = KmEu.EuclKmeans(H, K, maxit = Kmeans_maxit, initialize = Kmeans_initialize)
    else:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN, NORMALIZE IT
        L = KmBSM.BuildKernel(Data, kernel = SimilMat_kernel) #upper triangular W
        L = L + L.T - np.diag(np.diah(L)) # correct shape W
        Normalizer = np.diag( 1/np.sqrt(np.sum(L, axis = 1)) ) #Normalizer Factor D^(-1/2)
        L = np.eye(L.shape) - Normalizer @ L @ Normalizer #actual Normalized Laplacian
        
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS WITH OUR OWN SOLVER
        L_inv = np.linalg.pinv(L)
        H = find_eigen_vectors(L_inv, K)
        #RUN Kmeans ON ROWS OF MATRIX H
        [centers, clusters] = KmEu.EuclKmeans(H, K, maxit = Kmeans_maxit, initialize = Kmeans_initialize)
    
    return [centers, clusters]







        
        
