# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:54:37 2022

@author: Jim
"""
import numpy as np
from random import randint, choice

def Kmeans(kernel, Data = np.ndarray, K = int, maxit = 100):
    '''
     KernelKmeans: perform K means clustering using a custom kernel
     
    Parameters
    ----------
    kernel: desired kernel function from (1,Npix) x (1,Npix) to the positive reals
    Data: our set of images, (Nobs, Npix) shape
    K: number of desired clusters
    maxit: control value to limit iterations

    Returns
    -------
    clusters: list[K sublists], inside the i-th of these K sublists are the 
                indices of the images associated to the i-th cluster

    '''
    [Nobs, Npix] = np.shape(Data)
    
    # INITIALIZATION
    clusters = [[] for index in range(K)] 
    for point_idx, point in enumerate(Data):
        clusters[np.random.choice(K)].append(point_idx)
    print("Clusters are initialized randomly")
    
    # Compute the kernel of each point combination beforehand
    #Method 1: fromfunction
    #kernelmatrix = np.fromfunction(np.vectorize(lambda i,j:kernel(Data[i,:],Data[j,:])),(Nobs,Nobs),dtype=np.float32)
    
    #Method 2: for loop
    kernelmatrix = np.zeros((Nobs,Nobs),dtype=np.float32)
    for i in range(Nobs):
        kernelmatrix[i:,i] = kernel(Data[i,:],Data[i:,:])
    kernelmatrix = kernelmatrix + kernelmatrix.T - np.diag(kernelmatrix.diagonal())
    print("Kernel matrix computed")
    
    #Evaluate (6.1) from Lecture Notes with kernel instead of inner product:
    alpha = EvaluateKernelKmeans(kernelmatrix, Data, Nobs, K, clusters) 
    
    # INITIALIZATION pt II: perform one step outside the while loop
    clusters = AssignClusterKernel(kernelmatrix, Data, Nobs, K, clusters) #New cluster assignments
    beta = EvaluateKernelKmeans(kernelmatrix, Data, Nobs, K, clusters) #Did we improve? compare beta&alpha
    
    print("Step 1 is done outside the loop")
    count = 1 
    
    while beta < alpha and count < maxit: 
        alpha = beta
        clusters = AssignClusterKernel(kernelmatrix, Data, Nobs, K, clusters) #New cluster assignments
        beta = EvaluateKernelKmeans(kernelmatrix, Data, Nobs, K, clusters) #Did we improve? compare beta&alpha
        count = count+1        
        print("Step", count, "has been completed")

    if count == maxit:
        print("Maxit reached")
    else:
        print("Non-improved (6.1) reached")
        
    return clusters

def AssignClusterKernel(kernelmatrix, Data, Nobs, K, clusters):
    '''
    Updates the best cluster for each image
    
    Parameters
    ----------
    kernelmatrix
    Data
    Nobs
    K
    clusters

    Returns
    -------
    newclusters:
        Reassigned clusters for each image

    '''
    newclusters = [[] for index in range(K)]
    maxclusterlength = max(len(clust) for clust in clusters)
    
    # Normalized submatrix of kernelmatrix to allow for easy computation
    clustsubmatrices = np.zeros((K, maxclusterlength, maxclusterlength))
    for k in range(K):
        clustsubmatrices[k,:len(clusters[k]),:len(clusters[k])] = kernelmatrix[np.ix_(clusters[k],clusters[k])]
    
    for point_idx, point in enumerate(Data):
        #bestcluster = np.argmin([kernelmatrix[point_idx,point_idx] - 2* np.sum(kernelmatrix[point_idx, clusters[k]]) / len(clusters[k]) \
        #            + np.sum(clustsubmatrices[k,:,:]) / len(clusters[k])**2 for k in range(K)])
        
        disttoclusters = [kernelmatrix[point_idx,point_idx]*len(clusters[k])**2 \
            - 2*len(clusters[k])* np.sum(kernelmatrix[point_idx, clusters[k]]) + np.sum(clustsubmatrices[k,:,:])  for k in range(K)]
        
        bestcluster = np.argmin(disttoclusters)
        newclusters[bestcluster].append(point_idx)
    
    for k in range(K):
        if len(newclusters[k]) == 0:
            print('Empty cluster ' + str(k) + ' !')
    
    return newclusters

def EvaluateKernelKmeans(kernelmatrix, Data, Nobs, K, clusters):
    '''
    Evaluation of (6.1) from lecture notes for a custom kernel
    
    Parameters
    ----------
    kernelmatrix
    Data
    Nobs
    K
    clusters
    
    Returns
    -------
    obj : float
         Formula (6.1):
    '''
    obj = 0.0
    for clust_id, clust in enumerate(clusters):
        if len(clust)!=0:
            clustkernelmatrix = kernelmatrix[np.ix_(clust,clust)]
            obj = obj +  np.trace(clustkernelmatrix) - np.sum(clustkernelmatrix)/len(clust)
        

    return obj

def MostRepresentedInEachCluster(clusters, real_values):
    '''
    Looks at "real" values (y vector) to find out w
    
    Parameters
    ----------
    clusters : obtained using Kmeans algo
    real_values : vector of labels used to identify each cluster etc

    Returns
    -------
    result : returns most represented cypher & number of appearences, for each cluster.

    '''
    result = [[] for index in range(len(clusters))]
    for clust_id, clust in enumerate(clusters):
        current_chypers = real_values[clust]
        current_count = np.unique(current_chypers, return_counts = True)
        max_count = np.max(current_count[1])
        id_of_max = np.argmax(current_count[1])
        most_present_cypher = current_count[0][id_of_max]
        result[clust_id].append((max_count, most_present_cypher))
    
    return result


def Accuracy(clusters, real_values):
    '''
    Parameters
    ----------
    clusters :
    real_values : labels of the data

    Returns
    -------
    acc : accuracy of clustering obtained

    '''
    
    acc = 0.0
    Nobs = len(real_values)
    for clust_id, clust in enumerate(clusters):
        current_chypers = real_values[clust]
        current = np.unique(current_chypers, return_counts = True)[1] #For each cluster: counts the times that each number is present in the cluster
        acc = acc + np.max(current) #We look at the most present number, and we add it 
    
    acc = acc / Nobs
    
    return acc