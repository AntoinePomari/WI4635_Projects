# -*- coding: utf-8 -*-
import numpy as np


#Define function for K-means with standard Euclidean distance
def EuclKmeans(Data = np.ndarray, K = int, maxit = 100):
    '''
    EuclKmeans: perform K means clustering using classic Euclidean norm
    INPUT:
        Data: our set of images, (Nobs, Npix) shape
        K: number of desired clusters
        maxit: control value to limit iterations
    
    OUTPUT:
        centroids: ndarray(K,Npix), each (1,Npix) vector correpsonds to the centroid of one cluster.
        clusters: contains the list of indices associated to each cluster
    '''
    [Nobs, Npix] = np.shape(Data)
    
    # INITIALIZATION
    
    #(random samples W/OUT replacement: avoid 2 identical centroids at the start)
    centroids = Data[np.random.choice(Nobs,size = K,replace = False),:] 
    # centroids = Data[[0,1,2,3,4,5,7,13,15,17],:]
    clusters = [[] for index in range(K)] 
    clusters = AssignCluster(Data, Nobs, centroids, K, clusters)
    print("Centroids,clusters are initialized")
    
    #Evaluate (6.1) from Lecture Notes:
    alpha = EvaluateKmeans(Data, Nobs, centroids, K, clusters) 
    
    # INITIALIZATION pt II: perform one step outside the while loop
    centroids = UpdateCentroids(Data, Nobs, centroids, K, clusters)
    beta = EvaluateKmeans(Data, Nobs, centroids, K, clusters)
    
    print("Step 1 is done outside the loop")
    
    count = 1 #We already performed one iteration outside the loop
    
    while beta < alpha and count < maxit: 
        #NB WHY NOT UPDATE CLUSTERS AFTER CENTROIDS?
        alpha = beta
        clusters = AssignCluster(Data, Nobs, centroids, K, clusters) #Better centroids -> new assignments
        centroids = UpdateCentroids(Data, Nobs, centroids, K, clusters) #New assignment -> we redefine the centroids
        beta = EvaluateKmeans(Data, Nobs, centroids, K, clusters) #New centroids -> will they improve the clustering?
        count = count+1        
        print("Step", count, "has been completed")

    if count == maxit:
        print("Maxit reached")
    else:
        print("Non-improved (6.1) reached")
        
    return centroids, clusters


def AssignCluster(Data, Nobs, centroids, K, clusters):
    '''
    Parameters
    ----------
    Data
    Nobs
    centroids
    K

    Returns
    -------
    clusters:
        Given centroids, assigns clusters

    '''
    clusters = [[] for index in range(K)]
    for point_idx, point in enumerate(Data):
        closest_centroid = np.argmin( np.sqrt( np.sum( ( point - centroids) ** 2, axis=1) ) )
        clusters[closest_centroid].append(point_idx)
    
    return clusters

def EvaluateKmeans(Data, Nobs, centroids, K, clusters):
    '''
    Parameters
    ----------
    Data
    Nobs
    centroids
    K
    clusters
    
    Returns
    -------
    obj : float
         Formula (6.1): Euclidean distance-type of objective function for current centroids and cluster assignment.
    '''
    obj = 0.0
    # for clust_id, clust in enumerate(clusters):
    #     print("current value of (6.1):",obj)
    #     for point_idx, point in enumerate(Data):
    #         if PointInCluster(point_idx,clust):
    #             obj = obj + np.sqrt( np.sum((point - centroids[clust_id]) ** 2))
    for clust_id, clust in enumerate(clusters):
        # print("current value of (6.1):",obj)
        for point_idx in clust:
            point = Data[point_idx,:]
            obj = obj + np.sqrt( np.sum((point - centroids[clust_id]) ** 2))
    return obj

def UpdateCentroids(Data, Nobs, centroids, K, clusters):
    '''
    Parameters
    ----------
    Data
    Nobs
    centroids (old)
    K
    clusters (new)

    Returns
    -------
    centroids : ndarray, dtype = int
        If there is improvement, we perform a new assignment -> we must update the new centroids

    '''
    for idx, cluster in enumerate(clusters):
        # print("current index:", idx)
        new_centroid = np.mean(Data[cluster,:], axis=0)
        centroids[idx] = new_centroid
    
    return centroids


def MostRepresentedInEachCluster(centroids, clusters, real_values):
    '''

    Parameters
    ----------
    centroids & clusters : obtained using Kmeans algo
    real_values : vector of labels used to identify each cluster etc

    Returns
    -------
    result : returns most represented cypher & number of appearences, for each cluster.

    '''
    result = [[] for index in range(np.shape(centroids)[0])]
    for clust_id, clust in enumerate(clusters):
        current_chypers = real_values[clust]
        current_count = np.unique(current_chypers, return_counts = True)
        max_count = np.max(current_count[1])
        id_of_max = np.argmax(current_count[1])
        most_present_cypher = current_count[0][id_of_max]
        result[clust_id].append((max_count, most_present_cypher))
    
    return result


def Accuracy(centroids, clusters, real_values):
    '''
    Parameters
    ----------
    centroids :
    clusters :
    real_values : labels of the data

    Returns
    -------
    acc : accuracy of clustering obtained

    '''
    
    acc = 0.0
    
    for clust_id, clust in enumerate(clusters):
        current_chypers = real_values[clust]
        current = np.unique(current_chypers, return_counts = True)[1]
        acc = acc + np.max(current)
    
    acc = acc / 70000
    return acc