# -*- coding: utf-8 -*-
"""
Kmeans - with Kernel
@author: Antoine
"""
import numpy as np


#Define function for K-means with standard Euclidean distance
def Kernel_eucl_Kmeans(Data = np.ndarray, K = int, maxit = 100):
    '''
     EuclKmeans: perform K means clustering using a Kernel function which adds / reorganizes features to Data,
                 but for which there is no need to implement a non-Euclidean inner product.
                 Our Kernel is still finite-dimensional so we still perform the computation of the centroids each time!
                     

    Parameters
    ----------
    Data: our set of images, (Nobs, Npix) shape
    K: number of desired clusters
    maxit: control value to limit iterations

    Returns
    -------
    centroids: ndarray(K,Npix+Nadditional_features), each (1,Npix+Nadditional_features) vector correpsonds to the centroid of one cluster.
    clusters: list[K sublists], inside the i-th of these K sublists are the 
                indices of the images associated to the i-th cluster

    '''
    [Nobs, Npix] = np.shape(Data)
    
    #ADDING FEATURES: COLOR ADDITION BY QUADRANT: images are 28x28 -> we divide 4 quadrants
    # and we "count how much color" is present in each quadrant. This feature is then added to the 
    # Data we work with -> (Nobs, Npix+Nadditional_features)
    Data_kernelized = np.empty((Nobs,Npix+4))
    Data_kernelized[0:Nobs,0:Npix] = np.copy(Data)
    for point_id, point in enumerate(Data):
        point = point.reshape((28,28))
        feat = 0
        color_sum = 0.0
        additional_features = np.zeros( (1,4) )
        for i in range(1):
            for j in range(1):
                color_sum = np.sum(point[13*i+i:13*i+13+i,13*j+j:13*j+13+j])
                additional_features[feat] = color_sum
                feat = feat+1
        Data_kernelized[point_id,Npix:Npix+4] = additional_features
    
    
    #FROM HERE ON identical as Kmeans + eucl norm. Only we operate on extended data instead of "regular" data
    
    # INITIALIZATION 
    # centroids = Data_kernelized[np.random.choice(Nobs,size = K,replace = False),:] #avoid 2 identical centroids at the start
    centroids = K_initialize(Data_kernelized, Nobs, Npix+4, K) #Kmeans++ initialization (hopefully lol)
    clusters = [[] for index in range(K)] 
    clusters = AssignCluster(Data_kernelized, Nobs, centroids, K, clusters) #Assign Clusters
    print("Centroids,clusters are initialized")
    
    #Evaluate (6.1) from Lecture Notes:
    alpha = EvaluateKmeans(Data_kernelized, Nobs, centroids, K, clusters) 
    
    # INITIALIZATION pt II: perform one step outside the while loop
    centroids = UpdateCentroids(Data_kernelized, Nobs, centroids, K, clusters) #New centroids
    clusters = AssignCluster(Data_kernelized, Nobs, centroids, K, clusters) #New cluster assignments
    beta = EvaluateKmeans(Data_kernelized, Nobs, centroids, K, clusters) #Did we improve? compare beta&alpha
    
    print("Step 1 is done outside the loop")
    count = 1 
    
    while beta < alpha and count < maxit: 
        alpha = beta
        centroids = UpdateCentroids(Data_kernelized, Nobs, centroids, K, clusters) #New centroids
        clusters = AssignCluster(Data_kernelized, Nobs, centroids, K, clusters) #New cluster assignments
        beta = EvaluateKmeans(Data_kernelized, Nobs, centroids, K, clusters) #Did we improve? compare beta&alpha
        count = count+1        
        print("Step", count, "has been completed")

    if count == maxit:
        print("Maxit reached")
    else:
        print("Non-improved (6.1) reached")
        
    return centroids, clusters


def AssignCluster(Data, Nobs, centroids, K, clusters):
    '''
    Given the updated centroids, re-assigns each image to the "good" cluster
    
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
        closest_centroid = np.argmin( np.sum( ( point - centroids) ** 2, axis=1 ) ) #axis=1 is correct
        clusters[closest_centroid].append(point_idx)
    
    return clusters

def EvaluateKmeans(Data, Nobs, centroids, K, clusters):
    '''
    Evaluation of (6.1) from lecture notes
    
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
    for clust_id, clust in enumerate(clusters):
        obj = obj +  np.sum( (Data[clust,:] - centroids[clust_id]) ** 2 )
    
    print("value of 6.1 at this step:", obj)
    return obj

# for point_idx in clust:
#     point = Data[point_idx,:]
#     obj = obj +  np.sum( (point - centroids[clust_id]) ** 2 )
# print("value of 6.1 at this step:", obj)

def UpdateCentroids(Data, Nobs, centroids, K, clusters):
    '''
    Updates position of centroids
    
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
        new_centroid = np.mean(Data[cluster,:], axis=0) #axis 0 is correct
        centroids[idx] = new_centroid
    
    return centroids



def MostRepresentedInEachCluster(centroids, clusters, real_values):
    '''
    Looks at "real" values (y vector) to find out w
    
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
    Nobs = len(real_values)
    for clust_id, clust in enumerate(clusters):
        current_chypers = real_values[clust]
        current = np.unique(current_chypers, return_counts = True)[1] #For each cluster: counts the times that each number is present in the cluster
        acc = acc + np.max(current) #We look at the most present number, and we add it 
    
    acc = acc / Nobs
    
    return acc

def K_initialize(Data, Nobs, Npix, K):
    '''
    Kmeans++ type of initialization (if it is done correctly lol)

    Parameters
    ----------
    Data
    Nobs
    Npix
    K

    Returns
    -------
    centroids: initialized centroids, using kmeans++ approach: to choose the i-th centroid we look
                at the (Nobs- (i-1)) points left. The more distant a point is from the already defined (i-1) centroids,
                the higher the probabilty to choose said point as i-th centroid.

    '''
    banned_list = []
    random_id = np.random.choice([idx for idx in range(Nobs)])
    centroids = Data[random_id,:]
    banned_list.append(random_id)
    distanceSQ = np.sum( (Data-centroids) ** 2, axis = 1 )
    
    for repetition in range(K-1):
        while random_id in banned_list:
            # print("number", random_id, "is banned! See list of banned indices:", banned_list)
            random_id = np.random.choice( [idx for idx in range(Nobs)], p = distanceSQ/np.sum(distanceSQ) )
        banned_list.append(random_id)
        centroids = np.vstack( (centroids, Data[random_id,:]) )
        distanceSQ = K_distanceSQ(Data,centroids)

    return centroids


def K_distanceSQ(Data,centroids):
    
    distance = []
    
    for point in Data:        
        closest_centroid = np.argmin( np.sum( (point - centroids) ** 2, axis=1 ) )
        dist = np.sum( (point - centroids[closest_centroid]) ** 2 )  
        distance.append(dist)
    
    return distance