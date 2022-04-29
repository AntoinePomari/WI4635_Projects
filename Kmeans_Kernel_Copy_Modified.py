import numpy as np

def Kmeans(kernelmatrix, Data = np.ndarray, K = int, maxit = 100):
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
    
    # INITIALIZATION: random cluster assignment
    clusters = [[] for index in range(K)] 
    for point_idx in range(Nobs):
        clusters[np.random.choice(K)].append(point_idx)
    # for clust in clusters:
    #     print(len(clust))
    print("Clusters are initialized randomly")

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
    
    #Assign x_j to cluster k' := argmin_k K(x_j,x_j) - 2/|C_k|sum_i in C_k K(x_j,x_i) + 1/|C_k|^2 sum_i,l in C_k K(x_i,x_l) 
    for point_idx in range(Nobs):
        clusterfun_values = np.empty(K)
        for clust_id, clust in enumerate(clusters):
            clusterfun_values[clust_id] = kernelmatrix[point_idx,point_idx] - (2/len(clust)) * np.sum(kernelmatrix[point_idx,clust]) + (1/len(clust)**2) * np.sum(kernelmatrix[np.ix_(clust,clust)])
        newclust_index = np.argmin(clusterfun_values)
        newclusters[newclust_index].append(point_idx)
    
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
         Formula (6.1): sum_k = 1 to Nclust ( sum_i in C_k norm(xi-ck)^2 ) (NB needs to be "adapted" for Kernel ofc)
    '''
    obj = 0.0
    
    for clust_id, clust in enumerate(clusters):
        for point_idx in clust:
            obj += kernelmatrix[point_idx,point_idx] - (2/len(clust)) * np.sum(kernelmatrix[point_idx,clust]) + (1/len(clust)**2) * np.sum(kernelmatrix[np.ix_(clust,clust)])
    print("loss function is now:", obj)
   
    return obj

def MostRepresentedInEachCluster(clusters, real_values):
    '''
    Looks at "real" values (y vector) to find out w
    
    Parameters
    ----------
    clusters : obtained using Kmeans algo
    real_values : vector of labels used to identify most present number in each cluster

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
