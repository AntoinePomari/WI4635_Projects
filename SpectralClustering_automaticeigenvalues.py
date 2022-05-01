import numpy as np
import scipy.linalg as spla
import Kmeans_Kernel as KmK
import Kmeans_BuildKernelMatrix as KmBKM
import Kmeans_EuclDistance as KmEU


def SpecClust(Data, K, Kmeans_maxit = 100, kernel = "Gauss", normalize = False):
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
    SimilMat_kernel : The default is "Gausss". Init type for similarity matrix routine called by this function. 
    normalize : The defalut is False. If False Algorithm 24 from Lect Notes, if True Algorithm 25 from Lect Notes
    Returns
    -------
    centers: centroids of Kmeans ran on rows of matrix H
    clusters: clusters corresponding to each centroid
        
    '''
    if not normalize:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN
        L = KmBKM.BuildKernel(Data, kernel) #upper triangular W
        L = L + L.T - np.diag(np.diag(L)) # correct shape W
        L = np.diag( np.sum(L, axis = 1) ) - L #actual Laplacian L
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS
        eigenvalues, H = spla.eigh(L, subset_by_index=[0,K])
        #RUN Kmeans ON ROWS OF MATRIX H
        [centers, clusters] = KmEU.EuclKmeans(H, K, maxit = Kmeans_maxit)
    else:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN, NORMALIZE IT
        L = KmBKM.BuildKernel(Data, kernel) #upper triangular W
        L = L + L.T - np.diag(np.diag(L)) # correct shape W
        Normalizer = np.diag( 1/np.sqrt(np.sum(L, axis = 1)) ) #Normalizer Factor D^(-1/2)
        L = np.eye(L.shape) - Normalizer @ L @ Normalizer #actual Normalized Laplacian
        
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS
        eigenvalues, H = spla.eigh(L, subset_by_index=[0,K])
        #RUN Kmeans ON ROWS OF MATRIX H
        [centers, clusters] = KmEU.EuclKmeans(H, K, maxit = Kmeans_maxit)
    
    return [centers, clusters]