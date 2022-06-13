import numpy as np
import scipy.sparse as sps
import Kmeans_BuildKernelMatrix as KmBKM
import Kmeans_EuclDistance as KmEU
import Kmeans_Kernel as KmK

def SpecClust(Data, K, Kmeans_maxit = 100, Kmeans_initialize = "random", Laplacian_kernel = "Gauss", normalize = False, TOL = 1e-5):
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
    TOL: The default is 1e-5. Cutoff value for entries of Laplacian Matrix
    Kmeans_initialize: The default is "random". Initialization method for Kmeans algorithm
    
    Returns
    -------
    centers: centroids of Kmeans ran on rows of matrix H
    clusters: clusters corresponding to each centroid
        
    '''
    if not normalize:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN
        L = KmBKM.BuildKernel(Data, kernel = Laplacian_kernel, gamma = TOL)  #upper triangular W
        L = L + L.T - np.diag(np.diag(L)) # correct shape W
        L = np.diag( np.sum(L, axis = 1) ) - L #actual Laplacian L
        print("Laplacian obtained")
        #CUTOFF SMALL ENTRIES OF LAPLACIAN
        L[np.abs(L)<TOL] = 0
        # L = sps.csc_matrix(L)
        print("Values below TOL are cutoff")
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS. SA: smaller algebraic value of eigenvals
        vals, vecs = sps.linalg.eigsh(L, k = K+1, which = 'SA') #Symmetry -> we can use spsl.eigsh function!
        print("First K(+1) vecs obtained, now launching Kmeans")
        # vecs = vecs / np.linalg.norm(vecs, axis = 0) #Normalizing vecs -> again computational habit
        norms_of_rows = np.linalg.norm(vecs,axis = 1)
        for ii in range(vecs.shape[0]):
            vecs[ii,:] = vecs[ii,:]/norms_of_rows[ii]
        #RUN Kmeans ON ROWS OF MATRIX H -> ignore first vec, related to lambda = 0
        [centers, clusters] = KmEU.EuclKmeans(vecs[:,1:], K, maxit = Kmeans_maxit, initialize = Kmeans_initialize)
    else:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN, NORMALIZE IT
        L = KmBKM.BuildKernel(Data, kernel = Laplacian_kernel) #upper triangular W
        L = L + L.T - np.diag(np.diag(L)) # correct shape W
        Normalizer = np.diag( 1 / np.sqrt( np.sum(L, axis = 1) ) ) #Degree Matrix
        L = np.eye(L.shape[0]) - Normalizer @ L @ Normalizer #actual Normalized Laplacian
        del Normalizer
        print("Nomralized Laplacian obtained")
        #CUTOFF SMALL ENTRIES OF LAPLACIAN
        L[np.abs(L)<TOL] = 0
        L = sps.csc_matrix(L)
        print("Values below TOL are cutoff")
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS
        vals, vecs = sps.linalg.eigsh(L, k = K+1, which = 'SA') #Symmetry -> we can use spsl.eigsh function!
        print("First K(+1) vecs obtained, now launching Kmeans")
        vecs = vecs / np.linalg.norm(vecs, axis = 0) #Normalizing vecs -> again computational habit
        #RUN Kmeans ON ROWS OF MATRIX H
        [centers, clusters] = KmEU.EuclKmeans(vecs[:,1:], K, maxit = Kmeans_maxit, initialize = Kmeans_initialize)
    
    return [centers, clusters]




def SpecClust_Kernel(Data, K, Kmeans_maxit = 100,  Kmeans_kernel = "Gauss", Laplacian_kernel = "Gauss", normalize = False, TOL = 1e-5):
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
    TOL: The default is 1e-5. Cutoff value for entries of Laplacian Matrix
    Kmeans_initialize: The default is "random". Initialization method for Kmeans algorithm
    
    Returns
    -------
    centers: centroids of Kmeans ran on rows of matrix H
    clusters: clusters corresponding to each centroid
        
    '''
    if not normalize:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN
        L = KmBKM.BuildKernel(Data, kernel = Laplacian_kernel, gamma = 1e-4 )  #upper triangular W
        L = L + L.T - np.diag(np.diag(L)) # correct shape W
        L = np.diag( np.sum(L, axis = 1) ) - L #actual Laplacian L
        print("Laplacian obtained")
        #CUTOFF SMALL ENTRIES OF LAPLACIAN
        L[np.abs(L)<TOL] = 0
        L = sps.csc_matrix(L)
        print("Values below TOL are cutoff")
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS. SA: smaller algebraic value of eigenvals
        vals, vecs = sps.linalg.eigsh(L, k = K+1, which = 'SA') #Symmetry -> we can use spsl.eigsh function!
        print("First K(+1) vecs obtained, now launching Kmeans")
        norms_of_rows = np.linalg.norm(vecs,axis = 1)
        for ii in range(vecs.shape[0]):
            vecs[ii,:] = vecs[ii,:]/norms_of_rows[ii]
        # vecs = vecs / norms_of_rows
        # vecs = vecs / np.linalg.norm(vecs, axis = 1) #Normalizing vecs -> again computational habit -> NORMALIZE THE ROWS BECAUSE YOU CLUSTER THE ROWS!!
        #RUN KERNEL Kmeans ON ROWS OF MATRIX H -> ignore first vec, related to lambda = 0
        KernelMat = KmBKM.BuildKernel(vecs[:,1:], kernel = Kmeans_kernel, gamma = 1e-4) #For Kernel Kmeans -> build Kernel Matrix
        clusters = KmK.Kmeans(KernelMat, vecs[:,1:], K, maxit = Kmeans_maxit)
    else:
        #BUILD SIMILARITY MATRIX, COMPUTE LAPLACIAN, NORMALIZE IT
        L = KmBKM.BuildKernel(Data, kernel = Laplacian_kernel) #upper triangular W
        L = L + L.T - np.diag(np.diag(L)) # correct shape W
        Normalizer = np.diag( 1 / np.sqrt( np.sum(L, axis = 1) ) ) #Degree Matrix
        L = np.eye(L.shape[0]) - Normalizer @ L @ Normalizer #actual Normalized Laplacian
        del Normalizer
        print("Nomralized Laplacian obtained")
        #CUTOFF SMALL ENTRIES OF LAPLACIAN
        L[np.abs(L)<TOL] = 0
        L = sps.csc_matrix(L)
        print("Values below TOL are cutoff")
        #FIND K SMALLEST EIGENVALUES & CORRESPONDING EIGENVECTORS
        vals, vecs = sps.linalg.eigsh(L, k = K+1, which = 'SA') #Symmetry -> we can use spsl.eigsh function!
        print("First K(+1) vecs obtained, now launching Kmeans")
        vecs = vecs / np.linalg.norm(vecs, axis = 0) #Normalizing vecs -> again computational habit
        #RUN KERNEL Kmeans ON ROWS OF MATRIX H -> ignore first vec, related to lambda = 0
        KernelMat = KmBKM.BuildKernel(vecs[:,1:], kernel = Kmeans_kernel, gamma = 1e-4) #For Kernel Kmeans -> build Kernel Matrix
        clusters = KmK.Kmeans(KernelMat, vecs[:,1:], K, maxit = Kmeans_maxit)
    
    return clusters