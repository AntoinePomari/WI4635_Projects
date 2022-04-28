# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sps
import timeit


def BuildKernel(Data = np.ndarray, kernel = "Gauss", gamma = 0.1, alpha = 0.5, C = 1.0):
    '''
    Builds Kernel matrix to be used in Kernel-type K-means
    
    Parameters
    ----------
    Data : (Nobs,Npix) our set of images
    type : 
        "Gauss": Gaussian Kernel -> K(x,y) = exp(-gamma * x^T y)
        "Poly2": Polynomial Kernel, degree 2 -> K(x,y) = (x^Ty + 1)^2 -> Parameter TO-BE-OPTIMIZED
        "Sigmoid": sigmoid Kernel -> K(x,y) = tanh(alpha * x^Ty + C), alpha = 0.1, C = 1 -> Parameters TO-BE-OPTIMIZED
        Sigmoid Kernel should be similar to simple functioning neural network

    Returns
    -------
    K : Kernel matrix for the selected Kernel.
        NB symmetry -> only upper triangular part will be nonzero
        Shape: (Nobs,Nobs), Format: CSC
    '''
    #INITIALIZE
    [Nobs, Npix] = np.shape(Data)
    K = [[] for index in range(Nobs)]
    
    #KERNEL COMPUTATION USING LISTS
    #Faster than using ndarrays, not sure if "fast" in general tho
    start = timeit.default_timer()
    if kernel == "Gauss":
        for ii, vector in enumerate(Data):
            # tmp = np.exp( -gamma * np.sum( (vector - Data[ii:Data.shape[0],:]) ** 2, axis = 1 ))
            K[ii].append( np.exp( -gamma * np.sum( (vector - Data[ii:Data.shape[0],:]) ** 2, axis = 1 ) ) )
    
    elif kernel == "Poly2":
        for ii, vector in enumerate(Data):
            tmp = np.sum( vector * Data[ii:Data.shape[0],:], axis = 1 ) 
            K[ii].append( (tmp + np.ones(len(tmp))) ** 2)
            # for yy in range(xx,K.shape[1]):
            #     K[xx,yy] = ((Data[xx,:] @ Data[yy,:]) + 1 ) ** 2
    
    elif kernel == "Sigmoid":
        for ii, vector in enumerate(Data):
            tmp = np.sum( vector * Data[ii:Data.shape[0],:], axis = 1 )
            K[ii].append( np.tanh(alpha * tmp + C * np.ones(len(tmp))) )
            # for yy in range(xx,K.shape[1]):
            #     K[xx,yy] = np.tanh(alpha * Data[xx,:] @ Data[yy,:] + C)
    else:
        raise ValueError("Enter available Kernel type") 

    stop = timeit.default_timer()
    print("Time to get Kernel:", stop - start)

        
     
    #TURN LIST OF LISTS INTO TRIANGULAR MATRIX -> VERY SLOW AS Nobs INCREASES!!!
    # start = timeit.default_timer()
    # Kmat = np.empty((0,Nobs))
    # # Kmatsparse = sps.csc_matrix((0,Nobs))
    # for element in K:
    #     tmp = np.append( np.zeros(Nobs-len(element[0])), element )
    #     Kmat = np.vstack( (Kmat, tmp) )
    #     # Sparsetmp = sps.csc_matrix( np.append(np.zeros(Nobs-len(element[0])),element ) )
    #     # Kmatsparse = sps.vstack( (Kmatsparse, Sparsetmp) )
    # Kmat = sps.csc_matrix(Kmat)
    # stop = timeit.default_timer()
    # print("Time to post process:", stop - start)
    
    # return Kmat
    return K