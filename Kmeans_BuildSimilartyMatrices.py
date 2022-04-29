# -*- coding: utf-8 -*-
import numpy as np
# import scipy.sparse as sps
import timeit


def BuildKernel(Data = np.ndarray, kernel = "Gauss", gamma = 0.1, alpha = 1e-4, C = -1.0):
    '''
    Builds Kernel matrix to be used in Kernel-type K-means
    
    Parameters
    ----------
    Data : (Nobs,Npix) our set of images
    type : Various Kernels taken from course notes, internet search and personal inspiration
        "Gauss": Gaussian Kernel -> K(x,y) = exp(-gamma * x^T y)
        "Poly2": Polynomial Kernel, degree 2 -> K(x,y) = (x^Ty + 1)^2 -> Parameter TO-BE-OPTIMIZED
        "Sigmoid": sigmoid Kernel -> K(x,y) = tanh(alpha * x^Ty + C), alpha = 0.1, C = 1 -> Parameters TO-BE-OPTIMIZED. Sigmoid Kernel should be similar to simple functioning neural network
        
        For the following: we add features to the data and Kernel is <=> eucl distance on augmented data
        "nnzcount": count # of colored pixels per image
        "quadrant_col_sum": counts the amount of color in each (14x14) quadrant of the (28x28) image
            
    Returns
    -------
    K : Kernel matrix for the selected Kernel.
        NB symmetry -> only upper triangular part will be nonzero
        Shape: (Nobs,Nobs), Format: CSC
    '''
    #INITIALIZE
    [Nobs, Npix] = np.shape(Data)
    K = [[] for index in range(Nobs)]
    
    #KERNEL COMPUTATION USING LISTS -> there is some numpy mixed in. Would probably be faster without it but not sure how to take it off.
    start = timeit.default_timer()
    vecofzeros = np.zeros((0,))
    if kernel == "Gauss":
        for ii, vector in enumerate(Data):
            # tmp = np.exp( -gamma * np.sum( (vector - Data[ii:Data.shape[0],:]) ** 2, axis = 1 ))
            K[ii].append(np.append(vecofzeros, np.exp( -gamma * np.sum( (vector - Data[ii:Data.shape[0],:]) ** 2, axis = 1 ) ) ) )
            vecofzeros = np.append(vecofzeros, 0)
    elif kernel == "Poly2":
        for ii, vector in enumerate(Data):
            tmp = np.sum( vector * Data[ii:Data.shape[0],:], axis = 1 ) 
            K[ii].append( np.append( vecofzeros, (tmp + np.ones(len(tmp))) ** 2 ) )
            vecofzeros = np.append(vecofzeros, 0)
    elif kernel == "Sigmoid":
        for ii, vector in enumerate(Data):
            tmp = np.sum( vector * Data[ii:Data.shape[0],:], axis = 1 )
            K[ii].append(np.append( vecofzeros, np.tanh(alpha * tmp + C * np.ones(len(tmp))) ) )
            vecofzeros = np.append(vecofzeros, 0)            
    elif kernel == "quadrant_col_sum":
        kernel_data = QuadrantColorSum(Data, Nobs, Npix)
        for ii, vector in enumerate(kernel_data):
            tmp =  np.sum( (vector - kernel_data[ii:,:]) ** 2, axis = 1 )
            K[ii].append(tmp)
            #TODO: finish these lines and the other ones too.
    else:
        raise ValueError("Enter available Kernel type") 
    stop = timeit.default_timer()
    print("Time to evaluate Kernel:", stop - start)

    # TURN LIST OF LISTS INTO TRIANGULAR MATRIX -> VERY SLOW AS Nobs INCREASES!!!
    start = timeit.default_timer()
    K = np.vstack(K)
    stop = timeit.default_timer()
    print("Time to post process:", stop - start)
    
    return K
    # return Kmatsparse
    # return K
    
def QuadrantColorSum(Data,Nobs,Npix):
    '''
    
    Parameters
    ----------
    Data 
    Nobs
    Npix

    Returns
    -------
    Data_kernelized : data with augmented feature: "color sum" (normalized) by quadrant
    '''
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
        Data_kernelized[point_id,Npix:Npix+4] = additional_features / np.sum(additional_features)
    
    return Data_kernelized