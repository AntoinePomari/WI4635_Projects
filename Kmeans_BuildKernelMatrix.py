# -*- coding: utf-8 -*-
import numpy as np
# import scipy.sparse as sps
import timeit


def BuildKernel(Data = np.ndarray, kernel = "Gauss", gamma = 0.1, alpha = 1e-3, C = 0.0):
    '''
    Builds Kernel matrix to be used in Kernel-type K-means
    
    Parameters
    ----------
    Data : (Nobs,Npix) our set of images
    type : Various Kernels taken from course notes, internet search and personal inspiration
        "Gauss": Gaussian Kernel -> K(x,y) = exp(-gamma * x^T y) -> Parameter TO-BE-OPTIMIZED
        "Poly2": Polynomial Kernel, degree 2 -> K(x,y) = (x^Ty + 1)^2 
        "Sigmoid": sigmoid Kernel -> K(x,y) = tanh(alpha * x^Ty + C), alpha = 0.1, C = 1 -> Parameters TO-BE-OPTIMIZED. Sigmoid Kernel should be similar to simple functioning neural network
        
        For the following: we add features to the data and Kernel is <=> eucl distance on augmented data (NB still using formulas from Algorithm 23 Lect Notes)
        "nnzcount": count # of colored pixels per image
        "quadrant_col_sum": counts the amount of color in each (14x14) quadrant of the (28x28) image
            
    Returns
    -------
    K : Kernel matrix for the selected Kernel.
        NB symmetry -> only upper triangular part is computed. Final matrix has to be calculated "manually" outside
        Shape: (Nobs,Nobs), Format: np.ndarray (non sparse!)
    '''
    #INITIALIZE
    [Nobs, Npix] = np.shape(Data)
    K = [[] for index in range(Nobs)]
    
    #KERNEL COMPUTATION USING LISTS -> there is some numpy mixed in. Would probably be faster without it but not sure how to avoid it.
    start = timeit.default_timer()
    vecofzeros = np.zeros((0,), dtype= Data.dtype)
    print("Starting computation of kernel values...")
    if kernel == "Gauss":
        for ii, vector in enumerate(Data):
            # tmp = np.exp( -gamma * np.sum( (vector - Data[ii:,:]) ** 2, axis = 1 ))
            # tmp = np.append(vecofzeros,tmp)
            # K[ii].append(tmp)
            K[ii].append(np.append(vecofzeros, np.exp( -gamma * np.sum( (vector - Data[ii:,:]) ** 2, axis = 1 ) ) ) )
            vecofzeros = np.append(vecofzeros, 0)
    elif kernel == "Poly2":
        # vecofones = np.ones((0,Nobs))
        for ii, vector in enumerate(Data):
            tmp = np.sum( vector * Data[ii:,:], axis = 1 )
            # tmp = (tmp + vecofones) ** 2
            #vecofones = vecofones[1:]
            tmp = (tmp + np.ones(len(tmp))) ** 4 
            tmp = np.append(vecofzeros,tmp)
            K[ii].append(tmp)
            vecofzeros = np.append(vecofzeros, 0)
    elif kernel == "Sigmoid":
        for ii, vector in enumerate(Data):
            tmp = np.sum( vector * Data[ii:,:], axis = 1 )
            tmp = np.tanh(alpha * tmp + C * np.ones(len(tmp)))
            tmp = np.append(vecofzeros, tmp)
            K[ii].append( tmp ) 
            vecofzeros = np.append(vecofzeros, 0)            
    elif kernel == "quadrant_col_sum":
        kernel_data = QuadrantColorSum(Data, Nobs, Npix) #Lots of memory used, surely not optimal....
        for ii, vector in enumerate(kernel_data):
            tmp =  np.sum( (vector - kernel_data[ii:,:]) ** 2, axis = 1 )
            K[ii].append( np.append(vecofzeros, tmp) )
            vecofzeros = np.append(vecofzeros, 0)
    elif kernel == "nnzcount":
        kernel_data = NnzCount(Data, Nobs, Npix)  #Lots of memory used, surely not optimal....
        for ii, vector in enumerate(kernel_data):
            tmp =  np.sum( (vector - kernel_data[ii:,:]) ** 2, axis = 1 )
            K[ii].append( np.append(vecofzeros, tmp) )
            vecofzeros = np.append(vecofzeros, 0)
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

def NnzCount(Data,Nobs,Npix):
    '''
    Parameters
    ----------
    Data 
    Nobs
    Npix

    Returns
    -------
    Data_kernelized : data with augmented feature: count of # of nonzero (<=> colored) pixels, normalized
    '''
    NonzeroCount = np.count_nonzero(Data, axis = 1)
    SuperMax = np.max( NonzeroCount )
    NonzeroCount = NonzeroCount / SuperMax #Normalize
    Data_kernelized = np.empty((Nobs,Npix+1))
    Data_kernelized[0:Nobs,0:Npix] = np.copy(Data)
    for index in range(Nobs):
        Data_kernelized[index,Npix+1] = NonzeroCount
    
    return Data_kernelized


