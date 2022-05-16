# -*- coding: utf-8 -*-
import numpy as np
import timeit


def BuildKernel(Data = np.ndarray, kernel = "Gauss", gamma = 0.1, alpha = 1e-3, C = 0.0, r = 2):
    '''
    Builds Kernel matrix to be used in Kernel-type K-means
    
    Parameters
    ----------
    Data : (Nobs,Npix) our set of images
    kernel : Various Kernels taken from course notes, internet search and personal inspiration
        "Gauss": Gaussian Kernel -> K(x,y) = exp(-gamma * x^T y) 
        "Poly": Polynomial Kernel -> K(x,y) = (x^Ty + 1)^r 
        "Sigmoid": sigmoid Kernel -> K(x,y) = tanh(alpha * x^Ty + C)
        
        For the following: we add features as extra elements for each data point
        The kernel is then just euclidean distance on this augmented data (NB still using formulas from Algorithm 23 Lect Notes)
        "quadrant_col_sum" : counts the amount of color in each (14x14) quadrant of the (28x28) image
        "nonzerototal", "nonzerorows", "nonzerorows_cols" : counts the amount of non-zero elements
        in the entire vector, each row of the image, each row & each column of the image respectively
    Returns
    -------
    K : Kernel matrix for the selected Kernel.
        NB symmetry -> only upper triangular part is computed.
        Shape: (Nobs,Nobs), Format: np.ndarray 
    '''
    #INITIALIZE
    [Nobs, Npix] = np.shape(Data)
    K = [[] for index in range(Nobs)]
    
    #KERNEL COMPUTATION USING LISTS 
    start = timeit.default_timer()
    vecofzeros = np.zeros((0,), dtype= Data.dtype)
    print("Starting computation of kernel values...")
    if kernel == "Gauss":
        for ii, vector in enumerate(Data):
            # K[ii].append(np.append(vecofzeros, np.exp( -gamma * np.sum( (vector - Data[ii:,:]) ** 2, axis = 1 ) ) ) )
            K[ii].append(np.append(vecofzeros, np.exp( -gamma * np.linalg.norm(vector - Data[ii:,:], axis = 1 ) ) ) )            
            vecofzeros = np.append(vecofzeros, 0)
            
    elif kernel == "Poly":
        for ii, vector in enumerate(Data):
            tmp = np.sum( vector * Data[ii:,:], axis = 1 )
            tmp = (tmp + np.ones(len(tmp))) ** r 
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
        kernel_data = QuadrantColorSum(Data, Nobs, Npix) 
        for ii, vector in enumerate(kernel_data):
            tmp =  np.sum( (vector - kernel_data[ii:,:]) ** 2, axis = 1 )
            K[ii].append( np.append(vecofzeros, -tmp) ) 
            # Here we take -tmp since this is the Euclidean distance between the points with added features
            # and we want a measure of similarity instead of distance
            vecofzeros = np.append(vecofzeros, 0)
            
    elif kernel == "nonzerototal":
        kernel_data = Nonzerototal(Data, Nobs, Npix) 
        for ii, vector in enumerate(kernel_data):
            tmp =  np.sum( (vector - kernel_data[ii:,:]) ** 2, axis = 1 )
            K[ii].append( np.append(vecofzeros, -tmp) )
            vecofzeros = np.append(vecofzeros, 0)
            
    elif kernel == "nonzerorows":
        kernel_data = Nonzerorows(Data, Nobs, Npix) 
        for ii, vector in enumerate(kernel_data):
            tmp =  np.sum( (vector - kernel_data[ii:,:]) ** 2, axis = 1 )
            K[ii].append( np.append(vecofzeros, -tmp) )
            vecofzeros = np.append(vecofzeros, 0)
            
    elif kernel == "nonzerorows_cols":
        kernel_data = Nonzerorows_cols(Data, Nobs, Npix) 
        for ii, vector in enumerate(kernel_data):
            tmp =  np.sum( (vector - kernel_data[ii:,:]) ** 2, axis = 1 )
            K[ii].append( np.append(vecofzeros, -tmp) )
            vecofzeros = np.append(vecofzeros, 0)
            
    else:
        raise ValueError("Enter one of the available Kernel types") 
    stop = timeit.default_timer()
    print("Time to evaluate Kernel:", stop - start)

    # TURN LIST OF LISTS INTO TRIANGULAR MATRIX -> SLOW AS Nobs INCREASES
    start = timeit.default_timer()
    K = np.vstack(K)
    K = K.astype(Data.dtype)
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
    Data_kernelized = np.empty((Nobs,Npix+4),dtype=Data.dtype)
    Data_kernelized[0:Nobs,0:Npix] = np.copy(Data)
    for point_id, point in enumerate(Data):
        point = point.reshape((28,28))
        feat = 0
        color_sum = 0.0
        additional_features = np.zeros( (1,4) )
        for i in range(2):
            for j in range(2):
                color_sum = np.sum(point[13*i+i:13*i+13+i,13*j+j:13*j+13+j])
                additional_features[0,feat] = color_sum
                feat = feat+1
        Data_kernelized[point_id,Npix:Npix+4] = additional_features / np.sum(additional_features)
    
    return Data_kernelized

def Nonzerototal(Data,Nobs,Npix):
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
    Data_kernelized = np.empty((Nobs,Npix+1),dtype=Data.dtype)
    Data_kernelized[0:Nobs,0:Npix] = np.copy(Data)
    Data_kernelized[:,Npix] = NonzeroCount
    
    return Data_kernelized

def Nonzerorows(Data,Nobs,Npix):
    '''
    Parameters
    ----------
    Data 
    Nobs
    Npix

    Returns
    -------
    Data_kernelized : data with augmented feature: count of # of nonzero (<=> colored) pixels per row, normalized
    '''
    Data_kernelized = np.empty((Nobs,Npix+28),dtype=Data.dtype)
    Data_kernelized[0:Nobs,0:Npix] = np.copy(Data)
    
    for i in range(28):
        NonzeroCount = np.count_nonzero(Data[:,28*i:28*(i+1)], axis = 1)
        SuperMax = np.max( NonzeroCount )
        if SuperMax!=0:
            NonzeroCount = NonzeroCount / SuperMax #Normalize
    
        Data_kernelized[:,Npix+i] = NonzeroCount
    
    return Data_kernelized

def Nonzerorows_cols(Data,Nobs,Npix):
    '''
    Parameters
    ----------
    Data 
    Nobs
    Npix

    Returns
    -------
    Data_kernelized : data with augmented feature: count of # of nonzero (<=> colored) pixels per row and column, normalized
    '''
    Data_kernelized = np.empty((Nobs,Npix+56),dtype=Data.dtype)
    Data_kernelized[0:Nobs,0:Npix] = np.copy(Data)
    
    for i in range(28): #rows
        NonzeroCount = np.count_nonzero(Data[:,28*i:28*(i+1)], axis = 1)
        SuperMax = np.max( NonzeroCount )
        if SuperMax!=0:
            NonzeroCount = NonzeroCount / SuperMax #Normalize
    
        Data_kernelized[:,Npix+i] = NonzeroCount
        
    for i in range(28): #columns
        NonzeroCount = np.count_nonzero(Data[:,i:757+i:28], axis = 1)
        SuperMax = np.max( NonzeroCount )
        if SuperMax!=0:
            NonzeroCount = NonzeroCount / SuperMax #Normalize
    
        Data_kernelized[:,Npix+i] = NonzeroCount    
    return Data_kernelized