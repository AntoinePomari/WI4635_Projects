# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 04:39:49 2022

@author: 31649
"""

import numpy as np


L_inv = np.linalg.inv(L)
#L_inv = np.linalg.pinv(L)  pseudo inverse




def inverse_power(A, mu, v, niter, tol):
    for i in range(niter)
        Av = np.dot(A,v)
        Av_norm = np.linalg.norm(Av)
        v = Av/Av_norm
        mu_new = np.dot(np.transpose(v),np,dot(A,v))
        if (abs(mu_new - mu)/mu_new) < tol:
            return mu_new, v
        mu = mu_new
    
    return mu_new, v


def find_eigen_vectors(L, k):
    smallest_k = np.zeros((k, L.shape[0]))
    mu = 0
    v = np.ones(L.shape[0]) 
    for i in range(0, k):
        L = L - mu*np.dot(v, np.transpose(v))
        mu, v = inverse_power(L, mu, v, niter, tol)
        smallest_k[i,:] = v
    
    return smallest_k
        
        
        