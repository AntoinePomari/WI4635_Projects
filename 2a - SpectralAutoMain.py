import numpy as np
import Kmeans_Kernel as KmK
import SpectralClustering_automaticeigenvalues as SCA
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float64) / 255.0 #To float & normalize: general habit for computations
K = 10

# Number of data points to be used: ideally 70000
BigN = 10000
Data = Xarray[:BigN,:] 
real_val = y[:BigN] 

## Spectral clustering, see description below

# Options: (as for exercise 1.a and 1.b; normalize to choose normalized or unnormalized spectral clustering)
# Kmeans_initialize: random, ++ 
# kernel: Gauss, Poly, Sigmoid, quadrant_col_sum, nonzerototal, nonzerorows, nonzerorows_cols
# normalize: True, False

[centers, clust] = SCA.SpecClust(Data, K, Kmeans_maxit = 150, kernel = "Gauss", normalize = False)
centers = centers * 255.0

result = KmK.MostRepresentedInEachCluster(centers,clust, real_val)
print("Most present number in each cluster:",result)

acc = KmK.Accuracy(centers,clust, real_val)
print("Accuracy for this model:", acc)