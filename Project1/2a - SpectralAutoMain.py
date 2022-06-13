import numpy as np
import Kmeans_Kernel as KmK
import SpectralClustering_automaticeigenvalues as SCA
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
Xarray = X.to_numpy()
Xarray = Xarray.astype(np.float32) / 255.0 #To float & normalize: computational habit
# Number of clusters 
K = 10
# Random seed for reproducibility
np.random.seed(6)
# Number of data points for triela: ideally 70000 at least once
BigN = 1500
indices = np.random.choice(Xarray.shape[0], size = BigN, replace = False)
Data = Xarray[indices,:]
real_val = y[indices] 

# Spectral clustering, see description below

# Options: (as for exercise 1.a and 1.b; normalize to choose normalized or unnormalized spectral clustering)
# Kmeans_initialize: random, ++ 
# kernel: Gauss, Poly, Sigmoid, quadrant_col_sum, nonzerototal, nonzerorows, nonzerorows_cols
# normalize: True, False
'''
The complete routine, non normalized Spec Clust
'''
[centers, clust] = SCA.SpecClust(Data, K, Kmeans_maxit = 150, Kmeans_initialize = "clusters_random",  Laplacian_kernel = "Gauss", normalize = False, TOL = 1e-5)
clust_real_ids = [[] for kk in range(K)]
for cluster_id, cluster in enumerate(clust):
    for element in cluster:
        clust_real_ids[cluster_id].append(indices[element])
del cluster, cluster_id, element, clust


result = KmK.MostRepresentedInEachCluster(clust_real_ids, real_val)
print("Most present number in each cluster:",result)

acc = KmK.Accuracy(clust_real_ids, real_val)
print("Accuracy for this model:", acc)

'''
The complete rountine, normalized Spec Clust
'''
[centers_tilde, clust_tilde] = SCA.SpecClust(Data, K, Kmeans_maxit = 150, Kmeans_initialize = "Clusters_random", kernel = "Gauss", normalize = True, TOL = 1e-7)
clust_real_tilde = [[] for kk in range(K)]
for cluster_id, cluster in enumerate(clust_tilde):
    for element in cluster:
        clust_real_tilde[cluster_id].append(indices[element])
del cluster, cluster_id, element, clust_tilde



result_tilde = KmK.MostRepresentedInEachCluster(clust_real_tilde, real_val)
print("Most present number in each cluster:",result)

acc_tilde = KmK.Accuracy(clust_real_tilde, real_val)
print("Accuracy for this model:", acc)


'''
The Complete routine, non-normalized Kernel Spec Clust
'''
clust = SCA.SpecClust_Kernel(Data, K, Kmeans_maxit = 100,  Kmeans_kernel = "Poly", Laplacian_kernel = "Gauss", normalize = False, TOL = 1e-5)
clust_real_ids = [[] for kk in range(K)]
for cluster_id, cluster in enumerate(clust):
    for element in cluster:
        clust_real_ids[cluster_id].append(indices[element])
del cluster, cluster_id, element, clust
result = KmK.MostRepresentedInEachCluster(clust_real_ids, real_val)
print("Most present number in each cluster:",result)

acc = KmK.Accuracy(clust_real_ids, real_val)
print("Accuracy for this model:", acc)



