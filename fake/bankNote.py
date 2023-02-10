import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('Banknote-authentication-dataset-.csv')

v1 = data['V1']
v2 = data['V2']
v1v2 = np.column_stack((v1, v2))
plt.scatter(v1,v2)
km_res = KMeans(n_clusters=7).fit(v1v2)

print (min(v1))
print (max(v1))
mean_v1 = sum(v1)/len(v1)
print(f"Mean v1: {mean_v1}")
print("Standard deviation: ", np.std(v1))


print (min(v2))
print (max(v2))
mean_v2 = sum(v2)/len(v2)
print(f"Mean v2: {mean_v2}")
print("Standard deviation: ", np.std(v2))


clusters = km_res.cluster_centers_
plt.scatter(v1,v2)
plt.xlabel('V1')
plt.ylabel('V2')

plt.scatter(clusters[:,0],clusters[:,1], s=1000, alpha=0.5, color='w')

# suppress warnings from final output


import warnings
warnings.simplefilter("ignore")

#load data
df = pd.read_csv('Banknote-authentication-dataset-.csv')
#df_labels = pd.read_csv('data_with_labels.csv')
print(df)



from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans, SpectralClustering

n_iter = 9
fig, ax = plt.subplots(3, 3, figsize=(16, 16))
ax = np.ravel(ax)
centers = []
for i in range(n_iter):
    # Run local implementation of kmeans
    km = KMeans(n_clusters=2,
                max_iter=3)
    km.fit(data)
    centroids = km.cluster_centers_
    centers.append(centroids)
    ax[i].scatter(data[km.labels_ == 0, 0], data[km.labels_ == 0, 1],
                  label='cluster 1')
    ax[i].scatter(data[km.labels_ == 1, 0], data[km.labels_ == 1, 1],
                  label='cluster 2')
    ax[i].scatter(centroids[:, 0], centroids[:, 1],
                  c='r', marker='*', s=300, label='centroid')
    ax[i].legend(loc='lower right')
    ax[i].set_aspect('equal')
plt.tight_layout()