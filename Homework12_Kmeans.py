"""
PHSX 815 Homework #12 

Date: April 24, 2023
Author: @aelieber1

Goals: 
- Generate data from a "mixture model"

- Implement the K-means clustering algorithm and apply it to your data

- Some potential points to evaluate: How similar are the clusters to the "true" mixture? Does this depend on the amount of data? How does the model change with the number of mixture components (keeping the number fixed in the generating model)? How well can you visualize your data and algorithm?

Sources for help:
- https://towardsdatascience.com/clustering-out-of-the-black-box-5e8285220717

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

 
# load the iris dataset
iris = datasets.load_iris()
 
# select first two columns
X = iris.data[:, :2]
 
# turn it into a dataframe
d = pd.DataFrame(X)
 
# plot the data
plt.scatter(d[0], d[1])
plt.title("Data")
plt.show()

# Gaussian mixture implementation
gmm = GaussianMixture(n_components = 3)
 
# Fit the GMM model for the dataset
# which expresses the dataset as a
# mixture of 3 Gaussian Distribution
gmm.fit(d)
 
# Assign a label to each sample
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 
# plot three clusters in same plot
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')
plt.title("Gaussian Mixture Maximization Cluster")
plt.show()

# print the converged log-likelihood value
print(gmm.lower_bound_)
 
# print the number of iterations needed
# for the log-likelihood value to converge
print(gmm.n_iter_)

# Iteration plot
data = list(zip(d[0], d[1]))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init='auto')
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.show()

# K-means clustering
kmeans = KMeans(n_clusters=2, n_init='auto')
kmeans.fit(data)

plt.scatter(d[0], d[1], c=kmeans.labels_)
plt.title("K Means Clustering")
plt.show()