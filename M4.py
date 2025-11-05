'''Clustering Analysis (Any one)
A. Implement K-Means clustering on Iris.csv dataset. Determine the number of clusters
using the elbow method.'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

df = datasets.load_iris(as_frame=True)

df = pd.DataFrame(df.data)

df.info()

df.describe()

x = df.iloc[:, [0,1,2,3]].values

# Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = [] # WCSS -> Within Clusters Sum of Squares

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the results into a line graph, allowing us to observe the 'elbow'

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Numebr of clusters')
plt.ylabel('WCSS')
plt.show()

# Creating the KMeans Classifier
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Visualizing the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolor')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clsuters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 100, c = 'yellow', label = 'Centroids')

plt.legend()