import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv('Mall_Customers.csv')

# 2. Choose features (same as yours)
X = df[['Age','Annual Income (k$)','Spending Score (1-100)']].values

# 3. Scale features (important for Euclidean distance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dendrogram (use scaled data)
plt.figure(figsize=(8, 4))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance (on scaled data)')
plt.show()

# 5. Agglomerative clustering (same n_clusters as yours)
n_clusters = 5
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = hc.fit_predict(X_scaled)

# 6. Attach labels and inspect
df['Cluster'] = cluster_labels
print(df[['Age','Annual Income (k$)','Spending Score (1-100)','Cluster']].head())

# 7. Plot clusters on Annual Income vs Spending Score (as you did)
plt.figure(figsize=(7,6))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual Income (k$)'],
                cluster_data['Spending Score (1-100)'],
                label=f'Cluster {cluster}')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.title('Clusters (Income vs Spending Score)')
plt.show()

# 8. Quick summary: counts and cluster means (centroid-like)
print("\nCluster sizes:\n", df['Cluster'].value_counts().sort_index())
print("\nCluster means (Age, Income, Spending Score):\n", df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean())
