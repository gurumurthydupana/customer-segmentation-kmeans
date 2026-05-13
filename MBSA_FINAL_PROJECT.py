import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.cm as cm

#Importing the dataset
customer_data = pd.read_csv('Mall_Customers.csv')

# Selecting the attributes for analyze
selected_features = customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(selected_features)

# Elbow method to determine optimal k
distortions = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_data)
    distortion = sum(np.min(cdist(normalized_data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / normalized_data.shape[0]
    distortions.append(distortion)

# Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, distortions, 'bo-', marker='x')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Distortion')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.tight_layout()
plt.show()

# Apply KMeans with chosen k=5
chosen_k = 5
kmeans_final = KMeans(n_clusters=chosen_k, random_state=42)
labels = kmeans_final.fit_predict(normalized_data)
centroids = kmeans_final.cluster_centers_

# 3D Cluster Visualization plot of cluster data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors_3d = ['red', 'green', 'blue', 'purple', 'orange']

for i in range(chosen_k):
    cluster_points = normalized_data[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
               color=colors_3d[i], label=f'Cluster {i+1}', alpha=0.6)

# Ploting centroids
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
           color='black', marker='X', s=200, label='Centroids')

ax.set_title('Customer Segments (3D View)')
ax.set_xlabel('Standardized Age')
ax.set_ylabel('Standardized Income')
ax.set_zlabel('Standardized Spending Score')
ax.legend()
plt.tight_layout()
plt.show()

# Bar Chart for Distribution of Customers in Each Cluster
cluster_counts = pd.Series(labels).value_counts().sort_index()

# Assign different colors dynamically
bar_colors = cm.get_cmap('Set3', chosen_k).colors

plt.figure(figsize=(7, 5))
plt.bar(
    x=[f'Cluster {i+1}' for i in cluster_counts.index],
    height=cluster_counts.values,
    color=bar_colors
)
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Customers per Cluster')
plt.tight_layout()
plt.show()