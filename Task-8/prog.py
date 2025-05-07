# kmeans_clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Select relevant features: Annual Income and Spending Score
X = df.iloc[:, [3, 4]]  # Using Annual Income and Spending Score

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K_values = range(1, 11)

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_values, inertia, marker="o", linestyle="-")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Choose optimal K (based on elbow method observation)
optimal_K = 5  # Adjust this based on the Elbow Method plot

# Fit K-Means clustering model
kmeans = KMeans(n_clusters=optimal_K, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Evaluate clustering using Silhouette Score
silhouette_avg = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Visualize clusters in a 2D plot
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap="viridis", edgecolors="k", alpha=0.7)
plt.xlabel("Annual Income (Standardized)")
plt.ylabel("Spending Score (Standardized)")
plt.title("K-Means Clustering Results")
plt.colorbar(label="Cluster")
plt.show()
