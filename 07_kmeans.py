# 07_kmeans.py
# Dataset: Iris (built into sklearn)
# Goal: K-means clustering - first unsupervised algorithm
# no labels this time, the model finds groups on its own

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.to_csv('data/iris.csv', index=False)

# we know iris has 3 species but pretending we dont
# thats the whole point of unsupervised - discover groups without labels
print("=== Iris Dataset ===")
print(df.head())
print(f"\nShape: {df.shape}")

# Step 2: Scale - kmeans uses distance so scaling matters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 3: Finding the right K using elbow method
# try k from 2 to 10 and see where the "elbow" is
inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Step 4: Train with k=3 (since elbow should be around there)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df['Cluster'] = kmeans.labels_

print(f"\n=== Cluster Results (k=3) ===")
print(f"Cluster sizes:")
print(df['Cluster'].value_counts().sort_index())
print(f"\nSilhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.4f}")
# closer to 1 = better separated clusters

# Step 5: What did each cluster find?
print(f"\n=== Cluster Profiles ===")
for c in range(3):
    cluster_data = df[df['Cluster'] == c]
    print(f"\nCluster {c} ({len(cluster_data)} samples):")
    print(cluster_data.describe().loc[['mean']].round(2).to_string())

# Step 6: Compare to actual species (cheating a bit to see how well it did)
df['Actual'] = iris.target
print(f"\n=== Cluster vs Actual Species ===")
print(pd.crosstab(df['Cluster'], df['Actual']))
# if kmeans worked well each cluster should mostly map to one species

# Step 7: Plots

# elbow method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'b-o')
plt.xlabel('K (number of clusters)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Finding Best K')
plt.tight_layout()
plt.savefig('plots/kmeans_elbow.png', dpi=150)
plt.show()
# looking for the bend in the curve

# silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_range, sil_scores, 'g-o')
plt.xlabel('K (number of clusters)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.tight_layout()
plt.savefig('plots/kmeans_silhouette.png', dpi=150)
plt.show()

# scatter plot of clusters using first 2 features
plt.figure(figsize=(8, 6))
colors = ['steelblue', 'coral', 'green']
for c in range(3):
    mask = kmeans.labels_ == c
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                c=colors[c], label=f'Cluster {c}', alpha=0.6)

# plot centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X',
            s=200, label='Centroids')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means Clusters (first 2 features)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/kmeans_clusters.png', dpi=150)
plt.show()

# actual species for comparison
plt.figure(figsize=(8, 6))
for s in range(3):
    mask = iris.target == s
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                c=colors[s], label=iris.target_names[s], alpha=0.6)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Actual Species (for comparison)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/kmeans_actual_species.png', dpi=150)
plt.show()
# pretty close to what kmeans found on its own which is cool

print("\nDone!")