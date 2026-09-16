"""
Day 20 — K-Means Clustering

Practice: group a small synthetic customer dataset into behavioral segments.
Run: python python_ml/20_kmeans_clustering.py
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Features: annual income (thousand currency units), spending score (0–100)
X = np.array(
    [
        [25, 20], [28, 24], [30, 18], [32, 28],
        [55, 45], [58, 50], [62, 48], [65, 55],
        [90, 82], [95, 88], [100, 85], [105, 92],
    ],
    dtype=float,
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("K-Means clustering on customer behavior\n")

results = []
for k in range(2, 5):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    results.append((k, score, model.inertia_))
    print(f"k={k}: silhouette={score:.3f}, inertia={model.inertia_:.3f}")

best_k = max(results, key=lambda item: item[1])[0]
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = final_model.fit_predict(X_scaled)

print(f"\nSelected k={best_k} using the highest silhouette score.")
print("Cluster assignments:")
for row, label in zip(X.astype(int), labels):
    print(f"  income={row[0]:3d}, spending={row[1]:2d} -> cluster {label}")

centers_original = scaler.inverse_transform(final_model.cluster_centers_)
print("\nCluster centers (original units):")
for i, center in enumerate(centers_original):
    print(f"  cluster {i}: income={center[0]:.1f}, spending={center[1]:.1f}")

print("\nKey lesson: K-Means is sensitive to feature scale, so standardization matters when features use different units.")
