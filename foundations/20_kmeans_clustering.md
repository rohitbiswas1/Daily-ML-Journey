# Day 20 — Unsupervised Learning: K-Means Clustering

## Learning goals
- Understand supervised vs unsupervised learning.
- Explain centroids, clusters, inertia, and the role of `k`.
- Use silhouette score as a simple cluster-quality signal.
- Understand why feature scaling can change distance-based clustering.

## Core idea
K-Means partitions observations into `k` groups. It repeatedly assigns each point to its nearest centroid and then recomputes each centroid as the mean of the assigned points.

### Important concepts
- **Centroid:** the mean location representing a cluster.
- **Inertia:** the sum of squared distances from each sample to its assigned centroid. Lower is better, but increasing `k` almost always lowers inertia.
- **Silhouette score:** compares within-cluster cohesion with separation from other clusters. Higher values generally indicate better-separated clusters.
- **Choosing `k`:** use domain knowledge plus tools such as the elbow method and silhouette score; there is no universally correct `k`.
- **Scaling:** K-Means uses distances, so features with larger numeric ranges can dominate unless features are appropriately scaled.

## Practical experiment
The Python script creates 12 small customer examples with annual income and spending score. It standardizes the features, tests `k=2,3,4`, compares silhouette scores and inertia, then fits the selected model and prints cluster assignments and centers in the original units.

## Why this matters
Clustering is useful for exploratory segmentation when labels are unavailable—for example, discovering customer groups, organizing documents, or identifying behavioral patterns. A cluster is not automatically a meaningful real-world category; interpretation should come from the underlying features and domain context.

## Practice questions
1. Why does K-Means need a value for `k`?
2. Why can inertia alone be misleading when choosing `k`?
3. What can happen if one feature is measured in thousands and another from 0–100?
4. Why should cluster labels not be treated as ordered classes?

## Takeaway
K-Means is a simple, powerful introduction to unsupervised learning: represent groups with centroids, use distances for assignment, and validate the result instead of assuming every cluster is meaningful.
