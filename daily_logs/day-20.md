# Day 20 — Daily Log

## Topic
Unsupervised Learning: K-Means Clustering

## What I learned
- K-Means groups unlabeled observations by distance to learned centroids.
- The number of clusters, k, is a modeling choice.
- Inertia measures within-cluster squared distance; silhouette score gives a complementary measure of cohesion and separation.
- Standardization matters when feature scales differ.

## Practice
- Built a small synthetic customer-segmentation dataset in Python.
- Compared k=2, k=3, and k=4 using silhouette score and inertia.
- Trained a final model and converted cluster centers back to original units.

## Reflection
Today I moved from predicting known labels to discovering structure without labels. The experiment also showed why clustering needs validation and domain interpretation.

## Next
Day 21 — Dimensionality Reduction with PCA.
