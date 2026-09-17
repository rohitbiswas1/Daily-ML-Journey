# Day 21 — Dimensionality Reduction with PCA

## What is PCA?
Principal Component Analysis (PCA) transforms correlated features into a smaller set of new features called principal components. The first component captures the largest possible amount of variance, the second captures the largest remaining amount, and so on.

## Why reduce dimensions?
- simplify high-dimensional datasets
- speed up some models
- reduce noise and redundancy
- make data easier to visualize in 2D or 3D

## Core ideas
- **Standardization:** PCA is scale-sensitive, so numeric features are usually standardized first.
- **Components:** each component is a weighted combination of the original features.
- **Explained variance ratio:** the fraction of total variance captured by each component.
- **Loadings:** the weights showing how strongly original features contribute to a component.

## Practical workflow
1. Split the data into train and test sets.
2. Fit a scaler on training data only.
3. Standardize train and test data with that scaler.
4. Fit PCA on the standardized training data.
5. Transform train and test data.
6. Train a downstream model on the reduced representation.
7. Compare performance and variance retained with the baseline.

## Important caution
PCA is unsupervised: it does not use target labels when choosing components. Maximum variance is not the same thing as maximum predictive information, so reducing dimensions can sometimes hurt model performance.

## Experiment
The Python practice uses the Iris dataset. A logistic-regression baseline is compared with logistic regression after reducing four standardized features to two principal components. The script prints retained variance, component loadings, and test accuracy.

## Key takeaway
PCA trades representation size for information. Choose the number of components based on the amount of variance retained, visualization needs, computational constraints, and—when appropriate—validation performance.
