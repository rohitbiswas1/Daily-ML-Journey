# Day 21 — Dimensionality Reduction with PCA

## Learned
- PCA creates orthogonal principal components from the original features.
- Components are ordered by captured variance.
- Standardization is important when features use different scales.
- Explained variance helps decide how many components to keep.
- PCA can improve simplicity and visualization, but it can also remove predictive information.

## Practical Work
- Used the Iris dataset from scikit-learn.
- Split the data before preprocessing to avoid leakage.
- Standardized four input features using training data only.
- Reduced the representation from four dimensions to two with PCA.
- Compared Logistic Regression accuracy before and after reduction.
- Inspected explained-variance ratios and component loadings.

## Reflection
Today I learned that dimensionality reduction is not simply deleting columns. PCA builds new features that summarize variation across the original features. I also saw why scaling and fitting preprocessing only on training data matter.

## Next
Day 22 — Hyperparameter Tuning and Cross-Validation.
