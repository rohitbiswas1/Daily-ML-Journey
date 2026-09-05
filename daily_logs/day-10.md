# Day 10 — Feature Scaling: Normalization and Standardization

**Date:** 2026-09-05

## What I learned
- Features can have very different numeric ranges and units.
- Min-Max normalization commonly maps values into the 0–1 range.
- Standardization transforms a feature to approximately mean 0 and standard deviation 1.
- Scaling is important for many distance-based and gradient-based algorithms.
- Tree-based models usually do not need feature scaling.
- Scalers should be fitted on training data only to prevent data leakage.

## What I practiced
- Created a small two-feature dataset with deliberately different scales.
- Applied `MinMaxScaler` and `StandardScaler` from scikit-learn.
- Compared the transformed values and summary statistics.

## Experiment result
The Min-Max transformation placed each feature between 0 and 1 for this practice dataset. Standardization produced features with mean approximately 0 and standard deviation approximately 1.

## Reflection
Today I learned that preprocessing is closely connected to model choice. Scaling is not automatically required for every algorithm, but it can make a major difference when an algorithm depends on distances, magnitudes, or optimization steps.

## Next
Study linear regression in more depth: coefficients, predictions, residuals, and the best-fit line.
