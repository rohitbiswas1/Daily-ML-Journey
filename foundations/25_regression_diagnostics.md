# Day 25 — Regression Evaluation, Residual Analysis & Model Diagnostics

## Why this matters
A regression model can look reasonable from one metric while still making systematic errors. Evaluation should combine error metrics with residual analysis.

## Core metrics
- **MAE (Mean Absolute Error):** average absolute prediction error. Easy to interpret in the target's units.
- **RMSE (Root Mean Squared Error):** square-rooted average squared error. Larger mistakes receive more weight.
- **R²:** proportion of target variance explained relative to a constant-mean baseline. It can be negative on unseen data when the model is worse than that baseline.

## Residuals
A residual is:

`residual = actual - predicted`

Useful checks include:
1. Residual mean: large systematic bias is a warning sign.
2. Residual spread: unusually changing spread can indicate non-constant error variance.
3. Residuals vs predictions/features: visible patterns can indicate that the model is missing structure.
4. Large residuals: inspect unusual observations rather than automatically deleting them.

## Practical experiment
The Python practice uses scikit-learn's built-in diabetes regression dataset. It compares:
- Linear Regression
- Degree-2 Polynomial Regression in a preprocessing pipeline

Both are evaluated on the same held-out test set with MAE, RMSE, R², and residual summaries.

## Interpretation
A more flexible model is not automatically better. If polynomial features reduce training error but do not improve held-out metrics, they may be adding complexity without useful generalization.

For a real project, residual plots would be a useful next diagnostic: plot residuals against predictions and important features and look for curves, funnels, or clusters.

## Key takeaway
**Good regression evaluation is more than one score.** Use complementary metrics and inspect residual behavior to understand where and why a model is making errors.

## Next
Day 26 — **Model Interpretability: Feature Importance, Permutation Importance & SHAP Intuition**.
