# Day 26 — Daily Learning Log

## Topic
Model Interpretability: Feature Importance, Permutation Importance & SHAP Intuition

## What I learned
- Random Forests provide impurity-based feature importance through `feature_importances_`.
- Permutation importance measures how much a held-out score changes when one feature is shuffled.
- Feature importance describes predictive reliance, not causality.
- Correlated features can make importance rankings harder to interpret.
- SHAP uses additive feature contributions to explain individual predictions relative to a baseline.

## What I practiced
- Trained a Random Forest on the built-in Breast Cancer Wisconsin dataset.
- Evaluated baseline test accuracy.
- Printed the top features using impurity-based importance.
- Computed repeated permutation importance on the test set.
- Compared the two importance rankings.

## Experiment
The main comparison was:

1. Random Forest impurity importance.
2. Permutation importance using test-set accuracy.

The goal was not to find a single "correct" ranking, but to understand that different explanation methods answer different questions.

## Reflection
I learned that a feature being important to a model does not mean it causes the prediction. Permutation importance is more directly connected to predictive performance, while built-in tree importance is convenient but can be affected by correlated or high-cardinality features. SHAP gives a useful next step for explaining individual predictions.

## Next
Day 27 — Model Deployment Basics: Saving, Loading & Serving a Model
