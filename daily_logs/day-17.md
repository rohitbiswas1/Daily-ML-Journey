# Day 17 — Support Vector Machines (SVM) & Margins

## What I learned
- SVM classification looks for a separating hyperplane with a large margin.
- Support vectors are the closest training examples and have strong influence on the boundary.
- `C` controls the penalty for classification errors and therefore the margin/error trade-off.
- Feature scaling is important for SVM because feature magnitude affects optimization.

## What I practiced
I trained a linear SVM on the breast-cancer dataset after standardizing the features. I compared `C = 0.01, 0.1, 1, 10, 100` using a stratified train/test split.

## Experiment question
How does regularization strength affect test accuracy?

## Reflection
A useful mental model is that SVM tries to separate classes while leaving as much space as possible around the boundary. The closest points matter most, and `C` determines how strongly the model prioritizes avoiding training errors over keeping the margin wide.

## Next
Day 18 — Naive Bayes & Probabilistic Classification
