# Day 22 — Hyperparameter Tuning & Cross-Validation

**Date:** 2026-09-18

## What I learned
- Hyperparameters are chosen before model training and can change how a model generalizes.
- `C` in Logistic Regression controls the strength of regularization: smaller `C` means stronger regularization.
- Cross-validation gives a more stable estimate than relying on one validation split.
- Stratified K-fold keeps class proportions more consistent across classification folds.
- `GridSearchCV` can compare a small, explicit hyperparameter grid automatically.
- Preprocessing should live inside a `Pipeline` during cross-validation to avoid data leakage.

## What I practiced
- Loaded scikit-learn's built-in breast-cancer dataset.
- Created a stratified 80/20 train/test split.
- Built a baseline StandardScaler + LogisticRegression pipeline with `C=1`.
- Tuned `C` across `[0.01, 0.1, 1, 10, 100]` with 5-fold stratified cross-validation.
- Compared mean CV accuracy and standard deviation for each candidate.
- Evaluated the selected model once on the held-out test set.

## Experiment
The main lesson was the separation between **model selection** and **final evaluation**. The test set stayed untouched while `GridSearchCV` selected the best `C` using only the training data. The pipeline also ensured that scaling was learned separately inside each CV fold.

The experiment uses a built-in dataset rather than adding a downloaded file, keeping the practice reproducible and lightweight.

## Reflection
Today I understood why tuning a model on the test set is a mistake: repeated test-set decisions gradually make the test set part of the training process. Cross-validation gives me a safer place to compare configurations, while the final test set remains a final check of generalization.

## Next
Day 23 — Model Selection, Pipelines & Preventing Data Leakage
