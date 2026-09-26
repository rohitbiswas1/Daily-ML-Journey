# Day 30 — End-to-End ML Project & 30-Day Journey Review

## Goal
Bring the journey together in one reproducible classification workflow, from data to a saved model artifact.

## Workflow
1. Load labeled data.
2. Split into training and held-out test sets.
3. Put preprocessing and the estimator inside a Pipeline.
4. Run stratified 5-fold cross-validation on training data.
5. Fit the finalized pipeline on all training data.
6. Evaluate once on the held-out test set.
7. Save the complete pipeline for later inference.

## Why this is safer
The scaler is fitted inside each training fold, preventing preprocessing leakage. The final test set stays untouched until the end.

## Metrics
Use cross-validation ROC-AUC for model stability and final accuracy/ROC-AUC plus the confusion matrix for held-out evaluation.

## 30-day review
The journey progressed from Python/data foundations into supervised learning, ensembles, unsupervised learning, PCA, tuning, leakage prevention, calibration, interpretability, deployment, batch inference, and monitoring.

## Reusable ML lifecycle
Understand data -> split safely -> build a reproducible pipeline -> validate -> test once -> deploy -> monitor -> iterate.

## Next step
Extend this project with FastAPI, automated tests, experiment tracking, and production monitoring.