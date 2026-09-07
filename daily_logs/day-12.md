# Day 12 — Classification Fundamentals

**Date:** 2026-09-07

## What I learned
- Classification predicts discrete categories rather than continuous numeric values.
- Binary classification has two classes; multiclass classification has three or more.
- Logistic regression can estimate class probabilities using the sigmoid function.
- `predict()` returns the selected class, while `predict_proba()` exposes class probabilities.
- A confusion matrix helps separate true positives, true negatives, false positives, and false negatives.
- Accuracy should not be the only metric, especially when classes are imbalanced.

## What I practiced
- Built a small synthetic pass/fail dataset using study hours and practice-test count.
- Split the data into training and test sets using stratification.
- Trained a logistic regression classifier.
- Evaluated it with accuracy, a confusion matrix, precision, recall, and F1-score.
- Predicted the pass probability for a new student.

## Experiment result
The classifier learns a boundary between lower and higher study/practice patterns. The exact test metrics depend on the fixed train/test split, while `predict_proba()` demonstrates that a classification model can provide a probability estimate rather than only a hard label.

## Reflection
Classification is one of the core supervised learning tasks. Understanding probabilities and error types now makes the later evaluation topics—precision, recall, F1-score, and threshold selection—much easier to reason about.

## Next
Go deeper into classification evaluation: confusion matrices, precision, recall, F1-score, and threshold trade-offs.
