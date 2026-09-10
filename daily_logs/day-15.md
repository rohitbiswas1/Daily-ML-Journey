# Day 15 — Random Forests

## What I learned
- Ensemble learning combines multiple models to produce a stronger, more stable predictor.
- Random Forest uses bootstrap samples and randomized feature selection to build diverse decision trees.
- Averaging/voting across trees can reduce variance and the instability of a single tree.
- Important hyperparameters include `n_estimators`, `max_depth`, `max_features`, and `min_samples_leaf`.

## Practical work
Compared a depth-limited Decision Tree with a 200-tree Random Forest on the scikit-learn breast-cancer dataset using the same stratified 80/20 train-test split.

The experiment reports train/test accuracy, a confusion matrix, a classification report, and the top five feature importances.

## What I understood
A single decision tree can become sensitive to small changes in training data. A forest creates many different trees and aggregates their predictions, making the overall model less dependent on one set of splits.

Feature importance helps inspect what the model relied on, but it should not be interpreted as proof of causality.

## Practice challenge
Test several values of `n_estimators` and `max_depth`, record test accuracy and runtime, and compare the train/test gap.

## Reflection
Today connected decision trees to a practical ensemble method. The key idea is not simply “more trees,” but combining diverse trees so their individual errors are less likely to dominate the final prediction.
