# Day 15 — Ensemble Learning: Random Forests

## Learning goals
- Understand why ensembles can outperform a single model.
- Learn how bagging and random feature selection create diverse decision trees.
- Use a Random Forest classifier and compare it with one decision tree.
- Interpret feature importance without treating it as causation.

## Core ideas
A **Random Forest** trains many decision trees and combines their predictions. For classification, the forest commonly uses majority voting. Each tree sees a bootstrap sample of the training data and considers a randomized subset of features at candidate splits.

This combination mainly helps reduce the high variance of an individual decision tree. A deep tree can fit noise; averaging many less-correlated trees usually makes the final model more stable.

### Important hyperparameters
- `n_estimators`: number of trees. More trees can stabilize results but increase computation.
- `max_depth`: limits tree complexity and can reduce overfitting.
- `max_features`: controls how many features are considered at each split.
- `min_samples_leaf`: requires a minimum number of samples in a leaf.
- `random_state`: makes experiments reproducible.

## Practical experiment
`python_ml/15_random_forest_ensemble.py` uses scikit-learn's built-in breast-cancer classification dataset. It compares a depth-limited decision tree with a 200-tree random forest using the same stratified train/test split.

The script prints training/test accuracy, a confusion matrix, a classification report, and the five highest feature-importance values.

## Interpretation
If the forest has a smaller train/test gap and stronger test performance than the single tree, that is evidence that the ensemble is generalizing more reliably in this split. Exact scores can vary with the data split and hyperparameters.

Feature importance is useful for model inspection, but it does **not** prove that a feature causes the target. For more robust interpretation, compare multiple methods and use domain knowledge.

## Beginner challenge
Change `n_estimators` to 20, 50, 100, and 300. Record test accuracy and training time. Then change `max_depth` and explain how model complexity affects the train/test gap.

## Takeaway
**One tree learns one set of rules; a forest combines many varied trees to reduce variance and improve stability.**
