# Day 19 — Boosting & Gradient-Based Ensembles

## What I Learned
- Boosting builds an ensemble sequentially instead of training all learners independently.
- AdaBoost increases the influence of difficult training examples.
- Gradient Boosting adds trees that improve the current model by reducing the remaining loss.
- `n_estimators`, `learning_rate`, and tree depth interact to control model capacity.

## What I Practiced
- Compared a shallow Decision Tree, AdaBoost, and Gradient Boosting on a classification dataset.
- Tested Gradient Boosting with 20, 50, 100, and 200 estimators.
- Compared accuracy, precision, recall, and F1 rather than relying on accuracy alone.

## Experiment
The practice script uses scikit-learn's built-in breast-cancer dataset, keeping the experiment small and reproducible without downloading a separate file.

## Reflection
Boosting helped me understand a different ensemble idea from Random Forests. Random Forests reduce variance by combining independently trained trees, while boosting builds a sequence where each stage tries to improve the current ensemble. I also saw why estimator count should be tuned together with learning rate and tree complexity.

## Next
Day 20 — Unsupervised Learning: K-Means Clustering
