# Day 14 — Decision Trees & Interpretability

## Learning goals
- Understand how a decision tree makes predictions through if/then splits.
- Learn impurity intuition using Gini impurity and entropy.
- Understand `max_depth`, overfitting, and why pruning/regularization matters.
- Inspect feature importance and simple tree rules for interpretability.

## Core idea
A decision tree recursively splits the training data into smaller groups. For classification, a useful split creates child groups that are as pure as possible.

For a node with class proportions `p_i`, Gini impurity is:

`Gini = 1 - Σ p_i²`

A pure node has Gini = 0. The training algorithm searches for splits that reduce impurity.

## Practical workflow
1. Split data into training and test sets.
2. Fit a `DecisionTreeClassifier`.
3. Compare a shallow tree with a deeper tree.
4. Evaluate accuracy and a confusion matrix.
5. Inspect feature importances and export readable tree rules.

## Overfitting intuition
A very deep tree can memorize training examples and perform poorly on unseen data. Limiting `max_depth`, increasing `min_samples_leaf`, or using cross-validation helps control model complexity.

## Interpretability
Trees are easier to explain than many black-box models because predictions can be traced through a sequence of feature thresholds. Feature importance is useful for a quick summary, but it should not be treated as causal evidence.

## Experiment
The accompanying Python practice uses the breast-cancer dataset available through scikit-learn. It compares shallow and deeper trees and prints metrics, feature importance, and human-readable decision rules.

## Takeaway
Decision trees are both predictive models and interpretable rule systems. The key practical skill is balancing fit against complexity rather than simply growing the deepest possible tree.
