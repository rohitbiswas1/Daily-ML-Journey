# Day 14 — Decision Trees & Interpretability

## What I learned
- Decision trees make predictions using recursive if/then feature splits.
- Gini impurity measures how mixed the classes are in a node.
- Tree depth controls model complexity and can strongly affect generalization.
- Feature importance and exported rules provide useful model explanations, but they are not causal claims.

## Practical work
I trained a shallow decision tree and an unrestricted tree on scikit-learn's built-in breast-cancer classification dataset. I compared training and test accuracy, printed a confusion matrix and classification report, ranked feature importances, and inspected readable tree rules.

## Key observation
The unrestricted tree can fit the training set much more closely than a constrained tree, illustrating why controlling complexity matters. A shallow tree is easier to explain and can generalize better depending on the data.

## Reflection
Today connected classification metrics from Day 13 with an interpretable model. The important lesson is that a model should not only fit the training data; its complexity and behavior on unseen data also matter.

## Next
Day 15 — Ensemble Learning: Random Forests
