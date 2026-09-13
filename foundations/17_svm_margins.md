# Day 17 — Support Vector Machines (SVM) & Margins

## What is SVM?
Support Vector Machine is a supervised learning algorithm commonly used for classification. It searches for a decision boundary that separates classes while trying to keep the margin around that boundary as large as possible.

## Core ideas
- **Hyperplane:** the decision boundary used to separate classes.
- **Margin:** the distance from the boundary to the closest training examples.
- **Support vectors:** the examples closest to the boundary; they strongly influence the fitted boundary.
- **C:** regularization strength. A small C allows more training errors in exchange for a wider, simpler margin; a large C penalizes errors more strongly and can fit the training data more tightly.

## Why scaling matters
SVM optimization is sensitive to feature magnitude. If one feature has values much larger than another, it can disproportionately affect the model. Standardization is therefore a strong default for many SVM workflows.

## Practical experiment
`python_ml/17_svm_margins.py` trains a linear SVM on the breast-cancer dataset with standardized features and compares several values of `C` (0.01, 0.1, 1, 10, 100).

The goal is not to pick a magic value from one split, but to observe how regularization changes test performance and to connect `C` to the bias/variance trade-off.

## Beginner takeaway
Think of SVM as **finding a boundary with breathing room**. The closest examples define that breathing room, while `C` controls how willing the model is to tolerate mistakes for a wider margin.

## Next step
Learn probabilistic classification with **Naive Bayes** and connect probability ideas from earlier days to a practical classifier.
