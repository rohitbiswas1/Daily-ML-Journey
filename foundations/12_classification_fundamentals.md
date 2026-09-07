# Day 12 — Classification Fundamentals

## Goal
Understand how machine learning predicts **categories** instead of continuous numeric values.

## What is classification?
Classification is a supervised learning task where the target is a discrete class, such as:

- spam / not spam
- pass / fail
- fraud / not fraud
- disease / no disease

A classification model learns a relationship between input features `X` and a categorical target `y`.

## Binary vs. multiclass classification
- **Binary classification:** two classes, such as 0/1.
- **Multiclass classification:** three or more classes, such as low/medium/high.

## Logistic regression intuition
Despite its name, logistic regression is commonly used for classification. It converts a linear score into a probability between 0 and 1 using the logistic (sigmoid) function.

The model can then choose a class using a decision threshold, commonly 0.5 for binary classification.

```text
linear score -> sigmoid -> probability -> class
```

## Experiment
The practice script uses a tiny synthetic dataset with two features:

1. study hours
2. number of practice tests

The target is `0 = fail` and `1 = pass`. A logistic regression model is trained after a stratified train/test split.

The experiment prints:
- test accuracy
- confusion matrix
- precision, recall, and F1-score
- predicted probability for a new student
- predicted class for that student

## Confusion matrix intuition
For binary classification:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | True Negative | False Positive |
| Actual 1 | False Negative | True Positive |

Accuracy alone can be misleading when classes are imbalanced, so precision, recall, F1-score, and the confusion matrix are also useful.

## Important distinction from regression
- **Regression:** predicts a continuous value, such as price or temperature.
- **Classification:** predicts a class, such as approved/rejected.

## Practical ML lesson
A good classification workflow separates training and testing data before evaluation. Looking at test labels while training would cause data leakage and produce an overly optimistic evaluation.

## Practice questions
1. Why is predicting house price a regression problem?
2. Why is predicting whether an email is spam a classification problem?
3. What does `predict_proba()` provide that `predict()` does not?
4. What do false positives and false negatives mean in a real-world problem you choose?

## Next
Learn how classification models are evaluated in more depth, including confusion matrices, precision, recall, F1-score, and threshold trade-offs.
