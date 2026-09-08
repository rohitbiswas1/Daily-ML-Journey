# Day 13 — Classification Evaluation

## Core ideas
- A confusion matrix counts true positives, true negatives, false positives, and false negatives.
- Accuracy is the fraction of all predictions that are correct.
- Precision answers: among predicted positives, how many were actually positive?
- Recall answers: among actual positives, how many did the model find?
- F1 is the harmonic mean of precision and recall and is useful when both matter.
- A probability threshold converts predicted probabilities into class labels.
- Lowering the threshold often increases recall and can reduce precision; raising it often does the opposite.

## Practice
Train logistic regression on a synthetic binary dataset and evaluate thresholds of 0.30, 0.50, and 0.70. Compare the confusion matrices and metrics rather than relying on accuracy alone.

## ML takeaway
The best threshold depends on the cost of false positives versus false negatives. In a high-risk screening problem, missing a positive may be much worse than investigating an extra false positive.
