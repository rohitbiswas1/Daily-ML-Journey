# Day 18 — Naive Bayes & Probabilistic Classification

## What I learned
- Naive Bayes applies Bayes' theorem to estimate class probabilities.
- The “naive” assumption treats features as conditionally independent given the class.
- Gaussian Naive Bayes models continuous features with class-specific means and variances.
- Accuracy evaluates final labels, while log loss evaluates the quality of predicted probabilities.

## What I practiced
I trained Gaussian Naive Bayes on the breast-cancer dataset, inspected probability predictions, and compared its accuracy and log loss with scaled logistic regression.

## Experiment question
Can a simple probabilistic model produce useful class predictions even with a strong independence assumption?

## Reflection
Naive Bayes connects the probability concepts from earlier days to a practical classifier. I also learned that a probability output contains more information than a hard class label, so metrics such as log loss can reveal differences that accuracy hides.

## Next
Day 19 — Boosting & Gradient-Based Ensembles
