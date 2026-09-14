# Day 18 — Naive Bayes & Probabilistic Classification

## What is Naive Bayes?
Naive Bayes is a supervised classification family based on Bayes' theorem. It combines prior class probabilities with evidence from the input features to estimate the probability of each class.

## Bayes' theorem

`P(class | features) ∝ P(class) × P(features | class)`

The denominator normalizes the result across classes. In practice, scikit-learn implementations compute the quantities efficiently, often in log space for numerical stability.

## Why is it called “naive”?
The model makes a simplifying conditional-independence assumption: once the class is known, features are treated as independent of one another. This assumption is often unrealistic, but the model can still work surprisingly well.

## Gaussian Naive Bayes
`GaussianNB` assumes each continuous feature follows a Gaussian distribution within each class. It estimates a mean and variance for each feature/class combination and uses those estimates to calculate class probabilities.

## Practical experiment
`python_ml/18_naive_bayes.py` trains Gaussian Naive Bayes on the breast-cancer dataset, prints accuracy and log loss, compares it with scaled logistic regression, and displays the first five probability predictions.

The experiment connects today's probabilistic classifier to the probability and statistics ideas from Days 6–7.

## Accuracy vs probability quality
Accuracy only checks whether the final class label is correct. `log_loss` also evaluates how confidently the model assigns probabilities. A model can have good accuracy while producing poorly calibrated or overconfident probabilities.

## Beginner takeaway
Think of Naive Bayes as **updating class beliefs using evidence**. It starts with how common each class is, then asks how likely the observed features are under each class.

## Next step
Learn how to combine models through **ensemble boosting**, starting with the intuition behind AdaBoost and gradient boosting.
