# Day 24 — Classification Thresholds, Calibration & Probability Quality

## Today's goal
Understand how classification thresholds affect precision and recall, and learn why predicted probabilities should be evaluated for calibration rather than only accuracy.

## What I practiced
- Trained Logistic Regression inside a scaling Pipeline.
- Generated positive-class probabilities with `predict_proba`.
- Compared thresholds of 0.30, 0.50, and 0.70.
- Measured accuracy, precision, and recall for each threshold.
- Calculated Brier score for probability quality.
- Built a small calibration table from probability bins.

## Experiment
The experiment uses scikit-learn's breast-cancer dataset, keeping the workflow reproducible and avoiding an unnecessary external data download.

The main observation is that changing the threshold changes the operating point of the same trained classifier. A lower threshold generally makes the model more willing to predict the positive class; a higher threshold makes it more conservative.

## Reflection
Today helped separate two ideas that are easy to mix up: a model can estimate a probability, while the application chooses a decision threshold. Accuracy alone cannot tell whether probabilities are trustworthy or whether the chosen error trade-off matches the real task.

## Next topic
Day 25 — Regression Evaluation, Residual Analysis & Model Diagnostics.
