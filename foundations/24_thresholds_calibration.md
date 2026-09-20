# Day 24 — Classification Thresholds, Calibration & Probability Quality

## What I learned

A binary classifier often produces a probability rather than a final yes/no decision. A threshold converts that probability into a class. The default threshold is commonly 0.50, but it is a decision rule, not a law.

### Thresholds

For a positive-class probability `p`:

- threshold 0.30: more examples become positive, usually increasing recall and potentially lowering precision
- threshold 0.50: the common default
- threshold 0.70: fewer examples become positive, often increasing precision while reducing recall

The right threshold depends on the cost of false positives and false negatives. For medical screening or fraud detection, missing a positive case can be much more costly than generating an extra alert.

### Probability quality

A probability of 0.80 should mean that roughly 80% of similar predictions are positive. This property is called **calibration**. A model can rank examples well while still producing poorly calibrated probabilities.

The practice script uses a Logistic Regression pipeline on scikit-learn's breast-cancer dataset and reports the Brier score. The Brier score measures the squared difference between predicted probabilities and actual binary outcomes; lower is better.

It also prints a small reliability table using calibration bins. When predicted and observed rates are close, the probabilities are behaving more like trustworthy probabilities.

## Practical workflow

1. Train the classifier without using the test set for decisions.
2. Generate probabilities with `predict_proba`.
3. Choose a threshold using training/validation data and the business or application cost of errors.
4. Evaluate the chosen rule on the held-out test set.
5. If probabilities themselves matter, check calibration and consider calibration methods such as Platt scaling or isotonic regression.

## Key takeaway

**A classifier's default threshold is only a starting point.** Separating probability estimation from the final decision threshold makes model evaluation and real-world decision-making much clearer.

## Next
Day 25 — **Regression Evaluation, Residual Analysis & Model Diagnostics**.
