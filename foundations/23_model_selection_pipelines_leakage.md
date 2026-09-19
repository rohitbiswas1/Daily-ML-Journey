# Day 23 — Model Selection, Pipelines & Preventing Data Leakage

## What I learned
- Model selection means choosing a model or configuration using a validation strategy, not by repeatedly checking the test set.
- Cross-validation gives a more stable estimate than relying on one training/validation split.
- A `Pipeline` keeps preprocessing and the estimator together so learned preprocessing is fitted only on each training fold.
- Data leakage happens when information from validation/test data influences training or preprocessing.
- The final test set should be used once, after the model-selection decisions are finished.

## Practical workflow
1. Split the original data into training and held-out test sets.
2. Define candidate models.
3. Put learned preprocessing such as scaling inside a `Pipeline`.
4. Compare candidates with the same stratified cross-validation folds.
5. Select the best candidate from training-only CV results.
6. Fit that selected pipeline on all training data.
7. Evaluate once on the untouched test set.

## Experiment
The practice script compares Logistic Regression, KNN, and a Decision Tree on scikit-learn's breast-cancer dataset. It also compares a KNN evaluation where scaling is performed before CV with the safer Pipeline approach. The difference may be small on this dataset, but the workflow demonstrates the important rule: **validation folds must not influence preprocessing fitted for another fold**.

## Key takeaway
A high validation score is not useful if the validation data leaked into preprocessing or model decisions. Treat the test set as a final exam, and use pipelines plus cross-validation for the training process.

## Next
Day 24 — **Classification Thresholds, Calibration & Probability Quality**.
