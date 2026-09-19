# Day 23 — Daily Log

## Topic
Model Selection, Pipelines & Preventing Data Leakage

## What I practiced
- Compared Logistic Regression, KNN, and a shallow Decision Tree with the same 5-fold stratified CV setup.
- Selected a candidate using training-only CV and evaluated it once on the held-out test set.
- Used `Pipeline` to keep `StandardScaler` and the estimator together.
- Ran a small leakage demonstration by comparing KNN CV after pre-scaling with pipeline-safe CV.

## What I understood
The important distinction is between **model evaluation** and **model selection**. The test set should not influence which model I choose. Cross-validation gives me a fairer basis for selection while the final test set remains untouched until the end.

I also understood why preprocessing belongs inside a Pipeline. A scaler learns statistics such as mean and standard deviation. If those statistics are learned using validation data, the validation fold is no longer fully independent.

## Reflection
Today connected several earlier topics—scaling, KNN, trees, logistic regression, and cross-validation—into one reliable workflow. The main lesson is that a good ML score is only meaningful when the evaluation process is clean.

## Next topic
Day 24 — Classification Thresholds, Calibration & Probability Quality.
