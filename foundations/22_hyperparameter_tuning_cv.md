# Day 22 — Hyperparameter Tuning & Cross-Validation

## Why tune hyperparameters?
A model has **parameters** learned from data and **hyperparameters** chosen before training. Hyperparameters control the learning process and can strongly affect generalization.

For Logistic Regression, `C` controls the strength of regularization:
- smaller `C` → stronger regularization
- larger `C` → weaker regularization

The best value depends on the dataset, so we should compare candidates rather than guessing.

## Why cross-validation?
A single train/test split can make a model look unusually good or bad by chance. **K-fold cross-validation** repeatedly splits the training data into training and validation folds, trains on each training portion, and averages the validation scores.

With stratified K-fold cross-validation for classification, each fold tries to preserve the class proportions.

## Important workflow
1. Hold out the test set at the beginning.
2. Use only the training data for model selection and hyperparameter tuning.
3. Put preprocessing inside a `Pipeline` so each CV fold learns transformations only from its own training portion.
4. Use `GridSearchCV` to evaluate a defined hyperparameter grid.
5. Refit the best configuration on the full training data.
6. Evaluate the selected model once on the untouched test set.

## Why the Pipeline matters
Scaling before cross-validation can leak information from validation folds into training folds. A `Pipeline` keeps `StandardScaler` and `LogisticRegression` together, so scaling is fitted separately inside every CV training fold.

## Practical experiment
`python_ml/22_hyperparameter_tuning_cv.py` uses scikit-learn's built-in breast-cancer dataset, so no external download is required. It compares a baseline Logistic Regression model (`C=1`) with a `GridSearchCV` search over five `C` values using five stratified folds.

The script reports:
- best `C`
- mean CV accuracy and its spread
- test accuracy of the baseline
- test accuracy of the tuned model
- CV score for each candidate `C`

## Key takeaway
Cross-validation helps estimate how a configuration generalizes during model selection, while the final test set should remain untouched until the end. Hyperparameter tuning is not about finding a universally best value; it is about selecting a reasonable configuration for the data and objective without leaking information.
