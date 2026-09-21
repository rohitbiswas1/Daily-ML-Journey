# Day 25 — Regression Evaluation, Residual Analysis & Model Diagnostics

## What I learned
- MAE, RMSE, and R² answer different evaluation questions.
- Residuals are the difference between actual and predicted values.
- Residual patterns can reveal bias, missing nonlinear structure, changing error spread, or unusual observations.
- A more complex regression model should be judged on held-out data, not only training performance.

## Practical work
I used the built-in scikit-learn diabetes regression dataset and compared Linear Regression with degree-2 Polynomial Regression. Both models were evaluated on the same test split.

The practice script reports MAE, RMSE, R², residual mean, residual standard deviation, and the largest absolute residual.

## Reflection
Today connected familiar regression metrics to model diagnosis. A single score can tell me whether one model is better overall, but residuals help explain whether the model is making systematic mistakes. I also practiced keeping the test set separate while comparing model complexity.

## Next topic
Day 26 — Model Interpretability: Feature Importance, Permutation Importance & SHAP Intuition.
