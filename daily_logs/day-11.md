# Day 11 — Linear Regression in Depth

**Date:** 2026-09-06

## What I learned
- Linear regression predicts continuous values using learned coefficients.
- A simple model follows `y_hat = b0 + b1*x`.
- The coefficient represents the expected change in the prediction for a one-unit feature increase.
- Residuals measure prediction error as `y - y_hat`.
- MAE, MSE, and R^2 provide different views of model performance.

## What I practiced
- Fit a linear regression model with scikit-learn.
- Inspected the coefficient and intercept.
- Calculated predictions and residuals.
- Evaluated the fit with MAE, MSE, and R^2.
- Predicted a new target value from an unseen input.

## Experiment result
The practice dataset follows an exact linear relationship, so the fitted model recovers the expected line with zero residual error and an R^2 of 1.0.

## Reflection
Linear regression connects mathematical relationships with practical ML. Understanding coefficients, errors, and evaluation metrics provides a strong foundation for later supervised learning models.

## Next
Learn classification fundamentals and how classification differs from regression.
