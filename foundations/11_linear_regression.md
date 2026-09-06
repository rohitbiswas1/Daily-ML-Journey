# Day 11 — Linear Regression in Depth

## What is linear regression?
Linear regression predicts a continuous target by learning a relationship between input features and the target. With one feature, the model can be written as:

`y_hat = b0 + b1*x`

Here, `b0` is the intercept and `b1` is the coefficient.

## Coefficients and intercept
The coefficient describes how much the predicted target changes for a one-unit increase in a feature, while holding other features constant. The intercept is the predicted value when the feature is zero.

## Residuals
A residual is the difference between an observed value and its prediction:

`residual = y - y_hat`

Small, pattern-free residuals generally indicate that the fitted line captures the main relationship reasonably well.

## Evaluation metrics
- **MAE:** average absolute prediction error; easy to interpret in target units.
- **MSE:** average squared error; penalizes larger errors more strongly.
- **R^2:** proportion of target variance explained by the model, with 1 representing a perfect fit.

## Practice
`python_ml/11_linear_regression.py` fits a simple regression model, prints its coefficient and intercept, calculates residuals and evaluation metrics, and predicts a new value.

## ML intuition
Linear regression is a useful baseline because it is simple and interpretable. It also introduces core ideas used throughout ML: learning parameters from data, generating predictions, measuring error, and checking generalization.

## Key takeaway
A fitted regression line is not just a visual trend: it is a parameterized function that converts features into predictions. Always evaluate how well those predictions match observed targets.

## Next
Learn classification fundamentals and how classification differs from regression.
