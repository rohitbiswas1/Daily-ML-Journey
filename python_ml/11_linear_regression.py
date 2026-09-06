"""Day 11: Linear Regression in Depth
Practice coefficients, predictions, residuals, and R^2.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13, 15, 17], dtype=float)

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
residuals = y - predictions

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predictions:", predictions)
print("Residuals:", residuals)
print("MAE:", mean_absolute_error(y, predictions))
print("MSE:", mean_squared_error(y, predictions))
print("R^2:", r2_score(y, predictions))
print("Prediction for 9 hours:", model.predict([[9]])[0])
