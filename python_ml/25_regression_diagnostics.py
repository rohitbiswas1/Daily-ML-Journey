"""Day 25: regression evaluation, residual analysis, and diagnostics.

Practical goals:
- compare MAE, RMSE, and R^2 on a held-out set
- inspect residual mean and spread
- compare a simple linear model with a more flexible polynomial model
- use residual patterns to think about underfitting and model assumptions
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> None:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    residuals = y_test - predictions

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R^2 : {r2:.3f}")
    print(f"  residual mean: {residuals.mean():.2f}")
    print(f"  residual std : {residuals.std():.2f}")
    print(f"  largest |residual|: {np.abs(residuals).max():.2f}")

    assert np.isfinite(predictions).all()


def main() -> None:
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.20, random_state=42
    )

    linear = LinearRegression()
    polynomial = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    print("Day 25 - Regression Evaluation & Diagnostics")
    evaluate("Linear Regression", linear, X_train, X_test, y_train, y_test)
    evaluate("Degree-2 Polynomial Regression", polynomial, X_train, X_test, y_train, y_test)

    print("\nDiagnostic reminder: residuals should ideally have no obvious systematic pattern.")
    print("Use MAE for average absolute error, RMSE to emphasize larger errors, and R^2 for explained variance.")


if __name__ == "__main__":
    main()
