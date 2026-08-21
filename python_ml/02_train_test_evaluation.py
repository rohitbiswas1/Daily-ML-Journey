"""Day 2: Train/test split and regression evaluation."""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


def main() -> None:
    # Small synthetic dataset for learning the evaluation workflow.
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([42, 50, 58, 66, 74, 81, 88, 91, 96, 99])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")


if __name__ == "__main__":
    main()
