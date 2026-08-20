"""Day 1: First Machine Learning example.

A tiny supervised-learning example using scikit-learn's LinearRegression.
The model learns a relationship between study hours and a simple score.
"""

from sklearn.linear_model import LinearRegression


def main() -> None:
    # Feature: hours studied
    X = [[1], [2], [3], [4], [5]]
    # Target: example test score
    y = [42, 50, 58, 66, 74]

    model = LinearRegression()
    model.fit(X, y)

    hours = [[6]]
    prediction = model.predict(hours)[0]

    print(f"Predicted score for 6 study hours: {prediction:.2f}")
    print(f"Learned slope: {model.coef_[0]:.2f}")
    print(f"Learned intercept: {model.intercept_:.2f}")


if __name__ == "__main__":
    main()
