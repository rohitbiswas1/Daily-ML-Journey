"""Day 24: classification thresholds, calibration, and probability quality.

Practical goals:
- see how changing a classification threshold changes precision and recall
- compare predicted probabilities with actual outcomes using Brier score
- inspect calibration with a simple reliability table
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.20,
        random_state=42,
        stratify=data.target,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Day 24 - Classification Thresholds & Calibration")
    print(f"Brier score (lower is better): {brier_score_loss(y_test, probabilities):.3f}")
    print("\nThreshold comparison:")
    for threshold in (0.30, 0.50, 0.70):
        predictions = (probabilities >= threshold).astype(int)
        print(
            f"  threshold={threshold:.2f} | "
            f"accuracy={accuracy_score(y_test, predictions):.3f} | "
            f"precision={precision_score(y_test, predictions, zero_division=0):.3f} | "
            f"recall={recall_score(y_test, predictions, zero_division=0):.3f}"
        )

    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=5,
        strategy="quantile",
    )

    print("\nCalibration bins:")
    for predicted, observed in zip(mean_predicted, fraction_positive):
        print(f"  predicted={predicted:.3f} -> observed={observed:.3f}")

    assert np.isfinite(probabilities).all()
    assert 0.0 <= brier_score_loss(y_test, probabilities) <= 1.0


if __name__ == "__main__":
    main()
