"""Day 28: API-style predictions and batch inference.

Practical goals:
- reuse a saved model artifact instead of retraining
- validate a small JSON-like prediction request
- run predictions for a CSV batch
- write predictions with confidence scores to a new CSV

This keeps the API idea framework-free so the ML concepts stay clear.
A FastAPI route can later call the same predict_one() function.
"""

from pathlib import Path
import csv
import json

import joblib
from sklearn.datasets import load_iris


ARTIFACT_PATH = Path("data/day27_iris_model.joblib")
INPUT_PATH = Path("data/day28_batch_inputs.csv")
OUTPUT_PATH = Path("data/day28_batch_predictions.csv")
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAMES = load_iris().target_names


def load_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ARTIFACT_PATH}. Run python_ml/27_model_deployment.py first."
        )
    return joblib.load(ARTIFACT_PATH)


def validate_features(payload: dict[str, object]) -> list[float]:
    """Validate one API-style request and return ordered numeric features."""
    missing = [name for name in FEATURE_NAMES if name not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    try:
        features = [float(payload[name]) for name in FEATURE_NAMES]
    except (TypeError, ValueError) as exc:
        raise ValueError("All feature values must be numeric") from exc

    return features


def predict_one(model, payload: dict[str, object]) -> dict[str, object]:
    features = validate_features(payload)
    prediction = int(model.predict([features])[0])
    probabilities = model.predict_proba([features])[0]
    return {
        "predicted_class": prediction,
        "predicted_label": TARGET_NAMES[prediction],
        "confidence": round(float(probabilities[prediction]), 4),
    }


def run_batch(model) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result = predict_one(model, row)
            rows.append({**row, **result})
    return rows


def save_predictions(rows: list[dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FEATURE_NAMES + ["predicted_class", "predicted_label", "confidence"]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    model = load_model()

    api_request = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    api_response = predict_one(model, api_request)

    batch_results = run_batch(model)
    save_predictions(batch_results)

    print("Day 28 - ML APIs & Batch Inference")
    print("API-style response:", json.dumps(api_response, indent=2))
    print(f"Processed batch rows: {len(batch_results)}")
    print(f"Saved predictions: {OUTPUT_PATH}")

    assert len(batch_results) > 0
    assert set(api_response) == {"predicted_class", "predicted_label", "confidence"}
    assert 0.0 <= api_response["confidence"] <= 1.0


if __name__ == "__main__":
    main()
