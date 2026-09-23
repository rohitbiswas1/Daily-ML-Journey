"""Day 27: save, load, and serve a trained ML model locally.

Practical goals:
- train a small classification pipeline
- serialize the fitted pipeline with joblib
- load the model back as if it were a deployed artifact
- expose a tiny prediction function that represents an inference endpoint
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACT_PATH = Path("data/day27_iris_model.joblib")


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def serve_prediction(model: Pipeline, features: list[float]) -> dict[str, object]:
    """Return a JSON-like prediction payload for one request."""
    prediction = int(model.predict([features])[0])
    probabilities = model.predict_proba([features])[0]
    return {
        "predicted_class": prediction,
        "predicted_label": load_iris().target_names[prediction],
        "confidence": round(float(probabilities[prediction]), 4),
    }


def main() -> None:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.20,
        random_state=42,
        stratify=data.target,
    )

    model = build_model()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)

    loaded_model = joblib.load(ARTIFACT_PATH)
    loaded_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))

    sample = X_test[0].tolist()
    response = serve_prediction(loaded_model, sample)

    print("Day 27 - Model Deployment Basics")
    print(f"Test accuracy before saving: {accuracy:.3f}")
    print(f"Test accuracy after loading:  {loaded_accuracy:.3f}")
    print(f"Saved model artifact: {ARTIFACT_PATH}")
    print(f"Sample inference response: {response}")

    assert accuracy == loaded_accuracy
    assert set(response) == {"predicted_class", "predicted_label", "confidence"}
    assert 0.0 <= response["confidence"] <= 1.0


if __name__ == "__main__":
    main()
