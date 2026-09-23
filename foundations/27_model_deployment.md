# Day 27 — Model Deployment Basics: Saving, Loading & Serving a Model

## 1. Why deployment matters

Training a model is only one part of an ML system. A useful model must be packaged so another program can load it and make predictions on new data.

A simple deployment lifecycle is:

```text
train → validate → save artifact → load artifact → serve predictions → monitor
```

This day focuses on the first practical deployment step: creating a reproducible model artifact and using it for inference.

## 2. What is a model artifact?

A model artifact is a saved representation of a trained model and its preprocessing steps. For scikit-learn projects, `joblib` is a convenient option for Python-based workflows.

Saving a full `Pipeline` is important because preprocessing must match what was used during training. In this experiment the pipeline contains:

1. `StandardScaler`
2. `LogisticRegression`

Saving them together reduces the chance of forgetting the scaler during inference.

## 3. Training vs inference

Training learns parameters from historical data:

```text
training data → preprocessing → model.fit()
```

Inference uses the already-fitted artifact:

```text
new input → loaded pipeline → prediction + probability
```

The deployed process should **not** call `fit()` on incoming user data.

## 4. Practical experiment

The script uses scikit-learn's built-in Iris dataset so the experiment stays small and reproducible.

Run:

```bash
python python_ml/27_model_deployment.py
```

The script:
1. splits Iris into training and test sets;
2. trains a scaling + Logistic Regression pipeline;
3. evaluates test accuracy;
4. saves the complete pipeline to `data/day27_iris_model.joblib`;
5. loads the artifact into a fresh Python object;
6. verifies that predictions remain unchanged;
7. produces a JSON-like inference response for one sample.

## 5. What does “serving” mean here?

A production service might expose an HTTP endpoint such as:

```text
POST /predict
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

The server would load the artifact and return something like:

```text
{
  "predicted_class": 0,
  "predicted_label": "setosa",
  "confidence": 0.98
}
```

Our `serve_prediction()` function models this inference boundary without adding a web-framework dependency. A later project can wrap the same function in FastAPI or another serving framework.

## 6. Deployment safety checklist

Before deploying a model, check:

- **Artifact integrity:** Can the saved model be loaded successfully?
- **Preprocessing:** Are the same transformations applied at inference time?
- **Input schema:** Are feature order, names, types, and units correct?
- **Versioning:** Do you know which code and training data produced the artifact?
- **Dependencies:** Can the serving environment reproduce the model's Python/scikit-learn versions?
- **Validation:** Does the loaded model produce the same expected results as the trained model?
- **Monitoring:** Will you detect failures, drift, or degraded prediction quality?

## 7. Important caution about serialized models

Only load model files from trusted sources. Python serialization formats can execute code during deserialization. In a real deployment, protect the artifact supply chain and pin compatible dependencies.

## Key takeaway

Deployment starts with reproducibility. A trained model should become a portable artifact containing everything needed for consistent inference. Saving the preprocessing pipeline together with the estimator is safer than saving only the estimator.
