# Day 27 — Daily Learning Log

## Topic
Model Deployment Basics: Saving, Loading & Serving a Model

## What I learned
- Training and inference are separate stages of an ML system.
- A model artifact can package preprocessing and the estimator together.
- `joblib` can serialize a scikit-learn pipeline for a Python deployment workflow.
- A serving layer should load a fitted artifact and call `predict()`/`predict_proba()` rather than retraining.
- Input schema, dependency versions, artifact provenance, and monitoring matter after training.

## What I practiced
- Built a `StandardScaler` + Logistic Regression pipeline on the Iris dataset.
- Evaluated the model on a held-out test set.
- Saved the complete pipeline to `data/day27_iris_model.joblib`.
- Loaded the artifact into a fresh object and verified the same test accuracy.
- Created a small `serve_prediction()` function that returns a JSON-like inference response.

## Experiment
The key check was whether the model behaved identically before and after serialization:

```text
trained pipeline → test accuracy
       ↓
     joblib.dump
       ↓
     joblib.load
       ↓
loaded pipeline → test accuracy + sample prediction
```

The experiment uses the built-in Iris dataset, so it stays small and reproducible without introducing an external data dependency.

## Reflection
I learned that deployment is not just putting a model online. The preprocessing, model artifact, input format, dependency versions, and validation checks all need to stay consistent. Saving the whole pipeline is safer than saving only the estimator because inference then applies the same scaling automatically.

## Next
Day 28 — ML APIs & Batch Inference: Turning a Model into a Usable Service
