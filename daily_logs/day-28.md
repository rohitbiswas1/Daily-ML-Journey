# Day 28 — Daily Learning Log

## Topic
ML APIs & Batch Inference: Turning a Model into a Usable Service

## What I learned
- A trained model becomes useful when an application can send new data to it reliably.
- Online inference handles individual requests with a focus on latency.
- Batch inference processes many rows together and is useful for scheduled scoring.
- Input validation protects the model from missing, malformed, or misordered features.
- The inference service should load a trusted model artifact rather than retraining for each request.

## What I practiced
- Reused the Day 27 Iris `joblib` model artifact.
- Built a framework-free `predict_one()` function that behaves like an API endpoint.
- Added validation for the four expected Iris feature names and numeric values.
- Created a small CSV batch with five prediction requests.
- Generated a prediction CSV containing class, label, and confidence for each row.

## Experiment
The workflow was:

```text
API-style JSON → validate → loaded model → prediction + confidence

CSV batch → validate each row → loaded model → predictions → output CSV
```

The practice deliberately avoids retraining during inference. The same saved pipeline from Day 27 handles both online-style and batch predictions.

## Reflection
I learned that serving an ML model is an application-design problem as well as an ML problem. A good inference boundary needs a clear input schema, validation, predictable output, and a versioned model artifact. Batch inference is a useful alternative when predictions do not need to be returned immediately.

## Next
Day 29 — Monitoring, Data Drift & Model Reliability
