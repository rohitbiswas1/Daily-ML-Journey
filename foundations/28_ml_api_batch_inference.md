# Day 28 — ML APIs & Batch Inference: Turning a Model into a Usable Service

## 1. From model artifact to application

Day 27 created a saved model artifact. Today the focus is the next boundary: accepting new inputs and returning predictions.

A simple ML application has this flow:

```text
client input → validation → loaded model → prediction → response
```

For many systems there are two common inference patterns:

- **Online inference:** predict one request at a time with low latency.
- **Batch inference:** predict many rows together on a schedule or when a file arrives.

## 2. API-style inference

A web API commonly accepts JSON. For the Iris model, one request can contain four named features:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The application should validate:

1. required feature names;
2. numeric types;
3. feature order expected by the model;
4. reasonable input constraints when the domain provides them.

The practice code keeps the API layer framework-free. A FastAPI `POST /predict` route could later call the same `predict_one()` function.

## 3. Why validation belongs before prediction

A model assumes its input schema is correct. Without validation, malformed data can produce errors or, worse, plausible but meaningless predictions.

A useful boundary is:

```text
raw JSON
  ↓
validate schema
  ↓
ordered numeric features
  ↓
model.predict / predict_proba
  ↓
JSON response
```

## 4. Batch inference

Batch inference is useful when predictions do not need an immediate response. Examples include:

- scoring yesterday's transactions every morning;
- generating recommendations for a user table;
- processing uploaded CSV files;
- refreshing risk scores on a schedule.

The Day 28 experiment reads `data/day28_batch_inputs.csv`, predicts every row using the saved Day 27 pipeline, and writes:

```text
data/day28_batch_predictions.csv
```

The output includes the original features, predicted class, label, and confidence.

## 5. Online vs batch trade-offs

| Pattern | Online inference | Batch inference |
|---|---|---|
| Input | One request | Many rows |
| Response | Immediate | Saved/processed later |
| Priority | Latency | Throughput |
| Typical use | Web/app prediction | Scheduled scoring |
| Failure handling | Per request | Per row/file/job |

The right choice depends on product requirements rather than model type alone.

## 6. Practical experiment

Run Day 27 first if the model artifact does not exist:

```bash
python python_ml/27_model_deployment.py
python python_ml/28_ml_api_batch_inference.py
```

The Day 28 script:

1. loads the trusted Day 27 artifact;
2. validates an API-style dictionary;
3. returns a JSON-like prediction response;
4. reads a small CSV batch;
5. predicts every row;
6. writes a prediction CSV.

## 7. Production considerations

A real ML API needs more than `predict()`:

- request validation and authentication;
- model and API versioning;
- timeouts and error handling;
- logging and monitoring;
- latency and throughput measurements;
- safe artifact loading;
- reproducible Python dependencies;
- privacy and access controls for sensitive inputs.

Batch systems additionally need job retries, idempotency, data-quality checks, and a way to identify which model version produced each output.

## Key takeaway

A trained model becomes useful when an application can reliably send new data to it. Online APIs optimize for immediate responses, while batch inference optimizes for processing many records efficiently. Both should reuse the same validated model artifact rather than retraining during prediction.
