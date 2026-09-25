# Day 29 — Monitoring, Data Drift & Model Reliability

## What is monitoring?
A deployed ML model needs checks after release because data and real-world behavior can change.

## Data drift
Data drift means the distribution of current model inputs differs from a reference period. Monitor summary statistics, missing values, category frequencies, ranges, and distribution changes.

## PSI
Population Stability Index summarizes binned distribution change:

PSI = sum((current% - reference%) * ln(current% / reference%))

PSI is a signal for investigation, not proof that a model has failed. Thresholds should be validated for the specific application.

## Prediction and performance monitoring
Watch prediction/class rates and probability distributions. When fresh labels arrive, track precision, recall, F1, log loss, MAE, RMSE, or R2 as appropriate.

## Practical experiment
Run:

```bash
python python_ml/29_monitoring_drift.py
```

The script creates reproducible reference and current distributions with a controlled mean shift, calculates PSI, and compares positive-prediction rates.

## Reliability checklist
1. Check schema and data quality.
2. Check feature distributions for meaningful change.
3. Check prediction rates and confidence distributions.
4. Check fresh-label performance when labels arrive.
5. Alert on sustained/material changes rather than one noisy sample.
6. Define an investigation and response process.

## Key takeaway
**Drift is a signal, not a verdict.** Reliable ML systems combine data-quality checks, drift monitoring, prediction monitoring, and fresh-label performance evaluation.
