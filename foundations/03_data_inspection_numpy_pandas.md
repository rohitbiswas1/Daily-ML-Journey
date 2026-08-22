# Day 3 — Data Inspection with NumPy and Pandas

## Why this matters
Before training a model, the first job is to understand the data. Data inspection helps reveal the number of rows and columns, data types, missing values, summary statistics, and suspicious values.

## NumPy vs Pandas
- **NumPy** provides fast numerical arrays and mathematical operations.
- **Pandas** provides labeled Series and DataFrame structures for tabular data.
- In a typical ML workflow, Pandas is useful for loading and inspecting tables, while NumPy is useful for numerical operations and arrays.

## Core inspection commands
```python
import pandas as pd

df = pd.read_csv("data/study_progress.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df.describe())
```

## What to look for
1. **Shape** — how many examples and features are present?
2. **Data types** — are numeric columns actually numeric?
3. **Missing values** — do any columns need cleaning?
4. **Ranges** — are values plausible?
5. **Relationships** — do variables appear related, and could that relationship be misleading?

## Important ML lesson
A high correlation does not prove that one variable causes another. Correlation is useful for exploration, but model decisions should also consider the problem, data collection process, and possible leakage.

## Practice task
Load `data/study_progress.csv`, inspect it, calculate the mean score, and identify the feature most correlated with `score`. Then explain why correlation alone is not enough to choose a feature.
