# Day 3 — Data Inspection with NumPy and Pandas

**Date:** 2026-08-22

## What I learned
- Pandas DataFrames are useful for loading and inspecting tabular ML data.
- NumPy provides numerical arrays and operations.
- `shape`, `dtypes`, `isna()`, and `describe()` are basic data inspection tools.
- Correlation can reveal relationships, but it does not prove causation.

## What I practiced
- Added a small CSV dataset with study hours, attendance, assignments completed, and score.
- Loaded the CSV with Pandas.
- Checked dimensions, data types, missing values, and summary statistics.
- Converted the score column to a NumPy array and calculated its mean.
- Calculated correlations with the target score.

## Experiment result
The dataset contains 8 rows and 4 columns, with no missing values. The mean score is 66.25. In this practice dataset, attendance has an absolute correlation of 0.997 with score. These values describe this small learning dataset only and should not be treated as real-world evidence.

## Reflection
Today I focused on understanding data before modeling it. Inspecting the dataset first makes later preprocessing and model evaluation more reliable.

## Next
Learn basic data cleaning and preprocessing, including handling missing values and preparing features for a model.
