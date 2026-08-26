# Day 4 — Data Cleaning and Preprocessing

**Date:** 2026-08-26

## What I learned
- Real-world datasets can contain missing values, duplicate rows, and categorical features.
- Missing numeric values can be imputed with statistics such as the median when that choice makes sense.
- Categorical values can be converted to numeric features with one-hot encoding.
- Duplicate observations should be investigated and removed when they are accidental duplicates.
- A clean ML dataset should clearly separate input features from the target.

## What I practiced
- Created a small intentionally imperfect student-performance dataset.
- Used Pandas to count missing values and duplicate rows.
- Removed one exact duplicate row.
- Filled missing `study_hours` and `assignments_completed` values with their medians.
- Filled a missing `study_mode` value with the most frequent category.
- One-hot encoded `study_mode`.
- Split the cleaned table into features `X` and target `y`.

## Experiment result
The raw dataset contained 8 rows, including 1 exact duplicate. It also contained one missing value in each of `study_hours`, `assignments_completed`, and `study_mode`. After cleaning, the dataset contained 7 rows with no missing values. The cleaned target mean was approximately **69.71**.

These numbers are from a small learning dataset and are not evidence about real student performance.

## Reflection
Today I learned that preprocessing is part of the ML workflow, not an afterthought. A model can only learn reliably when the data has been inspected and transformed carefully. I also learned that preprocessing decisions should be justified by the meaning of the data rather than applied blindly.

## Next
Learn basic statistics for Machine Learning: mean, median, variance, standard deviation, distributions, and the intuition behind probability.
