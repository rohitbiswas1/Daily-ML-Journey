# Day 4 — Data Cleaning and Preprocessing

## Why cleaning matters
Real-world ML data is rarely ready to use directly. Missing values, duplicate rows, and categorical values can cause errors or reduce model quality.

## Key concepts

### 1. Missing values
Use `isna()` to find missing values. The right treatment depends on the data and problem.

Common options:
- Remove rows when the missing record is not useful.
- Fill numeric values with a statistic such as the median.
- Fill categorical values with the mode or a meaningful category.

### 2. Duplicate rows
`duplicated()` identifies repeated observations and `drop_duplicates()` removes exact duplicates.

### 3. Categorical encoding
Many ML algorithms expect numeric input. One-hot encoding converts categories into binary columns without imposing a false numeric order.

### 4. Features and target
The target is what the model tries to predict. Features are the input variables used to make that prediction.

## Important lesson
Cleaning decisions should be based on the meaning of the data. Imputation and encoding are not automatic guarantees of better models; they are preprocessing choices that should be validated later.

## Day 4 workflow
1. Load the raw dataset.
2. Inspect missing values and duplicates.
3. Remove duplicate observations.
4. Impute missing numeric values with the median.
5. Impute a missing categorical value with the mode.
6. One-hot encode the categorical feature.
7. Separate features (`X`) from target (`y`).
