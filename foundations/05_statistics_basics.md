# Day 5 — Statistics for Machine Learning

## What I learned

Statistics helps us summarize data, understand variability, detect unusual values, and reason about model inputs and outputs.

### Mean
The arithmetic average:

`mean = sum(values) / number_of_values`

### Median
The middle value after sorting the data. Median is often more resistant to extreme values than the mean.

### Variance
Measures how far values tend to spread from the mean. Larger variance means greater spread.

### Standard deviation
The square root of variance. It expresses spread in the same units as the original data.

### Range
The difference between the maximum and minimum values.

## ML intuition
- Mean can summarize a typical value but can be pulled by outliers.
- Median can be useful when data is skewed or contains extreme observations.
- Variability matters because two datasets can have the same mean but very different distributions.
- Statistics describe a dataset; they do not automatically establish causation.

## Practice
`python_ml/05_statistics_basics.py` calculates these statistics for the existing study-progress dataset and compares mean versus median on symmetric and skewed examples.

## Next
Learn probability basics and how probability supports uncertainty, classification, and model evaluation.
