# Day 7 — Distributions and Sampling

## What I learned
- A distribution describes how values are spread across possible outcomes.
- Symmetric data has a similar shape on both sides of its center.
- Right-skewed data has a longer tail toward larger values; extreme high values can pull the mean above the median.
- A population is the full group we want to understand; a sample is a subset used to estimate it.
- Different random samples can produce different statistics. This is sampling variation.

## Sampling practice
The Day 7 script creates a population of 10,000 normally distributed measurements, takes random samples of 100 observations, and repeats this 500 times. It compares the population mean with the distribution of sample means.

## Why this matters for ML
ML datasets are usually samples from a larger real-world process. Sampling affects training data, evaluation results, and how confidently we generalize to unseen data.

## Key intuition
A sample statistic is an estimate, not a guaranteed description of the population. Larger representative samples generally give more stable estimates, while biased samples can lead to misleading conclusions.

## Practice
Run `python_ml/07_distributions_sampling.py` to reproduce the experiment with a fixed random seed.

## Next
Learn correlation and covariance and connect them to feature relationships.
