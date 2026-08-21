# Day 2 — Train/Test Split and Evaluation

**Date:** 2026-08-21

## What I learned
- A model should be tested on data that was not used during fitting.
- MAE measures average absolute prediction error.
- RMSE penalizes larger errors more strongly than MAE.
- R² compares the model with a mean-value baseline.

## What I practiced
- Created a 70/30 train/test split with a fixed random state.
- Trained linear regression only on the training portion.
- Evaluated the held-out test portion with MAE, RMSE, and R².

## Experiment result
On the small synthetic study-hours dataset, the run produced MAE **2.0792**, RMSE **2.2844**, and R² **0.9858**. These are learning-experiment results and should not be generalized to real-world performance.

## Reflection
Today I moved from simply making a prediction to measuring model performance. This is an important step toward a complete ML workflow.

## Next
Practice NumPy and Pandas for loading, inspecting, and transforming data.
