# Day 2 — Train/Test Split and Model Evaluation

A model should be evaluated on data that was not used for fitting. This helps estimate how well it may perform on unseen examples.

## Core ideas

- **Training set:** data used to learn model parameters.
- **Test set:** held-out data used after training for evaluation.
- **Metric:** a numerical measure of prediction quality.

## Metrics practiced

### Mean Absolute Error (MAE)
`MAE = average(|y_true - y_pred|)`

Lower is better. MAE uses the same units as the target.

### Root Mean Squared Error (RMSE)
`RMSE = sqrt(average((y_true - y_pred)^2))`

Lower is better. Squaring makes larger errors contribute more strongly.

### R² score
R² compares the model with a baseline that always predicts the mean target.

- `1.0` means a perfect fit on the evaluated data.
- `0` means no improvement over the mean baseline.
- A negative value means the model is worse than that baseline.

## Practical experiment

The Day 2 script uses a small synthetic study-hours dataset, creates a 70/30 train/test split with `random_state=42`, trains linear regression on the training portion, and evaluates the held-out portion.

The run produced:

- MAE: **2.0792**
- RMSE: **2.2844**
- R²: **0.9858**

These results describe this small synthetic dataset only. They are not evidence that the model will perform similarly on real-world data.

## Key lesson

A prediction by itself does not tell us whether a model is reliable. Train/test separation and evaluation metrics make the workflow more disciplined and measurable.

## Next step

Practice NumPy/Pandas data handling and exploratory data analysis.
