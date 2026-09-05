# Day 10 — Feature Scaling: Normalization and Standardization

## Why scaling matters
ML datasets often contain features measured on very different scales. For example, study hours might range from 1–10 while income might be measured in thousands. Distance-based and gradient-based algorithms can be affected when one feature dominates because of its numeric scale.

## Min-Max normalization
Min-Max scaling maps a feature into a chosen range, commonly 0 to 1:

`x_scaled = (x - min(x)) / (max(x) - min(x))`

It is useful when a bounded range is desirable, but it can be sensitive to extreme values.

## Standardization
Standardization centers a feature around mean 0 and scales it to standard deviation 1:

`z = (x - mean) / standard_deviation`

The resulting values are not necessarily between 0 and 1.

## When scaling is important
Scaling is especially useful for algorithms that depend on distances or feature magnitudes, such as K-nearest neighbors, K-means, SVMs, and many gradient-based models.

Tree-based models generally do not require feature scaling because their splits depend on feature thresholds rather than geometric distance.

## Data leakage warning
In a train/test workflow, fit the scaler using **training data only**, then transform both training and test data with that fitted scaler. Fitting on the complete dataset can leak information from the test set into training.

## Practice
`python_ml/10_feature_scaling.py` compares Min-Max normalization and standardization on two features with very different units.

## Key takeaway
Scaling changes the representation of feature values, not the observations themselves. The right method depends on the algorithm and the data distribution.

## Next
Learn linear regression more deeply, including coefficients, predictions, residuals, and the intuition behind the best-fit line.
