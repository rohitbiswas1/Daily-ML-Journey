# Day 16 — K-Nearest Neighbors (KNN) & Distance-Based Learning

## What I learned
K-Nearest Neighbors is a supervised learning algorithm that predicts a new sample from the labels of its closest training examples. For classification, the majority class among the `k` nearest neighbors is selected.

### Core ideas
- **Distance:** Euclidean distance is common, but the choice depends on the data.
- **k:** Small values can be sensitive to noise; large values make predictions smoother but can underfit.
- **Scaling:** Distance-based models are sensitive to feature units, so standardization or normalization is often important.
- **Training:** KNN has little explicit model fitting; much of the work happens when making predictions.
- **Complexity:** Prediction can become expensive as the training set grows because neighbors must be searched.

## Practical experiment
`python_ml/16_knn_distance_learning.py` compares KNN with `k = 1, 3, 5, 11, 21` on the breast cancer dataset, both before and after `StandardScaler` preprocessing.

The experiment demonstrates that scaling can change KNN performance because features with larger numeric ranges otherwise contribute more strongly to distance.

## Beginner-to-intermediate takeaway
KNN is a useful baseline when the idea of "similar examples have similar labels" makes sense. Always inspect feature scales before trusting a distance-based model.

## Reflection
I learned that choosing `k` is a bias-variance trade-off and that preprocessing is not just cosmetic: for KNN, feature scale directly changes which samples count as neighbors.
