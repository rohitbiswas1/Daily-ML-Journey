# Day 16 — K-Nearest Neighbors & Distance-Based Learning

## Learned
- How KNN predicts from nearby labeled examples.
- How the choice of k affects sensitivity to noise and smoothness.
- Why feature scaling matters for distance calculations.
- Why KNN can be simple but more expensive at prediction time.

## Practiced
Compared KNN classifiers across several k values on the breast cancer dataset, then repeated the experiment after standardizing the features.

## Experiment
The script python_ml/16_knn_distance_learning.py prints test accuracy for unscaled and standardized features.

## Reflection
The main practical lesson is that preprocessing can change a distance-based model substantially. I should test a sensible range of k values instead of assuming one value is always best.

## Next
Day 17 — Support Vector Machines (SVM) and margins.
