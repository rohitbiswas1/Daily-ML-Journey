# Day 26 — Model Interpretability: Feature Importance, Permutation Importance & SHAP Intuition

## 1. Why interpretability matters

A strong score does not tell us why a model makes a prediction. Interpretability helps us inspect which inputs a trained model uses and whether its behavior is sensible enough for the problem.

For this beginner-to-intermediate experiment, we use a Random Forest classifier on scikit-learn's built-in Breast Cancer Wisconsin dataset.

## 2. Feature importance in a Random Forest

Random Forests expose `feature_importances_`, an impurity-based measure. It summarizes how much each feature contributes to reducing split impurity across the trees.

Useful for:
- a quick ranking of influential features
- feature exploration
- communicating a model at a high level

Important limitation: impurity importance is not a causal statement and can be biased by feature properties such as scale/cardinality and correlated predictors.

## 3. Permutation importance

Permutation importance starts with a trained model and a validation/test set. It repeatedly shuffles one feature column and measures how much the chosen score changes.

If shuffling a feature causes a large accuracy drop, the model was relying on information in that feature for prediction.

Conceptually:

```text
baseline score
      ↓
shuffle one feature
      ↓
score again
      ↓
importance ≈ baseline score - shuffled score
```

Permutation importance is often easier to explain because it is tied directly to a performance metric on held-out data. However, correlated features can make the result harder to interpret: if two columns contain similar information, shuffling one may not hurt much because the model can still use the other.

## 4. Feature importance is not causality

If a feature is highly important to a predictive model, that does **not** mean changing the feature will cause the target to change. Prediction and causal inference answer different questions.

Also watch for:
- target leakage
- proxy variables
- unstable importance across datasets
- correlated features
- distribution shift between training and deployment

## 5. SHAP intuition

SHAP (SHapley Additive exPlanations) is based on the idea of fairly distributing a prediction's difference from a baseline among the input features.

A simplified mental model is:

```text
prediction = baseline + feature contribution 1 + feature contribution 2 + ...
```

A positive SHAP contribution pushes a prediction toward the explained class/value, while a negative contribution pushes it away. SHAP can provide local explanations for individual predictions as well as aggregate views across a dataset.

This day focuses on the intuition rather than adding the external `shap` package, so the practice remains lightweight and reproducible with scikit-learn.

## 6. Practical experiment

Run:

```bash
python python_ml/26_model_interpretability.py
```

The script:
1. loads the breast-cancer dataset;
2. creates a stratified train/test split;
3. trains a Random Forest;
4. prints baseline test accuracy;
5. prints the top features from impurity importance;
6. computes permutation importance using repeated shuffles on the test set;
7. compares the two rankings and explains why they may differ.

## 7. What to look for

Do not focus only on the first-ranked feature. Compare the two importance methods and ask:

- Which features remain important under both methods?
- Which features change rank substantially?
- Could correlated features explain a small permutation score?
- Would the importance remain stable on another train/test split?

## Key takeaway

Interpretability is about understanding model behavior, not proving causation. Built-in feature importance is useful for a quick overview; permutation importance connects importance to predictive performance; SHAP provides a richer framework for explaining individual predictions and overall model behavior.
