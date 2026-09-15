# Day 19 — Boosting & Gradient-Based Ensembles

## Goal
Understand why combining many weak learners sequentially can produce a strong predictive model.

## 1. What is Boosting?
Boosting is an ensemble strategy that trains models one after another. Each new learner is influenced by the errors made by the current ensemble. The idea is to turn several simple learners into a stronger model.

This differs from Random Forests (Day 15), where trees are trained independently and then combined. Boosting is sequential and error-focused.

## 2. AdaBoost
AdaBoost starts with simple weak learners, commonly shallow decision trees. After each round it increases the influence of examples that were harder to classify, so later learners pay more attention to them.

Important parameters:
- `n_estimators`: number of boosting rounds.
- `learning_rate`: contribution of each learner.

## 3. Gradient Boosting
Gradient Boosting also builds an additive model sequentially, but each new tree is trained to improve the current model by approximating the remaining loss/error signal.

Important parameters:
- `n_estimators`: number of trees.
- `learning_rate`: step size for each tree.
- `max_depth`: complexity of each tree.

A smaller learning rate often needs more estimators. The combination controls the bias/variance trade-off.

## 4. Practical Experiment
`python_ml/19_boosting_ensembles.py` compares:
1. A shallow Decision Tree.
2. AdaBoost.
3. Gradient Boosting.

It then varies Gradient Boosting's estimator count across 20, 50, 100, and 200 trees while keeping the learning rate and tree depth fixed.

Run:

```bash
python python_ml/19_boosting_ensembles.py
```

The script uses scikit-learn's built-in breast-cancer classification dataset, so no external download is required.

## 5. What to Observe
- A boosted ensemble can improve over a single shallow tree.
- More estimators do not automatically guarantee better test performance.
- Learning rate, tree depth, and estimator count should be considered together.
- Stronger ensembles can still overfit when learners are too complex or the boosting process is pushed too far.

## Key Takeaways
- Bagging trains learners independently; boosting trains them sequentially.
- AdaBoost emphasizes difficult examples.
- Gradient Boosting fits new learners to improve the remaining loss.
- Hyperparameter tuning is important because boosting is powerful but can overfit.
