"""Day 26: model interpretability with feature importance.

Practical goals:
- compare impurity-based and permutation feature importance
- see why a model's built-in importance is not the same as causal importance
- measure how much validation performance changes when one feature is shuffled
- build intuition for SHAP values without adding a new dependency
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.20,
        random_state=42,
        stratify=data.target,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    baseline_accuracy = accuracy_score(y_test, model.predict(X_test))
    permutation = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="accuracy",
        n_jobs=-1,
    )

    builtin_order = np.argsort(model.feature_importances_)[::-1][:10]
    permutation_order = np.argsort(permutation.importances_mean)[::-1][:10]

    print("Day 26 - Model Interpretability")
    print(f"Baseline test accuracy: {baseline_accuracy:.3f}")

    print("\nTop features by Random Forest impurity importance:")
    for index in builtin_order:
        print(f"  {data.feature_names[index]:30s} {model.feature_importances_[index]:.4f}")

    print("\nTop features by permutation importance:")
    for index in permutation_order:
        mean_drop = permutation.importances_mean[index]
        std_drop = permutation.importances_std[index]
        print(f"  {data.feature_names[index]:30s} drop={mean_drop:.4f} +/- {std_drop:.4f}")

    print("\nInterpretation:")
    print("- Impurity importance asks how much a feature helped split training trees.")
    print("- Permutation importance asks how much test performance falls when a feature is shuffled.")
    print("- A large permutation score suggests the trained model relies on that feature for prediction.")
    print("- These are predictive explanations, not proof that a feature causes the target.")
    print("- SHAP builds a different explanation: it estimates how each feature contribution moves a prediction away from a baseline.")

    assert 0.0 <= baseline_accuracy <= 1.0
    assert np.isfinite(permutation.importances_mean).all()


if __name__ == "__main__":
    main()
