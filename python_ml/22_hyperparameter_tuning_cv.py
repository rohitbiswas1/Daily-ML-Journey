"""Day 22: Hyperparameter tuning and cross-validation.

Practical goals:
- understand why a single train/test split can be noisy
- use cross-validation to estimate generalization more reliably
- tune LogisticRegression's C with GridSearchCV
- keep preprocessing inside a Pipeline to avoid leakage
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Baseline: one reasonable C value, evaluated on the held-out test set.
    baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    # Cross-validation is performed only on the training split.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    tuned_predictions = search.best_estimator_.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, tuned_predictions)

    print("Day 22 - Hyperparameter Tuning & Cross-Validation")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"CV folds: {cv.get_n_splits()} stratified folds")
    print("C values tested:", search.param_grid["model__C"])
    print(f"Best C: {search.best_params_['model__C']}")
    print(f"Best mean CV accuracy: {search.best_score_:.3f}")
    print(f"Baseline test accuracy (C=1): {baseline_accuracy:.3f}")
    print(f"Tuned test accuracy: {tuned_accuracy:.3f}")
    print("CV mean scores by C:")
    for params, mean_score, std_score in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
    ):
        print(
            f"  C={params['model__C']:>5}: "
            f"{mean_score:.3f} +/- {std_score:.3f}"
        )

    # A compact sanity check: CV should only use the training split.
    assert len(search.cv_results_["params"]) == 5
    assert np.isfinite(search.best_score_)


if __name__ == "__main__":
    main()
