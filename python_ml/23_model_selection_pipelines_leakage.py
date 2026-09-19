"""Day 23: model selection, pipelines, and data leakage.

Practical goals:
- compare simple classifiers with the same cross-validation strategy
- keep preprocessing inside a Pipeline
- demonstrate why fitting a scaler before cross-validation can leak information
- select a final model using training-only cross-validation before touching the test set
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Each preprocessing step is fitted independently inside every CV fold.
    models = {
        "logistic regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "decision tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=42,
        ),
    }

    print("Day 23 - Model Selection, Pipelines & Data Leakage")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("\n5-fold CV on training data:")

    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
        )
        scores[name] = cv_scores.mean()
        print(
            f"  {name:20s}: "
            f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}"
        )

    selected_name = max(scores, key=scores.get)
    selected_model = models[selected_name]
    selected_model.fit(X_train, y_train)
    test_predictions = selected_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"\nSelected from CV: {selected_name}")
    print(f"Held-out test accuracy: {test_accuracy:.3f}")

    # Leakage demonstration: scaling the complete dataset before CV lets
    # each validation fold influence the scaler's mean/std statistics.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    leaky_knn = KNeighborsClassifier(n_neighbors=7)
    leaky_scores = cross_val_score(
        leaky_knn,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="accuracy",
    )
    safe_scores = cross_val_score(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )

    print("\nLeakage comparison (illustrative):")
    print(f"  Pre-scaled before CV: {leaky_scores.mean():.3f}")
    print(f"  Pipeline-safe CV:     {safe_scores.mean():.3f}")
    print("Best practice: put learned preprocessing inside the Pipeline.")

    assert np.isfinite(test_accuracy)
    assert set(scores) == set(models)


if __name__ == "__main__":
    main()
