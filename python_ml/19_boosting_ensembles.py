"""Day 19: Boosting and gradient-based ensembles.

Practical experiment: compare a single decision tree with AdaBoost and
Gradient Boosting on a classification dataset, then inspect how depth and
number of estimators affect performance.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(
        f"{name}: accuracy={accuracy_score(y_test, predictions):.3f}, "
        f"precision={precision_score(y_test, predictions):.3f}, "
        f"recall={recall_score(y_test, predictions):.3f}, "
        f"f1={f1_score(y_test, predictions):.3f}"
    )
    return model


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    print("Model comparison")
    evaluate(
        "Decision Tree",
        DecisionTreeClassifier(max_depth=3, random_state=42),
        X_train, X_test, y_train, y_test,
    )
    evaluate(
        "AdaBoost",
        AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        X_train, X_test, y_train, y_test,
    )
    evaluate(
        "Gradient Boosting",
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42),
        X_train, X_test, y_train, y_test,
    )

    print("\nGradient Boosting experiment: estimator count")
    for estimators in (20, 50, 100, 200):
        model = GradientBoostingClassifier(
            n_estimators=estimators,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"n_estimators={estimators:3d} -> accuracy={accuracy_score(y_test, predictions):.3f}")

    print("\nKey ideas:")
    print("- Boosting builds models sequentially so later learners focus on earlier errors.")
    print("- AdaBoost reweights difficult training examples after each weak learner.")
    print("- Gradient Boosting adds trees that reduce the remaining prediction error.")
    print("- Learning rate and number/depth of estimators control the fit and overfitting risk.")


if __name__ == "__main__":
    main()
