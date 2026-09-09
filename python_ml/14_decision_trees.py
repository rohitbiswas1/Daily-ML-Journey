"""Day 14: Decision Trees and interpretability.

Practical experiment: compare shallow and deeper decision trees on a built-in
classification dataset, then inspect feature importance and readable rules.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"\n{name}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy:     {test_acc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))
    return model, test_pred


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    shallow, _ = evaluate(
        DecisionTreeClassifier(max_depth=3, random_state=42),
        X_train, X_test, y_train, y_test, "Shallow tree (max_depth=3)"
    )
    deep, deep_pred = evaluate(
        DecisionTreeClassifier(max_depth=None, random_state=42),
        X_train, X_test, y_train, y_test, "Unrestricted tree"
    )

    print("\nDeep-tree classification report:")
    print(classification_report(y_test, deep_pred, target_names=data.target_names))

    ranked = sorted(
        zip(data.feature_names, shallow.feature_importances_),
        key=lambda item: item[1], reverse=True
    )
    print("Top shallow-tree feature importances:")
    for name, importance in ranked[:5]:
        print(f"  {name}: {importance:.3f}")

    print("\nFirst decision rules from the shallow tree:")
    print(export_text(shallow, feature_names=list(data.feature_names), max_depth=2))

    print("Interpretation: compare training vs test accuracy to spot overfitting.")


if __name__ == "__main__":
    main()
