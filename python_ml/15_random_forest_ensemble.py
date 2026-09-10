"""Day 15: Ensemble learning with Random Forests.

Practical experiment: compare a single decision tree with a random forest,
inspect test performance, and examine feature importance.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name}")
    print(f"Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.3f}")
    print(f"Test accuracy:  {accuracy_score(y_test, pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    return model, pred


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    tree, _ = evaluate(
        DecisionTreeClassifier(max_depth=5, random_state=42),
        X_train, X_test, y_train, y_test, "Single decision tree"
    )

    forest, forest_pred = evaluate(
        RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
        X_train, X_test, y_train, y_test, "Random forest (200 trees)"
    )

    print("\nRandom forest classification report:")
    print(classification_report(y_test, forest_pred, target_names=data.target_names))

    ranked = sorted(
        zip(data.feature_names, forest.feature_importances_),
        key=lambda item: item[1], reverse=True
    )
    print("Top 5 forest feature importances:")
    for name, importance in ranked[:5]:
        print(f"  {name}: {importance:.3f}")

    print("\nKey idea: many diverse trees can reduce variance compared with one tree.")
    print(f"Tree depth used: {tree.get_depth()} | Forest trees: {forest.n_estimators}")


if __name__ == "__main__":
    main()
