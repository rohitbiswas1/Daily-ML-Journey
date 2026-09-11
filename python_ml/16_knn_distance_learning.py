"""Day 16: K-Nearest Neighbors (KNN) and distance-based learning.

Practical experiment: compare KNN with several k values and show why feature
scaling matters for a distance-based model.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def evaluate_knn(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    print("Unscaled KNN")
    for k in (1, 3, 5, 11, 21):
        print(f"k={k:2d}: test accuracy={evaluate_knn(X_train, X_test, y_train, y_test, k):.3f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled KNN")
    for k in (1, 3, 5, 11, 21):
        print(f"k={k:2d}: test accuracy={evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, k):.3f}")

    print("\nKey idea: KNN predicts from nearby training examples, so distance is central.")
    print("Standardization prevents large-unit features from dominating the distance.")


if __name__ == "__main__":
    main()
