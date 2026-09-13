"""Day 17: Support Vector Machines (SVM) and margins.

Practical experiment: compare a linear SVM across regularization strengths and
observe how feature scaling supports a distance/margin-based model.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    print("Linear SVM with standardization")
    for C in (0.01, 0.1, 1, 10, 100):
        model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(f"C={C:6g}: test accuracy={accuracy_score(y_test, predictions):.3f}")

    print("\nKey ideas:")
    print("- SVM chooses a separating boundary with a large margin.")
    print("- Support vectors are the training points closest to the boundary.")
    print("- C controls the trade-off between a wider margin and training errors.")
    print("- Scaling is important because SVM optimization is sensitive to feature scale.")


if __name__ == "__main__":
    main()
