"""Day 18: Naive Bayes and probabilistic classification.

Practical experiment: train Gaussian Naive Bayes on the breast-cancer dataset,
inspect predicted class probabilities, and compare it with logistic regression.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    print(f"{name}: accuracy={accuracy_score(y_test, predictions):.3f}, "
          f"log_loss={log_loss(y_test, probabilities):.3f}")
    return model


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    naive_bayes = evaluate("Gaussian Naive Bayes", GaussianNB(), X_train, X_test, y_train, y_test)
    logistic = evaluate(
        "Logistic Regression",
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        X_train, X_test, y_train, y_test,
    )

    probabilities = naive_bayes.predict_proba(X_test[:5])
    print("\nFirst five Naive Bayes probability predictions:")
    for index, row in enumerate(probabilities, start=1):
        print(f"Sample {index}: class 0={row[0]:.3f}, class 1={row[1]:.3f}")

    print("\nKey ideas:")
    print("- Naive Bayes uses Bayes' theorem to estimate class probabilities.")
    print("- GaussianNB assumes each feature is conditionally independent given the class.")
    print("- Probability outputs can communicate model confidence, not just the predicted label.")
    print("- Log loss evaluates the quality of predicted probabilities, not only accuracy.")


if __name__ == "__main__":
    main()
