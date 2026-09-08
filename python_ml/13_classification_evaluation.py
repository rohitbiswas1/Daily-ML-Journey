"""Day 13: Precision, recall, F1, confusion matrix, and thresholds."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=600, n_features=5, n_informative=3,
                           n_redundant=0, weights=[0.65, 0.35], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]

for threshold in (0.30, 0.50, 0.70):
    pred = (prob >= threshold).astype(int)
    print(f"\nThreshold: {threshold:.2f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 3))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 3))
    print("F1 score :", round(f1_score(y_test, pred, zero_division=0), 3))

print("\nLower thresholds generally favor recall; higher thresholds generally favor precision.")
