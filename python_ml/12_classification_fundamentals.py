"""Day 12: Classification Fundamentals
A beginner-friendly binary classification experiment using logistic regression.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Small synthetic dataset: study hours and practice tests -> pass/fail.
X = np.array([
    [1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 2],
    [3, 3], [3, 4], [4, 3], [4, 4], [4, 5], [5, 4],
    [5, 5], [6, 5], [6, 6], [7, 6], [7, 7], [8, 7],
    [8, 8], [9, 8]
], dtype=float)

y = np.array([
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Test accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print("Classification report:\n", classification_report(y_test, predictions, zero_division=0))

# Predict the probability of passing for a student studying 5 hours
# and completing 5 practice tests.
new_student = np.array([[5, 5]], dtype=float)
print("P(pass | 5 study hours, 5 practice tests):", model.predict_proba(new_student)[0, 1])
print("Predicted class:", model.predict(new_student)[0])

# Key lesson: classification predicts a discrete class, while
# predict_proba() exposes the model's confidence for each class.
