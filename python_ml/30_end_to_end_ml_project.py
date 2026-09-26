"""Day 30: End-to-end ML project."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, stratify=data.target, random_state=RANDOM_STATE)
model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
print("CV ROC-AUC:", np.round(cv_scores, 3))
print("CV mean ROC-AUC:", round(cv_scores.mean(), 3))
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Test ROC-AUC:", round(roc_auc_score(y_test, probabilities), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=data.target_names))
joblib.dump(model, ARTIFACT_DIR / "day30_breast_cancer_pipeline.joblib")
print("Saved model artifact.")