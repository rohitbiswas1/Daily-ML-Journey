"""Day 21: Principal Component Analysis (PCA).

Practical goals:
- standardize correlated features before PCA
- reduce 3D data to 2D
- inspect explained variance
- compare a classifier before and after PCA
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train_scaled, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test_scaled))

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    reduced = LogisticRegression(max_iter=1000, random_state=42)
    reduced.fit(X_train_pca, y_train)
    reduced_acc = accuracy_score(y_test, reduced.predict(X_test_pca))

    print("Day 21 - PCA")
    print(f"Original features: {X.shape[1]}")
    print(f"Reduced features: {X_train_pca.shape[1]}")
    print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Baseline Logistic Regression accuracy: {baseline_acc:.3f}")
    print(f"PCA + Logistic Regression accuracy: {reduced_acc:.3f}")
    print("First principal component loadings:", np.round(pca.components_[0], 3))


if __name__ == "__main__":
    main()
