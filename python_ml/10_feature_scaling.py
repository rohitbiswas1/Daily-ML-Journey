"""Day 10: compare normalization and standardization for ML features."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main() -> None:
    # Same observations, but features have very different numeric scales.
    X = np.array([
        [2, 1200],
        [4, 1500],
        [6, 1800],
        [8, 2100],
        [10, 2400],
    ], dtype=float)

    minmax = MinMaxScaler()
    standard = StandardScaler()

    X_minmax = minmax.fit_transform(X)
    X_standard = standard.fit_transform(X)

    np.set_printoptions(precision=3, suppress=True)
    print("Original features:\n", X)
    print("\nMin-Max normalized features (0 to 1):\n", X_minmax)
    print("\nStandardized features (mean≈0, std≈1):\n", X_standard)
    print("\nOriginal means:", X.mean(axis=0))
    print("Original std devs:", X.std(axis=0))
    print("Standardized means:", X_standard.mean(axis=0))
    print("Standardized std devs:", X_standard.std(axis=0))

    print("\nKey idea: scaling changes feature units, not the underlying order of observations.")
    print("Fit scalers on training data only in a real ML workflow to avoid data leakage.")


if __name__ == "__main__":
    main()
