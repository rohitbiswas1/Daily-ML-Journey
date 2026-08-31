"""Day 5: basic statistics for machine learning."""

import numpy as np
import pandas as pd

DATA_PATH = "data/study_progress.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    score = df["score"].to_numpy(dtype=float)

    print("Scores:", score)
    print("Mean:", round(np.mean(score), 2))
    print("Median:", round(np.median(score), 2))
    print("Variance:", round(np.var(score), 2))
    print("Standard deviation:", round(np.std(score), 2))
    print("Minimum:", np.min(score))
    print("Maximum:", np.max(score))
    print("Range:", np.ptp(score))

    # Compare a symmetric and skewed example to build intuition.
    symmetric = np.array([10, 20, 30, 40, 50])
    skewed = np.array([10, 20, 20, 20, 100])
    print("\nSymmetric example -> mean:", np.mean(symmetric), "median:", np.median(symmetric))
    print("Skewed example -> mean:", np.mean(skewed), "median:", np.median(skewed))


if __name__ == "__main__":
    main()
