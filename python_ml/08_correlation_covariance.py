"""Day 8: correlation and covariance for machine learning."""

import numpy as np
import pandas as pd

DATA_PATH = "data/study_progress.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Select numeric variables whose relationships are useful to inspect.
    features = df[["study_hours", "assignments_completed", "score"]]

    print("Correlation matrix (Pearson):")
    print(features.corr().round(3))

    print("\nCovariance matrix:")
    print(features.cov().round(3))

    study_hours = df["study_hours"].to_numpy(dtype=float)
    score = df["score"].to_numpy(dtype=float)

    correlation = np.corrcoef(study_hours, score)[0, 1]
    covariance = np.cov(study_hours, score, ddof=1)[0, 1]

    print(f"\nStudy hours vs score correlation: {correlation:.3f}")
    print(f"Study hours vs score covariance: {covariance:.3f}")

    print("\nML intuition:")
    print("- Positive correlation means two variables tend to increase together.")
    print("- Negative correlation means one tends to increase as the other decreases.")
    print("- Correlation near zero means little linear relationship, not necessarily no relationship.")
    print("- Correlation does not prove that one variable causes the other.")


if __name__ == "__main__":
    main()
