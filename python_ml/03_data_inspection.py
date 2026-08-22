"""Day 3: inspect a small tabular dataset with Pandas and NumPy."""

import numpy as np
import pandas as pd

DATA_PATH = "data/study_progress.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nSummary statistics:\n", df.describe().round(2))

    # NumPy example: calculate the mean directly from a numeric column.
    scores = df["score"].to_numpy(dtype=float)
    print("\nMean score (NumPy):", np.mean(scores))

    # Correlation is for exploration, not proof of causation.
    correlations = df.corr(numeric_only=True)["score"].drop("score")
    print("\nCorrelation with score:\n", correlations.round(3))
    print("\nMost correlated feature:", correlations.abs().idxmax())


if __name__ == "__main__":
    main()
