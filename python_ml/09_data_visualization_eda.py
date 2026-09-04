"""Day 9: basic exploratory data analysis."""

import pandas as pd

DATA_PATH = "data/study_progress.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSummary statistics:")
    print(df.describe(numeric_only=True).round(2))
    print("\nStudy mode counts:")
    print(df["study_mode"].value_counts())
    print("\nCorrelation matrix:")
    print(df.select_dtypes(include="number").corr().round(2))


if __name__ == "__main__":
    main()
