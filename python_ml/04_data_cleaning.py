"""Day 4: clean and prepare a small tabular ML dataset."""

import pandas as pd

DATA_PATH = "data/day4_cleaning_practice.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("Original shape:", df.shape)
    print("\nMissing values before cleaning:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # Remove exact duplicate observations.
    df = df.drop_duplicates().copy()

    # Fill numeric missing values with the column median.
    numeric_columns = ["study_hours", "assignments_completed"]
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill a missing categorical value with the most frequent category.
    df["study_mode"] = df["study_mode"].fillna(df["study_mode"].mode()[0])

    # One-hot encode the categorical feature for model-ready numeric data.
    df = pd.get_dummies(df, columns=["study_mode"], dtype=int)

    # Separate features from the target.
    X = df.drop(columns=["student_id", "score"])
    y = df["score"]

    print("\nShape after cleaning:", df.shape)
    print("\nMissing values after cleaning:\n", df.isna().sum())
    print("\nCleaned data:\n", df)
    print("\nFeature columns:", list(X.columns))
    print("Target column: score")
    print("\nTarget mean:", round(y.mean(), 2))


if __name__ == "__main__":
    main()
