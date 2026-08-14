"""
data_loader.py

Purpose:
    Load the SMS spam/ham dataset from a CSV file and remove
    obviously bad rows (missing messages, missing labels, duplicates)
    before any further preprocessing happens.

Week:
    AI/ML Foundation - Week 3
"""

from pathlib import Path
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the raw spam/ham dataset into a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Dataset with 'message' and 'label' columns.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty.
    """

    dataset_path = Path(file_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path.resolve()}"
        )

    dataset = pd.read_csv(dataset_path)

    if dataset.empty:
        raise ValueError("The dataset is empty.")

    return dataset


def display_dataset_summary(dataset: pd.DataFrame) -> None:
    """
    Display a quick summary of the raw dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows    : {dataset.shape[0]}")
    print(f"Columns : {dataset.shape[1]}")

    print("\nLabel Counts")
    print("-" * 60)
    print(dataset["label"].value_counts(dropna=False))

    print("\nFirst Five Records")
    print("-" * 60)
    print(dataset.head())


def clean_missing_and_duplicates(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and duplicate rows in the raw dataset.

    - Rows with a missing/empty message are dropped (nothing to learn from).
    - Rows with a missing/empty label are dropped (label is required).
    - Rows with a label outside {ham, spam} are dropped.
    - Exact duplicate rows are removed.

    Args:
        dataset (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Dataset ready for text preprocessing.
    """

    cleaned = dataset.copy()

    # Treat blank strings the same as missing values
    cleaned["message"] = cleaned["message"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()

    cleaned.loc[cleaned["message"].isin(["", "nan", "none"]), "message"] = pd.NA
    cleaned.loc[cleaned["label"].isin(["", "nan", "none"]), "label"] = pd.NA

    missing_before = cleaned["message"].isna().sum() + cleaned["label"].isna().sum()

    cleaned = cleaned.dropna(subset=["message", "label"])

    # Keep only valid ham/spam labels
    cleaned = cleaned[cleaned["label"].isin(["ham", "spam"])]

    duplicates_found = cleaned.duplicated(subset=["message", "label"]).sum()
    cleaned = cleaned.drop_duplicates(subset=["message", "label"])

    cleaned = cleaned.reset_index(drop=True)

    print("\nMissing / Duplicate Handling")
    print("-" * 60)
    print(f"Rows with missing message/label removed : {missing_before}")
    print(f"Duplicate rows removed                  : {duplicates_found}")
    print(f"Rows remaining                          : {len(cleaned)}")

    return cleaned
