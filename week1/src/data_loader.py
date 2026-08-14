"""
data_loader.py

Purpose:
    Load the dataset from a CSV file and display basic information.

Author:
    Mohd Ibrahim

Week:
    AI/ML Foundation - Week 1
"""

from pathlib import Path
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset into a Pandas DataFrame.

    Args:
        file_path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.

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
    Display a quick summary of the dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows    : {dataset.shape[0]}")
    print(f"Columns : {dataset.shape[1]}")

    print("\nColumn Names")
    print("-" * 60)
    print(dataset.columns.tolist())

    print("\nData Types")
    print("-" * 60)
    print(dataset.dtypes)

    print("\nFirst Five Records")
    print("-" * 60)
    print(dataset.head())


def display_missing_values(dataset: pd.DataFrame) -> None:
    """
    Display missing values for every column.
    """

    print("\nMissing Values")
    print("-" * 60)
    print(dataset.isnull().sum())