"""
data_cleaner.py

Purpose:
    Clean the dataset by handling missing values,
    removing duplicates, and standardizing text columns.

Author:
    Mohd Ibrahim

Week:
    AI/ML Foundation - Week 1
"""

from pathlib import Path
import pandas as pd


def fill_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the dataset.

    - Numeric columns -> Mean
    - Text columns -> Mode
    """

    cleaned_dataset = dataset.copy()

    for column in cleaned_dataset.columns:

        if cleaned_dataset[column].dtype == "object":

            if cleaned_dataset[column].isnull().any():
                mode_value = cleaned_dataset[column].mode()[0]
                cleaned_dataset[column] = cleaned_dataset[column].fillna(mode_value)

        else:

            if cleaned_dataset[column].isnull().any():
                mean_value = cleaned_dataset[column].mean()
                cleaned_dataset[column] = cleaned_dataset[column].fillna(mean_value)

    return cleaned_dataset


def remove_duplicate_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """

    return dataset.drop_duplicates().reset_index(drop=True)


def standardize_text_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extra spaces and standardize text formatting.
    """

    cleaned_dataset = dataset.copy()

    text_columns = cleaned_dataset.select_dtypes(include="object").columns

    for column in text_columns:

        cleaned_dataset[column] = (
            cleaned_dataset[column]
            .astype(str)
            .str.strip()
            .str.title()
        )

    return cleaned_dataset


def save_clean_dataset(dataset: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned dataset to CSV.
    """

    output_file = Path(output_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(output_file, index=False)

    print(f"\nClean dataset saved to:\n{output_file}")