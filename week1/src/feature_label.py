"""
feature_label.py

Purpose:
    Separate the dataset into Features (X) and Label (y).

Author:
    Mohd Ibrahim

Week:
    AI/ML Foundation - Week 1
"""

import pandas as pd


def split_features_and_label(
    dataset: pd.DataFrame,
    label_column: str = "Disease"
):
    """
    Split the dataset into Features (X) and Label (y).

    Args:
        dataset (pd.DataFrame): Cleaned dataset.
        label_column (str): Target column.

    Returns:
        tuple:
            X -> Features
            y -> Label
    """

    if label_column not in dataset.columns:
        raise ValueError(
            f"'{label_column}' column was not found in the dataset."
        )

    features = dataset.drop(columns=[label_column])

    label = dataset[label_column]

    return features, label


def display_feature_label_info(
    features: pd.DataFrame,
    label: pd.Series
) -> None:
    """
    Display information about Features and Label.
    """

    print("\n" + "=" * 60)
    print("FEATURES & LABEL")
    print("=" * 60)

    print("\nFeature Columns")
    print("-" * 60)

    for column in features.columns:
        print(f"• {column}")

    print("\nLabel Column")
    print("-" * 60)
    print(label.name)

    print("\nFeature Shape")
    print("-" * 60)
    print(features.shape)

    print("\nLabel Shape")
    print("-" * 60)
    print(label.shape)