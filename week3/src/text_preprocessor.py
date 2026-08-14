"""
text_preprocessor.py

Purpose:
    Clean and normalize raw SMS text so it is ready to be converted
    into TF-IDF features.

Week:
    AI/ML Foundation - Week 3
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Normalize a single message.

    Steps:
        1. Lowercase the text.
        2. Remove URLs.
        3. Remove punctuation and digits (keep only letters and spaces).
        4. Collapse multiple spaces into a single space and strip.

    Args:
        text (str): Raw message text.

    Returns:
        str: Cleaned message text.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to every message in the dataset and drop any
    rows that become empty after cleaning.

    Args:
        dataset (pd.DataFrame): Dataset with a 'message' column.

    Returns:
        pd.DataFrame: Dataset with an added 'clean_message' column.
    """

    processed = dataset.copy()
    processed["clean_message"] = processed["message"].apply(clean_text)

    empty_after_cleaning = (processed["clean_message"] == "").sum()
    processed = processed[processed["clean_message"] != ""]
    processed = processed.reset_index(drop=True)

    print("\nText Preprocessing")
    print("-" * 60)
    print(f"Rows empty after cleaning (removed) : {empty_after_cleaning}")
    print(f"Rows remaining                      : {len(processed)}")

    print("\nSample Cleaned Messages")
    print("-" * 60)
    print(processed[["message", "clean_message"]].head())

    return processed
