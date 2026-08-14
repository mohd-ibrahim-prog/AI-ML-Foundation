"""
report_generator.py

Purpose:
    Generate a simple dataset cleaning report.

Author:
    Mohd Ibrahim

Week:
    AI/ML Foundation - Week 1
"""

from pathlib import Path
import pandas as pd


def generate_cleaning_report(
    original_dataset: pd.DataFrame,
    cleaned_dataset: pd.DataFrame,
    features: pd.DataFrame,
    label: pd.Series,
    report_path: str
) -> None:
    """
    Generate a text report summarizing the cleaning process.
    """

    rows_before = len(original_dataset)
    rows_after = len(cleaned_dataset)

    duplicate_rows_removed = rows_before - len(original_dataset.drop_duplicates())

    missing_values_before = int(original_dataset.isnull().sum().sum())
    missing_values_after = int(cleaned_dataset.isnull().sum().sum())

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as report:

        report.write("=" * 60 + "\n")
        report.write("        WEEK 1 DATA CLEANING REPORT\n")
        report.write("=" * 60 + "\n\n")

        report.write(f"Rows Before Cleaning : {rows_before}\n")
        report.write(f"Rows After Cleaning  : {rows_after}\n\n")

        report.write(f"Missing Values Before : {missing_values_before}\n")
        report.write(f"Missing Values After  : {missing_values_after}\n\n")

        report.write(f"Duplicate Rows Removed : {duplicate_rows_removed}\n\n")

        report.write("Feature Columns\n")
        report.write("-" * 60 + "\n")

        for column in features.columns:
            report.write(f"- {column}\n")

        report.write("\n")

        report.write(f"Label Column : {label.name}\n")

        report.write("\n")
        report.write("=" * 60 + "\n")
        report.write("Dataset cleaning completed successfully.\n")
        report.write("=" * 60 + "\n")

    print(f"\nCleaning report saved to:\n{report_file}")