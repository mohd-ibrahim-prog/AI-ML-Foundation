"""
main.py

Week 1 - AI/ML Foundation

This script performs the complete Week 1 workflow:

1. Load the dataset
2. Display dataset information
3. Clean the dataset
4. Save cleaned dataset
5. Separate Features and Label
6. Generate cleaning report
"""

from pathlib import Path

from data_loader import (
    load_dataset,
    display_dataset_summary,
    display_missing_values,
)

from data_cleaner import (
    fill_missing_values,
    remove_duplicate_rows,
    standardize_text_columns,
    save_clean_dataset,
)

from feature_label import (
    split_features_and_label,
    display_feature_label_info,
)

from report_generator import (
    generate_cleaning_report,
)


def main():

    print("=" * 60)
    print("      AI/ML FOUNDATION - WEEK 1")
    print("     DATA LOADING & CLEANING")
    print("=" * 60)

    # Locate the Week 1 project directory
    week1_dir = Path(__file__).resolve().parent.parent

    raw_dataset_path = (
        week1_dir
        / "data"
        / "raw"
        / "plant_health_dataset.csv"
    )

    cleaned_dataset_path = (
        week1_dir
        / "data"
        / "processed"
        / "cleaned_plant_health_dataset.csv"
    )

    report_path = (
        week1_dir
        / "reports"
        / "cleaning_report.txt"
    )

    # Load dataset
    original_dataset = load_dataset(raw_dataset_path)

    display_dataset_summary(original_dataset)

    display_missing_values(original_dataset)

    # Clean dataset
    cleaned_dataset = fill_missing_values(original_dataset)

    cleaned_dataset = remove_duplicate_rows(cleaned_dataset)

    cleaned_dataset = standardize_text_columns(cleaned_dataset)

    # Save cleaned dataset
    save_clean_dataset(
        cleaned_dataset,
        cleaned_dataset_path,
    )

    # Features & Label
    features, label = split_features_and_label(
        cleaned_dataset
    )

    display_feature_label_info(
        features,
        label,
    )

    # Generate report
    generate_cleaning_report(
        original_dataset,
        cleaned_dataset,
        features,
        label,
        report_path,
    )

    print("\n" + "=" * 60)
    print("Week 1 completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()