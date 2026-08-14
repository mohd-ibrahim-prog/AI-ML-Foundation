"""
main.py

Week 3 - AI/ML Foundation

This script performs the complete Week 3 workflow:

1. Load the raw SMS spam/ham dataset
2. Clean missing values and duplicate rows
3. Preprocess the message text
4. Split the data into train/test sets
5. Build TF-IDF features and train a Naive Bayes classifier
6. Evaluate the classifier
7. Save the trained model, predictions, and evaluation report
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import (
    load_dataset,
    display_dataset_summary,
    clean_missing_and_duplicates,
)

from text_preprocessor import preprocess_dataset

from model_trainer import train_model

from model_evaluator import evaluate_model


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "sms_spam_dataset.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "cleaned_sms_spam_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "spam_classifier.pkl"
PREDICTIONS_FILE = BASE_DIR / "outputs" / "predictions.csv"
REPORT_FILE = BASE_DIR / "reports" / "classification_report.txt"


def save_outputs(model, X_test, y_test, predictions, metrics):
    """Save the trained model, test predictions, and evaluation report."""

    # Save the trained TF-IDF + Naive Bayes pipeline
    joblib.dump(model, MODEL_FILE)

    # Save predictions alongside the actual labels
    prediction_data = pd.DataFrame({
        "message": X_test.values,
        "actual_label": y_test.values,
        "predicted_label": predictions,
    })
    prediction_data.to_csv(PREDICTIONS_FILE, index=False)

    # Save a simple evaluation report
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("WEEK 3 - SPAM CLASSIFIER REPORT\n")
        report.write("=" * 45 + "\n\n")

        report.write("Model: TF-IDF + Multinomial Naive Bayes\n\n")

        report.write("Evaluation Metrics\n")
        report.write("-" * 20 + "\n")
        report.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
        report.write(f"Precision : {metrics['precision']:.4f}  (spam)\n")
        report.write(f"Recall    : {metrics['recall']:.4f}  (spam)\n")
        report.write(f"F1-score  : {metrics['f1_score']:.4f}  (spam)\n\n")

        conf_matrix = metrics["confusion_matrix"]
        report.write("Confusion Matrix (rows = actual, columns = predicted)\n")
        report.write("-" * 45 + "\n")
        report.write("             Pred: ham   Pred: spam\n")
        report.write(f"Actual: ham   {conf_matrix[0][0]:<10} {conf_matrix[0][1]}\n")
        report.write(f"Actual: spam  {conf_matrix[1][0]:<10} {conf_matrix[1][1]}\n\n")

        report.write("Classification Report\n")
        report.write("-" * 45 + "\n")
        report.write(metrics["report_text"])

    print("\nFiles generated successfully:")
    print(f"Model       : {MODEL_FILE}")
    print(f"Predictions : {PREDICTIONS_FILE}")
    print(f"Report      : {REPORT_FILE}")


def main():
    print("=" * 60)
    print("             AI/ML FOUNDATION - WEEK 3")
    print("       TEXT PREPROCESSING + TF-IDF + NAIVE BAYES")
    print("=" * 60)

    # Load raw dataset
    raw_dataset = load_dataset(RAW_DATA)
    display_dataset_summary(raw_dataset)

    # Handle missing values and duplicates
    cleaned_dataset = clean_missing_and_duplicates(raw_dataset)

    # Preprocess message text
    processed_dataset = preprocess_dataset(cleaned_dataset)

    # Save the cleaned/processed dataset
    processed_dataset.to_csv(PROCESSED_DATA, index=False)

    # Features and label
    X = processed_dataset["clean_message"]
    y = processed_dataset["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/Test Split")
    print("----------------")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    predictions, metrics = evaluate_model(model, X_test, y_test)

    # Save all important outputs
    save_outputs(model, X_test, y_test, predictions, metrics)

    print("\n" + "=" * 60)
    print("             WEEK 3 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
