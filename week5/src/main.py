"""
main.py

Week 5 - AI/ML Foundation
Training & Validation for Images

This script performs the complete Week 5 workflow:

1. Locate and inspect the PlantVillage dataset (reused from Week 4)
2. Discover disease classes
3. Build a stratified train/validation split
4. Build the CNN (same architecture family as Week 4)
5. Train the model, validating after every epoch
6. Evaluate on the validation set
7. Save the trained model, training history and a short report
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from dataset_utils import (
    resolve_dataset_dir,
    discover_images_and_labels,
    print_dataset_summary,
)
from data_loader import create_datasets
from model import build_cnn
from train import train_model
from evaluate import evaluate_on_validation, save_training_history


WEEK5_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = WEEK5_DIR / "models" / "plant_disease_cnn_week5.keras"
HISTORY_CSV = WEEK5_DIR / "outputs" / "training_history.csv"
HISTORY_PLOT = WEEK5_DIR / "outputs" / "training_history.png"
REPORT_FILE = WEEK5_DIR / "reports" / "week5_report.txt"

EPOCHS = 5
VAL_SIZE = 0.2


def save_report(class_names, val_loss, val_accuracy, epochs):
    """Save a short, honest summary of what happened during this run."""

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("WEEK 5 - IMAGE CLASSIFICATION TRAINING REPORT\n")
        report.write("=" * 50 + "\n\n")

        report.write("Task: Training / validation for images\n")
        report.write("Model: CNN (3 conv blocks + dense classifier)\n\n")

        report.write(f"Disease classes ({len(class_names)}):\n")
        for name in class_names:
            report.write(f"  - {name}\n")

        report.write(f"\nEpochs trained     : {epochs}\n")
        report.write(f"Validation Loss    : {val_loss:.4f}\n")
        report.write(f"Validation Accuracy: {val_accuracy:.4f}\n")

    print(f"\nReport saved to:")
    print(f"  {REPORT_FILE}")


def main():

    print("\n" + "=" * 70)
    print("        AI/ML FOUNDATION - WEEK 5")
    print("        TRAINING & VALIDATION FOR IMAGES")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. LOAD DATASET
    # -----------------------------------------------------

    dataset_dir = resolve_dataset_dir()
    image_files, labels, class_names = discover_images_and_labels(dataset_dir)
    print_dataset_summary(dataset_dir, image_files, labels, class_names)

    # -----------------------------------------------------
    # 2. TRAIN / VALIDATION SPLIT + DATASETS
    # -----------------------------------------------------

    train_dataset, val_dataset = create_datasets(
        image_files, labels, class_names, val_size=VAL_SIZE
    )

    # -----------------------------------------------------
    # 3. BUILD MODEL
    # -----------------------------------------------------

    print("\nBuilding CNN model...")
    model = build_cnn(num_classes=len(class_names))
    model.summary()

    # -----------------------------------------------------
    # 4. TRAIN WITH VALIDATION
    # -----------------------------------------------------

    history = train_model(model, train_dataset, val_dataset, epochs=EPOCHS)

    # -----------------------------------------------------
    # 5. EVALUATE ON VALIDATION SET
    # -----------------------------------------------------

    val_loss, val_accuracy = evaluate_on_validation(model, val_dataset)

    # -----------------------------------------------------
    # 6. SAVE MODEL, HISTORY AND REPORT
    # -----------------------------------------------------

    model.save(MODEL_FILE)
    print(f"\nModel saved to:")
    print(f"  {MODEL_FILE}")

    save_training_history(history, HISTORY_CSV, HISTORY_PLOT)
    save_report(class_names, val_loss, val_accuracy, EPOCHS)

    print("\n" + "=" * 70)
    print("        WEEK 5 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
