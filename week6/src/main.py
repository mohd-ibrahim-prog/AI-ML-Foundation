"""
main.py

Week 6 - AI/ML Foundation
Model Evaluation & Improvement

This script continues from Week 5 and performs the complete
Week 6 workflow:

1. Locate and inspect the PlantVillage dataset (reused from Week 4)
2. Discover disease classes
3. Build a stratified train/validation/TEST split
4. Build an improved CNN (data augmentation + extra dropout)
5. Train with early stopping and model checkpointing
6. Properly evaluate on the held-out test set:
   accuracy, loss, confusion matrix, classification report
7. Save the trained model, evaluation report and confusion matrix
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
from model import build_improved_cnn
from train import train_model
from evaluate import evaluate_model, save_confusion_matrix_plot


WEEK6_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_FILE = WEEK6_DIR / "models" / "plant_disease_cnn_week6.keras"
CONFUSION_MATRIX_PLOT = WEEK6_DIR / "outputs" / "confusion_matrix.png"
REPORT_FILE = WEEK6_DIR / "reports" / "week6_report.txt"

MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def save_report(class_names, results, epochs_ran):

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("WEEK 6 - MODEL EVALUATION & IMPROVEMENT REPORT\n")
        report.write("=" * 55 + "\n\n")

        report.write("Task: Evaluate/improve the Week 5 image classifier\n")
        report.write(
            "Improvements applied: data augmentation, extra dropout, "
            "early stopping, model checkpointing\n\n"
        )

        report.write(f"Disease classes ({len(class_names)}):\n")
        for name in class_names:
            report.write(f"  - {name}\n")

        report.write(f"\nEpochs actually run : {epochs_ran}\n")
        report.write(f"Test Loss            : {results['test_loss']:.4f}\n")
        report.write(f"Test Accuracy        : {results['test_accuracy']:.4f}\n\n")

        report.write("Classification Report (test set)\n")
        report.write("-" * 55 + "\n")
        report.write(results["classification_report_text"])

    print(f"\nReport saved to:")
    print(f"  {REPORT_FILE}")


def main():

    print("\n" + "=" * 70)
    print("        AI/ML FOUNDATION - WEEK 6")
    print("        MODEL EVALUATION & IMPROVEMENT")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. LOAD DATASET
    # -----------------------------------------------------

    dataset_dir = resolve_dataset_dir()
    image_files, labels, class_names = discover_images_and_labels(dataset_dir)
    print_dataset_summary(dataset_dir, image_files, labels, class_names)

    # -----------------------------------------------------
    # 2. TRAIN / VALIDATION / TEST SPLIT + DATASETS
    # -----------------------------------------------------

    train_dataset, val_dataset, test_dataset = create_datasets(
        image_files, labels, class_names,
        val_size=VAL_SIZE, test_size=TEST_SIZE,
    )

    # -----------------------------------------------------
    # 3. BUILD IMPROVED MODEL
    # -----------------------------------------------------

    print("\nBuilding improved CNN model...")
    model = build_improved_cnn(num_classes=len(class_names))
    model.summary()

    # -----------------------------------------------------
    # 4. TRAIN WITH EARLY STOPPING + CHECKPOINTING
    # -----------------------------------------------------

    history = train_model(
        model, train_dataset, val_dataset,
        checkpoint_path=CHECKPOINT_FILE,
        epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
    )

    epochs_ran = len(history.history["loss"])

    # -----------------------------------------------------
    # 5. EVALUATE ON TEST SET
    # -----------------------------------------------------

    results = evaluate_model(model, test_dataset, class_names)

    save_confusion_matrix_plot(
        results["confusion_matrix"], class_names, CONFUSION_MATRIX_PLOT
    )

    # -----------------------------------------------------
    # 6. SAVE REPORT
    # (the best model was already saved by ModelCheckpoint)
    # -----------------------------------------------------

    print(f"\nBest model checkpoint saved to:")
    print(f"  {CHECKPOINT_FILE}")

    save_report(class_names, results, epochs_ran)

    print("\n" + "=" * 70)
    print("        WEEK 6 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
