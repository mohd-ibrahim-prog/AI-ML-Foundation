"""
main.py

Week 7 - AI/ML Foundation
Per-Class Accuracy - Evaluate Performance for Each Disease/Class

This script performs the complete Week 7 workflow:

1. Locate the PlantVillage dataset (reused from Week 4)
2. Rebuild the exact same held-out test split used in Week 6
3. Load the trained model produced by Week 6
4. Run predictions on the test set
5. Compute per-class accuracy, precision, recall and F1-score
6. Save a confusion matrix, a classification report and a
   per-class metrics CSV
7. Identify the strongest and weakest performing classes

Note:
    This week does NOT train a new model. It reuses the model
    that Week 6 already trained and evaluated, and focuses purely
    on breaking that evaluation down class-by-class.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from tensorflow.keras.models import load_model

from dataset_utils import (
    resolve_dataset_dir,
    discover_images_and_labels,
    print_dataset_summary,
)
from data_loader import create_test_dataset
from evaluate_per_class import (
    collect_predictions,
    compute_per_class_metrics,
    save_per_class_csv,
    save_classification_report,
    save_confusion_matrix_plot,
    summarize_strongest_and_weakest,
)


WEEK7_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = WEEK7_DIR.parent

WEEK6_MODEL = REPO_ROOT / "week6" / "models" / "plant_disease_cnn_week6.keras"

PER_CLASS_CSV = WEEK7_DIR / "reports" / "per_class_metrics.csv"
CLASSIFICATION_REPORT_TXT = WEEK7_DIR / "reports" / "classification_report.txt"
CONFUSION_MATRIX_PLOT = WEEK7_DIR / "outputs" / "confusion_matrix.png"
SUMMARY_REPORT = WEEK7_DIR / "reports" / "week7_summary.txt"

VAL_SIZE = 0.15
TEST_SIZE = 0.15


def load_week6_model():
    """
    Load the model trained in Week 6.

    Raises a clear error if it hasn't been generated yet, instead
    of silently failing or fabricating results.
    """

    if not WEEK6_MODEL.exists():
        raise FileNotFoundError(
            "Could not find the Week 6 trained model.\n\n"
            f"Expected it at: {WEEK6_MODEL}\n\n"
            "Please run Week 6 first:\n"
            "  cd week6/src\n"
            "  python main.py\n\n"
            "This will train the model and save it to that location."
        )

    print(f"\nLoading trained model from Week 6:")
    print(f"  {WEEK6_MODEL}")

    return load_model(WEEK6_MODEL)


def save_summary(class_names, per_class_df, strongest, weakest):

    with open(SUMMARY_REPORT, "w", encoding="utf-8") as f:
        f.write("WEEK 7 - PER-CLASS EVALUATION SUMMARY\n")
        f.write("=" * 55 + "\n\n")

        f.write("Task: Per-class accuracy - evaluate each disease/class\n")
        f.write(f"Model evaluated: {WEEK6_MODEL.name} (trained in Week 6)\n\n")

        f.write(f"Total classes evaluated: {len(class_names)}\n\n")

        f.write("Per-class results (accuracy = recall for that class):\n")
        f.write("-" * 55 + "\n")
        for _, row in per_class_df.iterrows():
            f.write(
                f"{row['class']:<35} "
                f"accuracy={row['per_class_accuracy']:.4f}  "
                f"precision={row['precision']:.4f}  "
                f"recall={row['recall']:.4f}  "
                f"f1={row['f1_score']:.4f}  "
                f"support={row['support']}\n"
            )

        f.write("\nStrongest class:\n")
        f.write(
            f"  {strongest['class']}  (F1-score: {strongest['f1_score']:.4f}, "
            f"per-class accuracy: {strongest['per_class_accuracy']:.4f})\n"
        )

        f.write("\nWeakest class:\n")
        f.write(
            f"  {weakest['class']}  (F1-score: {weakest['f1_score']:.4f}, "
            f"per-class accuracy: {weakest['per_class_accuracy']:.4f})\n"
        )

    print(f"\nSummary report saved to:")
    print(f"  {SUMMARY_REPORT}")


def main():

    print("\n" + "=" * 70)
    print("        AI/ML FOUNDATION - WEEK 7")
    print("        PER-CLASS ACCURACY EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. LOAD DATASET
    # -----------------------------------------------------

    dataset_dir = resolve_dataset_dir()
    image_files, labels, class_names = discover_images_and_labels(dataset_dir)
    print_dataset_summary(dataset_dir, image_files, labels, class_names)

    # -----------------------------------------------------
    # 2. REBUILD THE SAME TEST SPLIT AS WEEK 6
    # -----------------------------------------------------

    test_dataset, test_files, test_labels = create_test_dataset(
        image_files, labels, class_names,
        val_size=VAL_SIZE, test_size=TEST_SIZE,
    )
    print(f"\nTest images for per-class evaluation: {len(test_files)}")

    # -----------------------------------------------------
    # 3. LOAD THE WEEK 6 TRAINED MODEL
    # -----------------------------------------------------

    model = load_week6_model()

    # -----------------------------------------------------
    # 4. RUN PREDICTIONS
    # -----------------------------------------------------

    print("\nRunning predictions on the test set...")
    y_true, y_pred = collect_predictions(model, test_dataset)

    # -----------------------------------------------------
    # 5. COMPUTE PER-CLASS METRICS
    # -----------------------------------------------------

    per_class_df, conf_matrix = compute_per_class_metrics(y_true, y_pred, class_names)

    print("\nPer-class metrics:")
    print(per_class_df.to_string(index=False))

    # -----------------------------------------------------
    # 6. SAVE REPORTS
    # -----------------------------------------------------

    save_per_class_csv(per_class_df, PER_CLASS_CSV)
    save_classification_report(y_true, y_pred, class_names, CLASSIFICATION_REPORT_TXT)
    save_confusion_matrix_plot(conf_matrix, class_names, CONFUSION_MATRIX_PLOT)

    # -----------------------------------------------------
    # 7. STRONGEST / WEAKEST CLASSES
    # -----------------------------------------------------

    strongest, weakest = summarize_strongest_and_weakest(per_class_df)
    save_summary(class_names, per_class_df, strongest, weakest)

    print("\n" + "=" * 70)
    print("        WEEK 7 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
