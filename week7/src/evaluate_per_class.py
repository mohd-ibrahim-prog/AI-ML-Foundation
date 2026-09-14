"""
evaluate_per_class.py

Purpose:
    Evaluate the trained model separately for EVERY disease/class:
    per-class accuracy (recall), precision, recall, F1-score, and
    identify the strongest and weakest classes.

    All metrics are computed from real predictions made on the
    held-out test set - nothing here is invented or assumed.

Week:
    AI/ML Foundation - Week 7
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def collect_predictions(model, test_dataset):
    """Run the model on the test dataset and collect true/predicted labels."""

    true_labels = []
    predicted_labels = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_labels.extend(np.argmax(predictions, axis=1))
        true_labels.extend(labels.numpy())

    return np.array(true_labels), np.array(predicted_labels)


def compute_per_class_metrics(y_true, y_pred, class_names):
    """
    Compute per-class accuracy (recall), precision, recall, F1
    and support for every class, from real predictions.

    "Per-class accuracy" is defined here as the proportion of a
    class's own test images that were correctly classified
    (i.e. the diagonal of the confusion matrix divided by the
    row total). This is numerically the same as recall.

    Returns:
        per_class_df (pandas.DataFrame), conf_matrix (np.ndarray)
    """

    labels_range = list(range(len(class_names)))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels_range)

    report_dict = classification_report(
        y_true, y_pred,
        labels=labels_range,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for index, class_name in enumerate(class_names):
        row_total = conf_matrix[index].sum()
        correct = conf_matrix[index, index]
        per_class_accuracy = correct / row_total if row_total > 0 else 0.0

        metrics = report_dict[class_name]

        rows.append({
            "class": class_name,
            "support": int(row_total),
            "correct_predictions": int(correct),
            "per_class_accuracy": per_class_accuracy,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
        })

    per_class_df = pd.DataFrame(rows)

    return per_class_df, conf_matrix


def save_per_class_csv(per_class_df, output_path):
    per_class_df.to_csv(output_path, index=False)
    print(f"\nPer-class metrics saved to:")
    print(f"  {output_path}")


def save_classification_report(y_true, y_pred, class_names, output_path):
    labels_range = list(range(len(class_names)))
    report_text = classification_report(
        y_true, y_pred,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEEK 7 - PER-CLASS CLASSIFICATION REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(report_text)

    print(f"Classification report saved to:")
    print(f"  {output_path}")

    return report_text


def save_confusion_matrix_plot(conf_matrix, class_names, output_path):

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6),
                                     max(5, len(class_names) * 0.5)))

    im = ax.imshow(conf_matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Per-Class Evaluation (Test Set)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Confusion matrix plot saved to:")
    print(f"  {output_path}")


def summarize_strongest_and_weakest(per_class_df):
    """Identify the best and worst performing classes by F1-score."""

    sorted_df = per_class_df.sort_values("f1_score", ascending=False)

    strongest = sorted_df.iloc[0]
    weakest = sorted_df.iloc[-1]

    print("\nStrongest class:")
    print(f"  {strongest['class']}  (F1-score: {strongest['f1_score']:.4f}, "
          f"per-class accuracy: {strongest['per_class_accuracy']:.4f})")

    print("Weakest class:")
    print(f"  {weakest['class']}  (F1-score: {weakest['f1_score']:.4f}, "
          f"per-class accuracy: {weakest['per_class_accuracy']:.4f})")

    return strongest, weakest
