"""
evaluate.py

Purpose:
    Properly evaluate the trained model on the held-out TEST
    dataset (data the model never saw during training or
    validation): accuracy, loss, confusion matrix, precision,
    recall and F1-score.

Week:
    AI/ML Foundation - Week 6
"""

import numpy as np
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


def evaluate_model(model, test_dataset, class_names):
    """
    Evaluate the model on the test dataset.

    Returns:
        dict with keys: test_loss, test_accuracy, confusion_matrix,
                         classification_report_text, y_true, y_pred
    """

    print("\nEvaluating on the TEST dataset...")

    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    y_true, y_pred = collect_predictions(model, test_dataset)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_text = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    print("\nClassification Report")
    print("-" * 60)
    print(report_text)

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report_text": report_text,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_confusion_matrix_plot(conf_matrix, class_names, output_path):
    """Save the confusion matrix as a labeled heatmap image."""

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6),
                                     max(5, len(class_names) * 0.5)))

    im = ax.imshow(conf_matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Test Set)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nConfusion matrix plot saved to:")
    print(f"  {output_path}")
