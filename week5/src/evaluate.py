"""
evaluate.py

Purpose:
    Evaluate the trained model on the validation dataset and save
    the training history (accuracy/loss per epoch) as a CSV and a
    simple plot, so training/validation behaviour can be inspected.

Week:
    AI/ML Foundation - Week 5
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_on_validation(model, val_dataset):
    """Evaluate the model on the validation dataset."""

    print("\nEvaluating on validation dataset...")

    val_loss, val_accuracy = model.evaluate(val_dataset)

    print(f"Validation Loss     : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_accuracy:.4f}")

    return val_loss, val_accuracy


def save_training_history(history, history_csv_path, history_plot_path):
    """Save per-epoch training/validation accuracy and loss."""

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(history_csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history_df["epoch"], history_df["accuracy"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["loss"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(history_plot_path)
    plt.close(fig)

    print(f"\nTraining history saved to:")
    print(f"  {history_csv_path}")
    print(f"  {history_plot_path}")
