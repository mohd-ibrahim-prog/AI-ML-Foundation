"""
model_evaluator.py

Purpose:
    Generate predictions on the test set and evaluate the spam
    classifier using standard classification metrics.

Week:
    AI/ML Foundation - Week 3
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test):
    """
    Generate predictions and evaluate the spam classifier.

    Args:
        model: Trained TF-IDF + Naive Bayes pipeline.
        X_test: Test messages (cleaned text).
        y_test: True labels for the test messages.

    Returns:
        tuple: (predictions, metrics_dict)
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="spam")
    recall = recall_score(y_test, predictions, pos_label="spam")
    f1 = f1_score(y_test, predictions, pos_label="spam")
    conf_matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    report_text = classification_report(y_test, predictions)

    print("\nModel Evaluation")
    print("-" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}  (spam)")
    print(f"Recall    : {recall:.4f}  (spam)")
    print(f"F1-score  : {f1:.4f}  (spam)")

    print("\nConfusion Matrix (rows = actual, columns = predicted)")
    print("-" * 60)
    print("             Pred: ham   Pred: spam")
    print(f"Actual: ham   {conf_matrix[0][0]:<10} {conf_matrix[0][1]}")
    print(f"Actual: spam  {conf_matrix[1][0]:<10} {conf_matrix[1][1]}")

    print("\nClassification Report")
    print("-" * 60)
    print(report_text)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "report_text": report_text,
    }

    return predictions, metrics
