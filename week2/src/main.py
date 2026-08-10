from pathlib import Path
import sys
import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from data_preprocessor import prepare_data
from model_trainer import train_model
from model_evaluator import evaluate_model


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "plant_growth_dataset.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "cleaned_plant_growth_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "linear_regression_model.pkl"
PREDICTIONS_FILE = BASE_DIR / "outputs" / "predictions.csv"
REPORT_FILE = BASE_DIR / "reports" / "model_report.txt"


def save_outputs(model, predictions, X_test, y_test, metrics):
    """Save the trained model, predictions, and evaluation report."""

    # Save the trained model
    joblib.dump(model, MODEL_FILE)

    # Save predictions alongside actual values
    prediction_data = X_test.copy()
    prediction_data["Actual_Growth"] = y_test.values
    prediction_data["Predicted_Growth"] = predictions
    prediction_data.to_csv(PREDICTIONS_FILE, index=False)

    # Save a simple evaluation report
    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("WEEK 2 - LINEAR REGRESSION MODEL REPORT\n")
        report.write("=" * 45 + "\n\n")

        report.write("Model: Linear Regression\n\n")

        report.write("Evaluation Metrics\n")
        report.write("-" * 20 + "\n")
        report.write(f"MAE  : {metrics['MAE']:.4f}\n")
        report.write(f"MSE  : {metrics['MSE']:.4f}\n")
        report.write(f"RMSE : {metrics['RMSE']:.4f}\n")
        report.write(f"R2   : {metrics['R2']:.4f}\n")

    print("\nFiles generated successfully:")
    print(f"Model       : {MODEL_FILE}")
    print(f"Predictions : {PREDICTIONS_FILE}")
    print(f"Report      : {REPORT_FILE}")


def main():
    print("=" * 60)
    print("             AI/ML FOUNDATION - WEEK 2")
    print("              LINEAR REGRESSION MODEL")
    print("=" * 60)

    # Load, clean and split the dataset
    X_train, X_test, y_train, y_test = prepare_data(RAW_DATA)

    # Save the cleaned source dataset
    cleaned_data = pd.read_csv(RAW_DATA).dropna()
    cleaned_data.to_csv(PROCESSED_DATA, index=False)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    predictions, metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save all important outputs
    save_outputs(
        model,
        predictions,
        X_test,
        y_test,
        metrics
    )

    print("\n" + "=" * 60)
    print("             WEEK 2 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()