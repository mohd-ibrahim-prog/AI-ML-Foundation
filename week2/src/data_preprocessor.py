import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(file_path):
    """Load the dataset and prepare it for model training."""

    data = pd.read_csv(file_path)

    print("\nDataset loaded successfully.")
    print(f"Rows    : {len(data)}")
    print(f"Columns : {len(data.columns)}")

    # Check for missing values before training
    missing_values = data.isnull().sum()

    if missing_values.any():
        print("\nMissing values found:")
        print(missing_values[missing_values > 0])

        data = data.dropna()
        print(f"\nRows after removing missing values: {len(data)}")
    else:
        print("\nNo missing values found.")

    # Features used to make the prediction
    feature_columns = [
        "Temperature",
        "Humidity",
        "Soil_Moisture",
        "Sunlight_Hours",
        "Fertilizer_Amount"
    ]

    # Value we want the model to predict
    target_column = "Growth"

    X = data[feature_columns]
    y = data[target_column]

    # Keep 20% of the data for testing the trained model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nFeatures:")
    print(feature_columns)

    print(f"\nTarget: {target_column}")

    print("\nTrain/Test Split")
    print("----------------")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test