"""
Week 4 - Plant Disease Detector

Workflow:
1. Load PlantVillage dataset
2. Detect disease classes
3. Split dataset
4. Build CNN
5. Train CNN
6. Evaluate model
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_dataset_info, create_datasets
from cnn_model import build_cnn


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("        AI/ML FOUNDATION - WEEK 4")
    print("        PLANT DISEASE DETECTOR")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. LOAD DATASET
    # -----------------------------------------------------

    image_files, labels, class_names = load_dataset_info()

    # -----------------------------------------------------
    # 2. CREATE TRAIN / TEST DATASETS
    # -----------------------------------------------------

    train_dataset, test_dataset = create_datasets(
        image_files,
        labels,
        class_names
    )

    # -----------------------------------------------------
    # 3. BUILD CNN
    # -----------------------------------------------------

    print("\nBuilding CNN model...")

    model = build_cnn(
        num_classes=len(class_names)
    )

    model.summary()

    # -----------------------------------------------------
    # 4. TRAIN MODEL
    # -----------------------------------------------------

    print("\nStarting CNN training...")

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=5
    )

    # -----------------------------------------------------
    # 5. EVALUATE
    # -----------------------------------------------------

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(test_dataset)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")

    # -----------------------------------------------------
    # 6. SAVE MODEL
    # -----------------------------------------------------

    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "plant_disease_cnn.keras"
    )

    model.save(model_path)

    print(f"\nModel saved to:")
    print(model_path)

    print("\nWeek 4 completed successfully!")


if __name__ == "__main__":
    main()