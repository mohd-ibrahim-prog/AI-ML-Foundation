"""
train.py

Purpose:
    Train the CNN model on the training dataset, validating after
    every epoch on the validation dataset.

Week:
    AI/ML Foundation - Week 5
"""


def train_model(model, train_dataset, val_dataset, epochs=5):
    """
    Train the model with validation.

    Returns:
        history: the Keras History object returned by model.fit(),
                  containing per-epoch loss/accuracy for both the
                  training and validation sets.
    """

    print("\nStarting training (with validation)...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
    )

    return history
