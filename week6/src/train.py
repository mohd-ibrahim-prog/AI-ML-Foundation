"""
train.py

Purpose:
    Train the improved CNN with:
    - Early stopping (stop once validation loss stops improving,
      and restore the best weights seen during training)
    - Model checkpointing (save the best model seen so far)

Week:
    AI/ML Foundation - Week 6
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(model, train_dataset, val_dataset, checkpoint_path,
                 epochs=15, patience=3):
    """
    Train the model with early stopping and checkpointing.

    Returns:
        history: the Keras History object from model.fit()
    """

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    print("\nStarting training (with early stopping + checkpointing)...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history
