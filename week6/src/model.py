"""
model.py

Purpose:
    Define an improved CNN architecture for plant disease
    classification.

    Compared to Week 4/5, this adds:
    - Built-in data augmentation layers (random flip / rotation /
      zoom), active only during training, to reduce overfitting
      on a limited number of images per class.
    - An extra Dropout layer.

    These are standard, beginner-appropriate techniques - no
    unnecessary complexity was added.

Week:
    AI/ML Foundation - Week 6
"""

from tensorflow.keras import layers, models

from dataset_utils import IMAGE_SIZE


def build_augmentation():
    """Simple, standard image augmentation, active only in training."""

    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")


def build_improved_cnn(num_classes):

    model = models.Sequential([

        layers.Input(shape=(*IMAGE_SIZE, 3)),

        # Data augmentation (training only)
        build_augmentation(),

        # CNN Block 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Classification head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
