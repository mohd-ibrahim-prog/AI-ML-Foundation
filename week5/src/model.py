"""
model.py

Purpose:
    Define the CNN architecture used to classify plant disease
    images. This reuses the same architecture introduced in
    Week 4, since Week 5's goal is training/validation, not a
    new architecture.

Week:
    AI/ML Foundation - Week 5
"""

from tensorflow.keras import layers, models

from dataset_utils import IMAGE_SIZE


def build_cnn(num_classes):

    model = models.Sequential([

        layers.Input(shape=(*IMAGE_SIZE, 3)),

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
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
