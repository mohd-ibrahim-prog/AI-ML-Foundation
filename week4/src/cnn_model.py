"""
Week 4 - Convolutional Neural Network
Plant Disease Detector
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(num_classes):

    model = models.Sequential([
        
        # Input
        layers.Input(shape=(128, 128, 3)),

        # CNN Block 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        # Classification
        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.3),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model