"""
data_loader.py

Purpose:
    Build training and validation tf.data.Dataset objects from the
    PlantVillage images discovered by dataset_utils.py.

    This follows the same image loading/preprocessing approach
    introduced in Week 4 (resize + normalize), but this week the
    split is explicitly a train/validation split, since Week 5's
    goal is to demonstrate training WITH validation.

Week:
    AI/ML Foundation - Week 5
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataset_utils import IMAGE_SIZE, BATCH_SIZE, SEED


def load_image(path, label):
    """Read, decode, resize and normalize a single image."""

    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def build_train_val_split(image_files, labels, class_names, val_size=0.2):
    """
    Create a stratified train/validation split.

    Returns:
        train_files, val_files, train_labels, val_labels (numpy arrays)
    """

    label_to_index = {label: index for index, label in enumerate(class_names)}
    numeric_labels = np.array([label_to_index[label] for label in labels])
    file_paths = np.array([str(path) for path in image_files])

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths,
        numeric_labels,
        test_size=val_size,
        random_state=SEED,
        stratify=numeric_labels,
    )

    print("\nTrain / Validation split:")
    print(f"Training images  : {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    return train_files, val_files, train_labels, val_labels


def create_datasets(image_files, labels, class_names, val_size=0.2):
    """
    Build the training and validation tf.data.Dataset pipelines.

    Returns:
        train_dataset, val_dataset
    """

    train_files, val_files, train_labels, val_labels = build_train_val_split(
        image_files, labels, class_names, val_size=val_size
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

    train_dataset = (
        train_dataset
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset
