"""
Week 4 - Plant Disease Detector
Dataset Loader

Loads PlantVillage images and extracts disease labels
from filenames or folder names.
"""

import os
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "PlantVillage"


# ---------------------------------------------------------
# LABEL EXTRACTION
# ---------------------------------------------------------

def extract_label(file_path):
    """
    Extract disease label from the dataset.

    Supports:
    1. Folder-based labels
    2. Filename-based labels

    Example:
    abc123__FREC_Scab 3112.JPG

    -> FREC_Scab
    """

    file_path = Path(file_path)

    # If dataset uses class folders
    parent_name = file_path.parent.name

    if parent_name != "PlantVillage":
        return parent_name

    # Otherwise extract label from filename
    filename = file_path.stem

    if "__" in filename:
        label_part = filename.split("__", 1)[1]

        # Remove trailing numeric image ID
        label_part = re.sub(r"\s+\d+$", "", label_part)

        return label_part.strip()

    return "Unknown"


# ---------------------------------------------------------
# FIND IMAGES
# ---------------------------------------------------------

def find_images():
    """Find all JPG/JPEG/PNG images in the dataset."""

    extensions = {".jpg", ".jpeg", ".png"}

    image_files = []

    for file_path in DATASET_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(file_path)

    return image_files


# ---------------------------------------------------------
# LOAD DATASET INFORMATION
# ---------------------------------------------------------

def load_dataset_info():

    print("\n" + "=" * 60)
    print("PLANT DISEASE DETECTOR - WEEK 4")
    print("=" * 60)

    print(f"\nDataset location:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    image_files = find_images()

    if not image_files:
        raise ValueError("No image files found in PlantVillage.")

    labels = [extract_label(path) for path in image_files]

    unique_labels = sorted(set(labels))

    print(f"\nTotal images found : {len(image_files)}")
    print(f"Total disease classes: {len(unique_labels)}")

    print("\nDisease classes:")

    for index, label in enumerate(unique_labels, start=1):
        count = labels.count(label)
        print(f"{index:2}. {label:<35} {count} images")

    return image_files, labels, unique_labels


# ---------------------------------------------------------
# CREATE DATASET
# ---------------------------------------------------------

def create_datasets(image_files, labels, class_names):

    label_to_index = {
        label: index
        for index, label in enumerate(class_names)
    }

    numeric_labels = np.array(
        [label_to_index[label] for label in labels]
    )

    image_files = np.array(
        [str(path) for path in image_files]
    )

    # Stratified split
    train_files, test_files, train_labels, test_labels = train_test_split(
        image_files,
        numeric_labels,
        test_size=0.2,
        random_state=42,
        stratify=numeric_labels
    )

    print("\nDataset split:")
    print(f"Training images : {len(train_files)}")
    print(f"Testing images  : {len(test_files)}")

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_files, train_labels)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_files, test_labels)
    )

    def load_image(path, label):

        image = tf.io.read_file(path)

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    train_dataset = (
        train_dataset
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        test_dataset
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, test_dataset