"""
data_loader.py

Purpose:
    Rebuild the exact same held-out TEST dataset that Week 6 used,
    so this week's per-class evaluation is measured on data the
    Week 6 model never saw during training.

Week:
    AI/ML Foundation - Week 7
"""

import tensorflow as tf

from dataset_utils import IMAGE_SIZE, BATCH_SIZE, build_train_val_test_split


def load_image(path, label):
    """Read, decode, resize and normalize a single image."""

    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_test_dataset(image_files, labels, class_names,
                         val_size=0.15, test_size=0.15):
    """
    Rebuild the train/val/test split (identical to Week 6, since
    it uses the same seed, split ratios and sorted file order) and
    return only the test portion as a tf.data.Dataset.

    Returns:
        test_dataset, test_files, test_labels
    """

    split = build_train_val_test_split(
        image_files, labels, class_names,
        val_size=val_size, test_size=test_size,
    )

    test_files = split["test_files"]
    test_labels = split["test_labels"]

    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_dataset = (
        test_dataset
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return test_dataset, test_files, test_labels
