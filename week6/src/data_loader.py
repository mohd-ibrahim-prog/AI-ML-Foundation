"""
data_loader.py

Purpose:
    Build training, validation and test tf.data.Dataset objects
    from the PlantVillage images.

    Week 6 introduces a genuine held-out TEST set (in addition to
    the train/validation split from Week 5), since this week's
    focus is proper model evaluation, not just training.

Week:
    AI/ML Foundation - Week 6
"""

import tensorflow as tf

from dataset_utils import IMAGE_SIZE, BATCH_SIZE, SEED, build_train_val_test_split


def load_image(path, label):
    """Read, decode, resize and normalize a single image."""

    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def _make_dataset(files, labels, shuffle, batch_size=BATCH_SIZE):

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1000, seed=SEED)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def create_datasets(image_files, labels, class_names,
                     val_size=0.15, test_size=0.15):
    """
    Build train, validation and test tf.data.Dataset pipelines
    from a stratified 3-way split.

    Returns:
        train_dataset, val_dataset, test_dataset
    """

    split = build_train_val_test_split(
        image_files, labels, class_names,
        val_size=val_size, test_size=test_size,
    )

    train_dataset = _make_dataset(split["train_files"], split["train_labels"], shuffle=True)
    val_dataset = _make_dataset(split["val_files"], split["val_labels"], shuffle=False)
    test_dataset = _make_dataset(split["test_files"], split["test_labels"], shuffle=False)

    return train_dataset, val_dataset, test_dataset
