"""FROZEN -- do not edit without notifying the team; changes invalidate prior results.

Shared dataset loading.

Every team member loads data through :func:`get_datasets` so that the exact same
images, label encoding, batching, and (for training) shuffle order are used.

IMPORTANT -- NO normalization here:
    Different pretrained backbones expect different input scaling. ResNet uses
    Caffe-style mean subtraction, EfficientNet expects raw [0, 255], MobileNetV2
    expects [-1, 1], etc. Baking one ``preprocess_input`` into the shared loader
    would silently break every other backbone. Therefore this module returns the
    RAW datasets (pixel values in [0, 255], float32). Each member applies their
    own ``keras.applications.<backbone>.preprocess_input`` INSIDE their model
    (see models/*.py), so preprocessing travels with the saved model.
"""

from __future__ import annotations

import tensorflow as tf  # only for the tf.data pipeline (cache/prefetch/AUTOTUNE)
from tensorflow import keras

from . import config


def get_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load the fixed train and validation datasets.

    The train/validation split is fixed by the professor (separate directories),
    so this function NEVER re-splits -- it loads each directory as-is.

    Returns:
        A ``(train_ds, val_ds)`` tuple of batched ``tf.data.Dataset`` objects.
        Each element is ``(images, labels)`` where ``images`` are RAW float32
        pixels in ``[0, 255]`` of shape ``(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)``
        and ``labels`` are one-hot vectors of shape ``(BATCH_SIZE, NUM_CLASSES)``
        (``label_mode="categorical"``).

        Train is shuffled (with the fixed seed); val is NOT shuffled so that the
        prediction order matches the label order in evaluate.py.
    """
    train_ds = keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,  # pin label order to the shared contract
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.SEED,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # keep deterministic order for evaluation
    )

    # Cache decoded images in memory and prefetch the next batch while the GPU
    # works on the current one. Order matters: cache the raw decode result, then
    # prefetch.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds
