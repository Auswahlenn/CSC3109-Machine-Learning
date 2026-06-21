"""Example / template model -- ResNet50 transfer learning.

Copy this file to your own ``models/<yourname>.py`` and swap the backbone +
``preprocess_input`` for the architecture you are assigned. This file is NOT
frozen; each member owns their own model module. The ONLY hard requirement is
the function signature below, which train.py depends on::

    build_model(num_classes: int, augmentation: keras.Sequential) -> keras.Model

Your model MUST internally stack, in this order:
    1. the shared `augmentation` (passed in -- do not build your own),
    2. your backbone's `preprocess_input` (preprocessing travels with the model;
       the shared data loader returns RAW [0, 255] pixels on purpose),
    3. the pretrained backbone (frozen for feature extraction),
    4. a new classification head ending in softmax over `num_classes`.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from shared import config


def build_model(
    num_classes: int, augmentation: keras.Sequential
) -> keras.Model:
    """Build and compile a ResNet50 transfer-learning model.

    Args:
        num_classes: Number of output classes (``config.NUM_CLASSES``).
        augmentation: The shared training-time augmentation pipeline from
            ``shared.augment.get_augmentation`` (active only during training).

    Returns:
        A compiled ``keras.Model`` outputting softmax probabilities.
    """
    inputs = keras.Input(
        shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3), name="image"
    )

    # 1. Shared augmentation (raw [0, 255] in, raw [0, 255] out).
    x = augmentation(inputs)

    # 2. Backbone-specific preprocessing -- different for every backbone, which
    #    is exactly why it lives here and NOT in shared/data.py.
    x = preprocess_input(x)

    # 3. Pretrained backbone as a frozen feature extractor.
    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    # 4. New classification head.
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="example_resnet50")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
