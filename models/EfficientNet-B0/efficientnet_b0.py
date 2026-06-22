"""EfficientNet-B0 transfer-learning model for aerial scene classification.

This module follows the shared model contract consumed by ``train.py``::

    build_model(num_classes: int, augmentation: keras.Sequential) -> keras.Model

The saved model owns its preprocessing so it accepts the raw ``[0, 255]``
images returned by ``shared.data.get_datasets``.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from shared import config


def build_model(
    num_classes: int, augmentation: keras.Sequential
) -> keras.Model:
    """Build and compile a frozen-backbone EfficientNet-B0 classifier.

    Args:
        num_classes: Number of output categories.
        augmentation: Shared training-only augmentation pipeline.

    Returns:
        A compiled Keras model that outputs softmax class probabilities.
    """
    inputs = keras.Input(
        shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3), name="image"
    )

    x = augmentation(inputs)
    x = preprocess_input(x)

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = keras.layers.Dropout(0.3, name="classifier_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(x)

    model = keras.Model(inputs, outputs, name="efficientnet_b0")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
