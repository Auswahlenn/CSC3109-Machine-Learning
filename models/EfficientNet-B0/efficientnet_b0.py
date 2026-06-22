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

_dropout_rate = 0.3
_learning_rate = 1e-3


def configure(*, dropout: float | None = None, learning_rate: float | None = None) -> None:
    """Configure controlled experiment factors before building the model."""
    global _dropout_rate, _learning_rate
    if dropout is not None:
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        _dropout_rate = dropout
    if learning_rate is not None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        _learning_rate = learning_rate


def get_experiment_config() -> dict[str, float]:
    """Return the active model factors for reproducibility metadata."""
    return {"dropout": _dropout_rate, "learning_rate": _learning_rate}


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

    x = keras.layers.Dropout(_dropout_rate, name="classifier_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(x)

    model = keras.Model(inputs, outputs, name="efficientnet_b0")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
