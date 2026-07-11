"""MobileNetV2 transfer-learning model for aerial scene classification.

This module follows the shared model contract consumed by ``train.py``::

    build_model(num_classes: int, augmentation: keras.Sequential) -> keras.Model

Stack order (required):
    1. shared ``augmentation`` (passed in -- do not build your own),
    2. MobileNetV2 ``preprocess_input`` (raw [0, 255] from the shared loader),
    3. pretrained MobileNetV2 backbone (frozen feature extractor),
    4. classification head ending in softmax over ``num_classes``.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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
    """Build and compile a frozen-backbone MobileNetV2 classifier.

    Args:
        num_classes: Number of output categories (``config.NUM_CLASSES``).
        augmentation: Shared training-only augmentation pipeline from
            ``shared.augment.get_augmentation``.

    Returns:
        A compiled Keras model that outputs softmax class probabilities.
    """
    inputs = keras.Input(
        shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3), name="image"
    )

    # 1. Shared augmentation (raw [0, 255] in, raw [0, 255] out).
    x = augmentation(inputs)

    # 2. Backbone-specific preprocessing -- lives here, not in shared/data.py.
    x = preprocess_input(x)

    # 3. Pretrained backbone as a frozen feature extractor.
    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    # 4. New classification head.
    x = keras.layers.Dropout(_dropout_rate, name="classifier_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(x)

    model = keras.Model(inputs, outputs, name="mobilenet_v2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
