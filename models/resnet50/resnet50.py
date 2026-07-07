"""ResNet50 model.

This model uses the ResNet50 architecture pretrained on ImageNet
as a frozen feature extractor. It compiles the model with Adam and categorical crossentropy loss,
internally stacking data augmentation, preprocessing, the ResNet50 backbone, and a classification head.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

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

    # 2. Backbone-specific preprocessing.
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
    x = keras.layers.Dropout(_dropout_rate, name="classifier_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities"
    )(x)

    model = keras.Model(inputs, outputs, name="resnet50")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
