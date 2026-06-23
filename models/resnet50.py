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
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="daryl")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
