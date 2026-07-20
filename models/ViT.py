"""Vision Transformer (ViT-B/16) transfer-learning model.

Fine-grained aerial image classification using a pre-trained ViT-B/16 backbone
(ImageNet weights) as a frozen feature extractor, with a lightweight
classification head on top.

ViT-B/16 splits the 224x224 input into 16x16 patches (196 patches total),
encodes them with a 12-layer Transformer (768 hidden dim, 12 heads), and uses
the CLS token embedding as the global image representation.

Preprocessing: raw [0, 255] -> [-1, 1] (ViT expects mean-centred unit-range
input, NOT the channel-wise ImageNet mean/std used by CNNs).
"""

from __future__ import annotations

import keras
import keras_hub

from shared import config


@keras.saving.register_keras_serializable(package="ViT")
class ViTPreprocess(keras.layers.Layer):
    """Scale raw [0, 255] pixels to [-1, 1] for ViT."""

    def call(self, x):
        return (keras.ops.cast(x, "float32") / 127.5) - 1.0


def build_model(
    num_classes: int, augmentation: keras.Sequential
) -> keras.Model:
    """Build and compile a ViT-B/16 transfer-learning model.

    Args:
        num_classes: Number of output classes (``config.NUM_CLASSES``).
        augmentation: Shared training-time augmentation pipeline from
            ``shared.augment.get_augmentation`` (active only during training).

    Returns:
        A compiled ``keras.Model`` outputting softmax probabilities.
    """
    inputs = keras.Input(
        shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3), name="image"
    )

    # 1. Shared augmentation (raw [0, 255] in, raw [0, 255] out).
    x = augmentation(inputs)

    # 2. ViT-specific preprocessing: [0, 255] -> [-1, 1].
    x = ViTPreprocess(name="vit_preprocess")(x)

    # 3. Pre-trained ViT-B/16 backbone as a frozen feature extractor.
    #    Returns token sequence (batch, num_patches+1, 768); index 0 is CLS.
    backbone = keras_hub.models.ViTBackbone.from_preset(
        "vit_base_patch16_224_imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = x[:, 0, :]  # CLS token embedding, shape: (batch, 768)

    # 4. New classification head.
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="vit_b16")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
