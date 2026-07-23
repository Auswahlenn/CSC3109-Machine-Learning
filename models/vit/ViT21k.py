"""Vision Transformer (ViT-B/16) ablation: ImageNet-21k pretrained weights.

Identical architecture and head to ``ViT.py`` (frozen ViT-B/16 feature
extractor, CLS token, dropout 0.3, softmax head); the ONLY change is the
pretraining checkpoint. ``ViT.py`` uses ``vit_base_patch16_224_imagenet``
(pretrained on ImageNet-21k, then finetuned on ImageNet-1k); this variant
uses ``vit_base_patch16_224_imagenet21k`` (the raw 21k-pretrained weights,
never finetuned to the 1k label space).

Rationale for the ablation: aerial imagery is a domain shift from natural
photos, so features finetuned toward the 1k object categories may be more
specialised than helpful, while the 21k weights may transfer more generic
structure. Whichever way it lands, the comparison is reportable.

Both variants share the same seed, augmentation, split, and training
protocol, so tuning-split metrics are directly comparable.
"""

from __future__ import annotations

import keras
import keras_hub

from shared import config

_VIT_INPUT_SIZE = 224


@keras.saving.register_keras_serializable(package="ViT21k")
class ViTPreprocess(keras.layers.Layer):
    """Scale raw [0, 255] pixels to [-1, 1] for ViT."""

    def call(self, x):
        return (keras.ops.cast(x, "float32") / 127.5) - 1.0


def build_model(
    num_classes: int, augmentation: keras.Sequential
) -> keras.Model:
    """Build and compile the ImageNet-21k ViT-B/16 ablation variant.

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

    # 2. Resize 256x256 (shared config.IMAGE_SIZE) -> 224x224, the fixed grid
    #    this preset's positional embeddings were trained on.
    x = keras.layers.Resizing(
        _VIT_INPUT_SIZE, _VIT_INPUT_SIZE, name="vit_resize"
    )(x)

    # 3. ViT-specific preprocessing: [0, 255] -> [-1, 1].
    x = ViTPreprocess(name="vit_preprocess")(x)

    # 4. Pre-trained ViT-B/16 backbone as a frozen feature extractor.
    #    Returns token sequence (batch, num_patches+1, 768); index 0 is CLS.
    backbone = keras_hub.models.ViTBackbone.from_preset(
        "vit_base_patch16_224_imagenet21k",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = x[:, 0, :]  # CLS token embedding, shape: (batch, 768)

    # 5. New classification head.
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="vit_b16_21k")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
