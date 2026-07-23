"""FROZEN -- do not edit without notifying the team; changes invalidate prior results.

Shared data augmentation.

A single augmentation pipeline shared by all members so that the regularization
applied during training is identical -- only the backbone/head should differ
between members.

Aerial imagery has no inherent "up": a scene is equally valid flipped or rotated
in any direction. That makes full horizontal+vertical flips and arbitrary
rotations label-preserving, unlike natural photos. Photometric jitter (contrast,
brightness) is kept MILD so we model sensor/lighting variation without distorting
the fine-grained cues that separate these confusable categories.

WARNING: This pipeline is for TRAINING ONLY. Validation data must NEVER be
augmented -- evaluation must see the clean, untouched held-out images so the
reported metrics reflect true generalization. (The Keras RandomX layers are
inactive at inference time, but never deliberately apply this to ``val_ds``.)
"""

from __future__ import annotations

from tensorflow import keras

from . import config


def get_augmentation() -> keras.Sequential:
    """Build the shared training-time augmentation pipeline.

    The returned ``Sequential`` of preprocessing layers is meant to be embedded
    as the first stage of each member's model (so it is active during
    ``fit``/training and automatically bypassed during inference). It operates on
    RAW ``[0, 255]`` pixels and leaves values in the same range, so it must come
    BEFORE the backbone's ``preprocess_input`` in each model.

    Returns:
        A ``keras.Sequential`` of random augmentation layers.
    """
    return keras.Sequential(
        [
            # All four flip orientations are valid for top-down aerial views.
            keras.layers.RandomFlip(
                "horizontal_and_vertical", seed=config.SEED
            ),
            keras.layers.RandomRotation(
                factor=0.5, fill_mode="reflect", seed=config.SEED
            ),
            keras.layers.RandomZoom(0.2, seed=config.SEED),
            keras.layers.RandomContrast(factor=0.1, seed=config.SEED),
            keras.layers.RandomBrightness(
                factor=0.1, value_range=(0, 255), seed=config.SEED
            ),
        ],
        name="aerial_augmentation",
    )
