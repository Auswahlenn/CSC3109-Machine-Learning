"""FROZEN -- do not edit without notifying the team; changes invalidate prior results.

Shared project configuration.

This module is the single source of truth for the constants every team member
relies on (image size, batch size, class list, data paths, RNG seed). Keeping
these identical across all five members is what makes their reported numbers
directly comparable.
"""

from __future__ import annotations

import os

import tensorflow as tf
from tensorflow import keras

# --- Image / model geometry ------------------------------------------------
# 256 is the native input size for most ImageNet backbones (ResNet, EfficientNet
# B0, MobileNet, etc.). Keep it fixed so every member feeds the backbone the
# same spatial resolution.
IMAGE_SIZE: int = 256

# Batch size used for both training and evaluation. Comparable hardware is
# assumed; if a member must change this for memory reasons it does NOT affect
# evaluation metrics, but keep it fixed for comparable training dynamics.
BATCH_SIZE: int = 32

# Four confusable fine-grained aerial categories.
NUM_CLASSES: int = 4

# Single global seed used everywhere (data shuffling, augmentation, weight init
# where applicable) so runs are reproducible and comparable across members.
SEED: int = 42

# --- Class names -----------------------------------------------------------
# Order is FIXED and must match the alphabetical directory order that
# keras.utils.image_dataset_from_directory produces, so that the integer /
# one-hot label index is identical to this list's index. Do not reorder.
CLASS_NAMES: list[str] = [
    "coastal_mansion",
    "dense_residential",
    "nursing_home",
    "sparse_residential",
]

# --- Data paths ------------------------------------------------------------
# Paths are resolved relative to the repository root (the parent of this
# `shared/` package) so the project works regardless of the current working
# directory. The professor-provided folders are literally named "set 23"
# (training, 700 images/class) and "val 23" (held-out validation,
# 100 images/class).
_REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR: str = os.environ.get(
    "CSC3109_DATA_DIR", os.path.join(_REPO_ROOT, "dataset")
)

TRAIN_DIR: str = os.path.join(_DATA_DIR, "set 23")
VAL_DIR: str = os.path.join(_DATA_DIR, "val 23")

# Derive tuning data only from the professor-provided training split. The
# separate validation directory remains evaluation-only.
TUNING_SPLIT: float = 0.15

# Exact byte-identical images assigned to conflicting classes. The source
# files remain untouched and are excluded logically by the shared loader.
EXCLUDED_TRAIN_FILES: tuple[str, ...] = (
    "coastal_mansion/coastalmansion029.jpg",
    "sparse_residential/sparseresidential021.jpg",
)

# Where evaluate.py writes per-model JSON + confusion-matrix PNG.
RESULTS_DIR: str = os.path.join(_REPO_ROOT, "results")


def set_seed(seed: int = SEED) -> None:
    """Seed all RNGs and enable deterministic ops for reproducibility.

    Call this once at the very start of every script (before building datasets
    or models) so that data shuffling, augmentation, and weight initialization
    are reproducible and comparable across team members.

    Args:
        seed: Integer seed to apply. Defaults to the shared project ``SEED``.
    """
    keras.utils.set_random_seed(seed)
    # Forces deterministic GPU/CPU kernels. This lives on tf.config (not Keras)
    # because op determinism is a property of the TensorFlow backend runtime.
    # Costs some speed but improves repeatability within the recorded software
    # and hardware environment, which supports fair comparison.
    tf.config.experimental.enable_op_determinism()
