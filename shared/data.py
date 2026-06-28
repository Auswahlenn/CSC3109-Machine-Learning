"""Deterministic dataset loading without held-out evaluation leakage.

The professor-provided ``set 23`` directory is split reproducibly into model
training and tuning subsets. ``val 23`` is loaded separately and is never used
by ``model.fit`` or checkpoint selection.

Images remain raw float32 values in ``[0, 255]``. Backbone-specific
preprocessing belongs inside each saved model.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import tensorflow as tf

from . import config

LabeledPath = tuple[str, int]

def get_datasets(batch_size: int | None = None) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load the fixed train and validation datasets.

    The train/validation split is fixed by the professor (separate directories),
    so this function NEVER re-splits -- it loads each directory as-is.

    Args:
        batch_size: Optional override for the batch size. Defaults to
            ``config.BATCH_SIZE``. This exists only so members with limited GPU
            memory can fit large models -- it does NOT affect the data, labels,
            split, or evaluation, so results remain comparable. Leaving it
            ``None`` preserves the original shared behaviour.

    Returns:
        A ``(train_ds, val_ds)`` tuple of batched ``tf.data.Dataset`` objects.
        Each element is ``(images, labels)`` where ``images`` are RAW float32
        pixels in ``[0, 255]`` of shape ``(batch, IMAGE_SIZE, IMAGE_SIZE, 3)``
        and ``labels`` are one-hot vectors of shape ``(batch, NUM_CLASSES)``
        (``label_mode="categorical"``).

        Train is shuffled (with the fixed seed); val is NOT shuffled so that the
        prediction order matches the label order in evaluate.py.
    """
    batch_size = batch_size or config.BATCH_SIZE

    train_ds = keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,  # pin label order to the shared contract
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=config.SEED,

def _collect_labeled_paths(
    root: str, excluded_relative_paths: Iterable[str] = ()
) -> list[list[LabeledPath]]:
    """Collect sorted image paths grouped by the fixed class order."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root_path}")

    excluded = {Path(path).as_posix() for path in excluded_relative_paths}
    grouped: list[list[LabeledPath]] = []
    for label, class_name in enumerate(config.CLASS_NAMES):
        class_dir = root_path / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

        class_paths: list[LabeledPath] = []
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            relative_path = path.relative_to(root_path).as_posix()
            if relative_path in excluded:
                continue
            class_paths.append((str(path), label))
        if not class_paths:
            raise ValueError(f"No usable images found in: {class_dir}")
        grouped.append(class_paths)
    return grouped


def _split_training_paths() -> tuple[list[LabeledPath], list[LabeledPath]]:
    """Create a deterministic stratified train/tuning split."""
    grouped = _collect_labeled_paths(
        config.TRAIN_DIR, excluded_relative_paths=config.EXCLUDED_TRAIN_FILES
    )
    training: list[LabeledPath] = []
    tuning: list[LabeledPath] = []

    for label, class_paths in enumerate(grouped):
        shuffled = class_paths.copy()
        random.Random(config.SEED + label).shuffle(shuffled)
        tuning_count = max(1, round(len(shuffled) * config.TUNING_SPLIT))
        tuning.extend(shuffled[:tuning_count])
        training.extend(shuffled[tuning_count:])

    random.Random(config.SEED).shuffle(training)
    random.Random(config.SEED).shuffle(tuning)
    return training, tuning


def _decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(
        image,
        (config.IMAGE_SIZE, config.IMAGE_SIZE),
        method="bilinear",
        antialias=True,
    )
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0)
    return image, tf.one_hot(label, depth=config.NUM_CLASSES)


def _build_dataset(
    labeled_paths: list[LabeledPath], *, shuffle: bool
) -> tf.data.Dataset:
    paths = [path for path, _ in labeled_paths]
    labels = [label for _, label in labeled_paths]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        _decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(
            len(labeled_paths),
            seed=config.SEED,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    val_ds = keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=False,  # keep deterministic order for evaluation
def get_training_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Return deterministic training and internal tuning datasets."""
    training_paths, tuning_paths = _split_training_paths()
    return (
        _build_dataset(training_paths, shuffle=True),
        _build_dataset(tuning_paths, shuffle=False),
    )


def get_held_out_dataset() -> tf.data.Dataset:
    """Return the untouched professor-provided held-out dataset."""
    grouped = _collect_labeled_paths(config.VAL_DIR)
    held_out_paths = [item for class_paths in grouped for item in class_paths]
    return _build_dataset(held_out_paths, shuffle=False)


def get_split_counts() -> dict[str, object]:
    """Return the exact reproducible split counts for experiment metadata."""
    training_paths, tuning_paths = _split_training_paths()
    held_out_grouped = _collect_labeled_paths(config.VAL_DIR)

    def per_class(paths: list[LabeledPath]) -> dict[str, int]:
        counts = {class_name: 0 for class_name in config.CLASS_NAMES}
        for _, label in paths:
            counts[config.CLASS_NAMES[label]] += 1
        return counts

    held_out_paths = [item for group in held_out_grouped for item in group]
    return {
        "training": {
            "total": len(training_paths),
            "per_class": per_class(training_paths),
        },
        "tuning": {
            "total": len(tuning_paths),
            "per_class": per_class(tuning_paths),
        },
        "held_out": {
            "total": len(held_out_paths),
            "per_class": per_class(held_out_paths),
        },
        "excluded_training_files": list(config.EXCLUDED_TRAIN_FILES),
    }


def get_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Backward-compatible alias returning training and internal tuning data."""
    return get_training_datasets()
