from __future__ import annotations

import numpy as np

from shared import config
from shared.data import (
    _split_training_paths,
    get_held_out_dataset,
    get_split_counts,
    get_training_datasets,
)


def test_split_is_deterministic_disjoint_and_excludes_conflicts() -> None:
    first_training, first_tuning = _split_training_paths()
    second_training, second_tuning = _split_training_paths()

    assert first_training == second_training
    assert first_tuning == second_tuning

    training_paths = {path for path, _ in first_training}
    tuning_paths = {path for path, _ in first_tuning}
    assert training_paths.isdisjoint(tuning_paths)

    combined_paths = training_paths | tuning_paths
    for relative_path in config.EXCLUDED_TRAIN_FILES:
        assert not any(
            path.replace("\\", "/").endswith(relative_path)
            for path in combined_paths
        )


def test_split_counts_match_the_verified_dataset() -> None:
    counts = get_split_counts()

    assert counts["training"]["total"] == 2378
    assert counts["tuning"]["total"] == 420
    assert counts["held_out"]["total"] == 400
    assert counts["training"]["per_class"] == {
        "coastal_mansion": 594,
        "dense_residential": 595,
        "nursing_home": 595,
        "sparse_residential": 594,
    }


def test_dataset_batches_have_shared_shape_and_raw_range() -> None:
    training, tuning = get_training_datasets()
    held_out = get_held_out_dataset()

    for dataset in (training, tuning, held_out):
        images, labels = next(iter(dataset))
        assert images.shape[1:] == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
        assert labels.shape[1:] == (config.NUM_CLASSES,)
        assert images.dtype.name == "float32"
        assert float(np.min(images.numpy())) >= 0.0
        assert float(np.max(images.numpy())) <= 255.0
        np.testing.assert_allclose(labels.numpy().sum(axis=1), 1.0)
