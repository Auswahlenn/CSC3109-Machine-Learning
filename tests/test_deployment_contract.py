from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tensorflow import keras

from frontend.inference import (
    MODEL_PATH,
    load_model,
    predict_probabilities,
    prepare_image,
)
from shared.config import CLASS_NAMES, IMAGE_SIZE


ROOT = Path(__file__).resolve().parents[1]


class FixedProbabilityModel:
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert verbose == 0
        return np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)


def test_deployment_targets_selected_efficientnet_checkpoint() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert MODEL_PATH == ROOT / "results" / "efficientnet_b0_best.keras"
    assert (
        "COPY results/efficientnet_b0_best.keras "
        "results/efficientnet_b0_best.keras"
    ) in dockerfile
    assert "custom_cnn_best.keras" not in dockerfile


def test_prepare_image_matches_training_input_contract() -> None:
    image = Image.new("L", (32, 48), color=128)

    batch = prepare_image(image)

    assert batch.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert batch.dtype == np.float32
    assert float(batch.min()) == pytest.approx(128.0)
    assert float(batch.max()) == pytest.approx(128.0)


def test_predict_probabilities_preserves_fixed_class_order() -> None:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(10, 20, 30))

    scores = predict_probabilities(FixedProbabilityModel(), image)

    assert len(scores) == len(CLASS_NAMES)
    np.testing.assert_allclose(scores, [0.1, 0.2, 0.3, 0.4], atol=1e-6)
    assert float(scores.sum()) == pytest.approx(1.0)


def test_load_model_reads_a_saved_checkpoint(tmp_path: Path) -> None:
    inputs = keras.Input(shape=(2,))
    outputs = keras.layers.Dense(4, activation="softmax")(inputs)
    checkpoint = tmp_path / "tiny.keras"
    keras.Model(inputs, outputs).save(checkpoint)

    model, error = load_model(checkpoint)

    assert error is None
    assert model is not None
    assert model.output_shape == (None, 4)


@pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason="Deployable checkpoint is distributed outside Git",
)
def test_selected_checkpoint_loads_and_predicts_real_image() -> None:
    sample = (
        ROOT
        / "dataset"
        / "val 23"
        / "coastal_mansion"
        / "coastalmansion701.jpg"
    )
    model, error = load_model()

    assert error is None
    assert model is not None
    scores = predict_probabilities(model, Image.open(sample))
    assert len(scores) == len(CLASS_NAMES)
    assert float(scores.sum()) == pytest.approx(1.0, abs=1e-5)
