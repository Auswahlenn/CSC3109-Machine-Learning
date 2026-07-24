"""Model-loading and image-inference helpers for the Streamlit deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shared.config import CLASS_NAMES, IMAGE_SIZE


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "results" / "efficientnet_b0_best.keras"


def load_model(model_path: str | Path = MODEL_PATH) -> tuple[Any | None, str | None]:
    """Load a Keras checkpoint for inference without restoring training state."""
    try:
        from tensorflow import keras

        return keras.models.load_model(Path(model_path), compile=False), None
    except Exception as exc:
        return None, str(exc)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Convert one image to the raw float32 batch expected by saved models."""
    rgb_image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.expand_dims(np.asarray(rgb_image, dtype=np.float32), axis=0)


def predict_probabilities(model: Any, image: Image.Image) -> np.ndarray:
    """Return one validated probability for every fixed project class."""
    probabilities = np.asarray(
        model.predict(prepare_image(image), verbose=0),
        dtype=np.float64,
    )
    if probabilities.shape != (1, len(CLASS_NAMES)):
        raise ValueError(
            "Model output must have shape "
            f"(1, {len(CLASS_NAMES)}), got {probabilities.shape}"
        )

    scores = probabilities[0]
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Model output contains invalid probability values")

    total = float(scores.sum())
    if total <= 0:
        raise ValueError("Model output probabilities must have a positive sum")
    return scores / total
