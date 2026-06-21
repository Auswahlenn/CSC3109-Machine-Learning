"""FROZEN -- do not edit without notifying the team; changes invalidate prior results.

Shared evaluation.

Every member calls :func:`evaluate` with the SAME signature and gets back a dict
in the SAME schema, plus a JSON file and a confusion-matrix PNG written to
``results/``. This fixed output schema is the contract that makes all five
members' numbers directly comparable -- do not change keys, types, or the
metric definitions (macro averaging, zero_division policy) without telling the
team, as it invalidates every previously saved result.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import tensorflow as tf  # only for the tf.data.Dataset type hint
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import config

# Bumped only when the schema itself changes, so downstream tooling can detect
# results produced by an incompatible version of this file.
SCHEMA_VERSION: str = "1.0"


def _collect_labels_and_preds(
    model: keras.Model, val_ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return aligned integer ``(y_true, y_pred)`` arrays.

    Relies on ``val_ds`` being UNSHUFFLED (see shared/data.py) so that the order
    of ``model.predict`` outputs matches the order of the iterated labels.
    """
    # One-hot -> integer class index, in dataset order.
    y_true = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in val_ds], axis=0
    )
    probs = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    return y_true, y_pred


def evaluate(
    model: keras.Model, val_ds: tf.data.Dataset, model_name: str
) -> dict[str, Any]:
    """Evaluate a trained model on the validation set and persist results.

    Computes overall accuracy, macro-averaged precision/recall/F1, per-class
    precision/recall/F1, and the confusion matrix. Writes
    ``results/<model_name>.json`` and ``results/<model_name>_confusion_matrix.png``.

    Args:
        model: A trained ``keras.Model`` that outputs class probabilities of
            shape ``(batch, NUM_CLASSES)``.
        val_ds: The unshuffled validation dataset from
            :func:`shared.data.get_datasets` (raw pixels; the model is expected
            to apply its own ``preprocess_input`` internally).
        model_name: Identifier for this member's model, used both as the dict
            ``"model_name"`` field and as the output filename stem.

    Returns:
        A dict with the FIXED schema::

            {
              "schema_version": str,
              "model_name": str,
              "num_classes": int,
              "class_names": list[str],
              "accuracy": float,
              "macro": {"precision": float, "recall": float, "f1": float},
              "per_class": {
                  <class_name>: {
                      "precision": float, "recall": float,
                      "f1": float, "support": int
                  },
                  ...
              },
              "confusion_matrix": list[list[int]],  # rows=true, cols=pred
            }
    """
    class_names = config.CLASS_NAMES
    labels = list(range(config.NUM_CLASSES))

    y_true, y_pred = _collect_labels_and_preds(model, val_ds)

    accuracy = float(accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(per_support[i]),
        }

    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "num_classes": config.NUM_CLASSES,
        "class_names": class_names,
        "accuracy": accuracy,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    _save_json(results, model_name)
    _save_confusion_matrix_png(cm, class_names, model_name)

    return results


def _save_json(results: dict[str, Any], model_name: str) -> None:
    """Write ``results/<model_name>.json``."""
    path = os.path.join(config.RESULTS_DIR, f"{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _save_confusion_matrix_png(
    cm: np.ndarray, class_names: list[str], model_name: str
) -> None:
    """Write ``results/<model_name>_confusion_matrix.png``."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- {model_name}")

    # Annotate each cell; use a contrasting text color on dark cells.
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
