"""Shared leakage-free training and evaluation entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from shared import config
from shared.augment import get_augmentation
from shared.data import get_held_out_dataset, get_split_counts, get_training_datasets
from shared.evaluate import evaluate


def load_model_module(name: str) -> tuple[str, ModuleType]:
    """Find and import one uniquely named model module under ``models/``."""
    stem = Path(name).stem
    models_dir = Path(__file__).resolve().parent / "models"
    matches = [
        path
        for path in models_dir.rglob(f"{stem}.py")
        if path.name != "__init__.py"
    ]
    if not matches:
        raise FileNotFoundError(f"No model file '{stem}.py' found under models/.")
    if len(matches) > 1:
        locations = ", ".join(
            str(path.relative_to(models_dir.parent)) for path in matches
        )
        raise ValueError(f"Multiple model files named '{stem}.py' found: {locations}")

    spec = importlib.util.spec_from_file_location(stem, matches[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load model module: {matches[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return stem, module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free shared training entrypoint."
    )
    parser.add_argument("--model", required=True, help="Model filename stem.")
    parser.add_argument("--run-name", help="Unique artifact stem.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config.BATCH_SIZE for memory-limited GPUs (does not affect "
        "comparability; only batching changes).",
    )
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--evaluate-held-out",
        action="store_true",
        help="Evaluate held-out data after training; use only for smoke/final runs.",
    )
    parser.add_argument(
        "--run-type",
        choices=("smoke", "experiment", "final"),
        default="experiment",
    )
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe_history(
    history: keras.callbacks.History,
) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _validate_run_name(run_name: str) -> str:
    if not run_name or not all(
        character.isalnum() or character in "_-" for character in run_name
    ):
        raise ValueError(
            "run-name may contain only letters, numbers, underscores, and hyphens"
        )
    return run_name


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.patience < 0:
        raise ValueError("patience must be non-negative")

    config.set_seed()
    train_ds, tuning_ds = get_training_datasets(batch_size=args.batch_size)
    augmentation = get_augmentation()

    model_name, model_module = load_model_module(args.model)
    run_name = _validate_run_name(args.run_name or model_name)

    factors = {
        key: value
        for key, value in {
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
        }.items()
        if value is not None
    }
    configure = getattr(model_module, "configure", None)
    if factors and configure is None:
        raise AttributeError(f"{model_name}.py does not support configurable factors")
    if configure is not None:
        configure(**factors)
    if not hasattr(model_module, "build_model"):
        raise AttributeError(
            f"{model_name}.py must expose build_model(num_classes, augmentation)"
        )

    model: keras.Model = model_module.build_model(
        config.NUM_CLASSES, augmentation
    )
    model.summary()

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / f"{run_name}_best.keras"
    # Optional per-model callbacks (e.g. an LR scheduler justified by that
    # member's own research). Each model file may define get_callbacks(); models
    # that don't are unaffected. The shared list below stays model-agnostic.
    extra_callbacks = (
        model_module.get_callbacks() if hasattr(model_module, "get_callbacks") else []
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        *extra_callbacks,
    ]

    started = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=tuning_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    duration_seconds = time.perf_counter() - started

    best_model = keras.models.load_model(checkpoint_path)
    history_payload = _json_safe_history(history)
    _save_json(results_dir / f"{run_name}_history.json", history_payload)

    tuning_metrics = evaluate(
        best_model, tuning_ds, model_name=f"{run_name}_tuning"
    )
    held_out_metrics = None
    if args.evaluate_held_out:
        held_out_metrics = evaluate(
            best_model, get_held_out_dataset(batch_size=args.batch_size), model_name=run_name
        )

    active_config = getattr(
        model_module, "get_experiment_config", lambda: {}
    )()
    best_epoch = int(np.argmax(history_payload["val_accuracy"]) + 1)
    metadata: dict[str, Any] = {
        "run_name": run_name,
        "run_type": args.run_type,
        "model_name": model_name,
        "git_commit": _git_commit(),
        "seed": config.SEED,
        "image_size": config.IMAGE_SIZE,
        # Record the batch size actually used, not the shared default, so runs
        # with --batch-size stay verifiable.
        "batch_size": args.batch_size or config.BATCH_SIZE,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.epoch),
        "patience": args.patience,
        "best_epoch": best_epoch,
        "duration_seconds": float(duration_seconds),
        "model_config": active_config,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(
            sum(np.prod(weight.shape) for weight in model.trainable_weights)
        ),
        "split_counts": get_split_counts(),
        "tensorflow_version": tf.__version__,
        "gpus": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
        "checkpoint": str(checkpoint_path),
        "tuning_metrics_file": f"{run_name}_tuning.json",
        "held_out_metrics_file": (
            f"{run_name}.json" if held_out_metrics else None
        ),
    }
    _save_json(results_dir / f"{run_name}_run.json", metadata)

    print("\n=== Run summary ===")
    print(f"Run:            {run_name}")
    print(f"Best epoch:     {best_epoch}")
    print(f"Tuning F1:      {tuning_metrics['macro']['f1']:.4f}")
    if held_out_metrics:
        print(f"Held-out F1:    {held_out_metrics['macro']['f1']:.4f}")
    print(f"Checkpoint:     {checkpoint_path}")


if __name__ == "__main__":
    main()
