"""Shared training entrypoint.

Selects a team member's model via ``--model`` and trains it with the SAME data,
augmentation, callbacks, and evaluation as everyone else. Only the model file
differs between members.

Each model module in ``models/`` must expose::

    def build_model(num_classes: int, augmentation: keras.Sequential) -> keras.Model

returning a COMPILED ``keras.Model`` that internally stacks, in order:
    1. the shared ``augmentation`` layers (passed in),
    2. the backbone's own ``preprocess_input``,
    3. the pretrained backbone,
    4. a new classification head ending in a softmax over ``num_classes``.

Usage:
    python train.py --model example_resnet50
    python train.py --model example_resnet50 --epochs 30 --patience 5
"""

from __future__ import annotations

import argparse
import importlib
import os

from tensorflow import keras

from shared import config
from shared.augment import get_augmentation
from shared.data import get_datasets
from shared.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Shared training entrypoint.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model module name in models/ (e.g. 'example_resnet50'), "
        "without the .py extension.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs (EarlyStopping may stop sooner).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="EarlyStopping patience (epochs without val_accuracy improvement).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility first, before anything touches an RNG.
    config.set_seed()

    # Shared data + augmentation (identical for every member).
    train_ds, val_ds = get_datasets()
    augmentation = get_augmentation()

    # Dynamically load the selected member's model module and build the model.
    model_module = importlib.import_module(f"models.{args.model}")
    if not hasattr(model_module, "build_model"):
        raise AttributeError(
            f"models/{args.model}.py must expose build_model(num_classes, augmentation)."
        )
    model: keras.Model = model_module.build_model(
        num_classes=config.NUM_CLASSES, augmentation=augmentation
    )
    model.summary()

    # Best model (by val_accuracy) is checkpointed so evaluation runs on the
    # best epoch, not the last one.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.RESULTS_DIR, f"{args.model}_best.keras")
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
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Reload the best checkpoint to guarantee evaluation uses the best epoch
    # even if restore_best_weights behavior changes.
    best_model = keras.models.load_model(checkpoint_path)

    results = evaluate(best_model, val_ds, model_name=args.model)
    print("\n=== Validation results ===")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Macro F1:    {results['macro']['f1']:.4f}")
    print(f"Saved JSON:  {os.path.join(config.RESULTS_DIR, args.model + '.json')}")


if __name__ == "__main__":
    main()
