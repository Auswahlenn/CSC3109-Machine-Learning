"""Evaluate a saved checkpoint on the held-out set without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import keras

from shared.data import get_held_out_dataset
from shared.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .keras file")
    parser.add_argument("--run-name", required=True, help="Output filename stem (e.g. vit-baseline)")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Import the model module so @register_keras_serializable decorators run
    # before load_model tries to deserialize the custom layer.
    import importlib.util
    models_dir = Path(__file__).parent.parent / "models"
    for model_file in models_dir.glob("*.py"):
        spec = importlib.util.spec_from_file_location(model_file.stem, model_file)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass

    print(f"Loading checkpoint: {checkpoint}")
    model = keras.models.load_model(checkpoint)

    print("Running held-out evaluation...")
    held_out_ds = get_held_out_dataset(batch_size=args.batch_size)
    metrics = evaluate(model, held_out_ds, model_name=args.run_name)

    print(f"\n=== Held-out results ({args.run_name}) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro']['f1']:.4f}")
    print(f"Saved to: results/{args.run_name}.json")
    print(f"          results/{args.run_name}_confusion_matrix.png")


if __name__ == "__main__":
    main()
