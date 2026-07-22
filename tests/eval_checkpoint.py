"""Evaluate a saved classifier checkpoint against labeled image folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.root}")

    dataset = keras.utils.image_dataset_from_directory(
        args.root,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = dataset.class_names
    model = keras.models.load_model(args.checkpoint, compile=False)

    probabilities = model.predict(dataset, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.concatenate(
        [np.argmax(batch_labels.numpy(), axis=1) for _, batch_labels in dataset]
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.root),
        "images": int(labels.size),
        "class_names": class_names,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
