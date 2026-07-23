"""Plot training curves from a saved train.py history file.

train.py writes results/<run-name>_history.json every run. This reads that file
and draws the loss + accuracy curves, then saves a PNG next to it.

Usage:
    python plot_history.py --run-name custom_cnn
    python plot_history.py --run-name custom_cnn --no-show   # save only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from shared import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train.py history curves.")
    parser.add_argument(
        "--run-name", required=True, help="Run name stem, e.g. custom_cnn"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save the PNG without opening a window"
    )
    args = parser.parse_args()

    results_dir = Path(config.RESULTS_DIR)
    history_path = results_dir / f"{args.run_name}_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"No history file: {history_path}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["loss"], label="train_loss")
    ax_loss.plot(epochs, history["val_loss"], label="val_loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, history["accuracy"], label="train_accuracy")
    ax_acc.plot(epochs, history["val_accuracy"], label="val_accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()

    # Mark the best (checkpointed) epoch so the graph shows what train.py kept.
    best_epoch = max(epochs, key=lambda epoch: history["val_accuracy"][epoch - 1])
    for axis in (ax_loss, ax_acc):
        axis.axvline(best_epoch, color="grey", linestyle="--", alpha=0.6)
    ax_acc.set_title(f"Accuracy (best epoch = {best_epoch})")

    fig.suptitle(args.run_name)
    fig.tight_layout()

    out_path = results_dir / f"{args.run_name}_curves.png"
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
