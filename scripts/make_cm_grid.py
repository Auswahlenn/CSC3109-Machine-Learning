"""Combine the five held-out confusion matrices into one 3-over-2 slide image.

Models are ordered by held-out macro-F1 so the visual degradation reads
left-to-right, top-to-bottom. Writes docs/figures/all_confusion_matrices.png.

Usage:
    python scripts/make_cm_grid.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs" / "figures"
RESULTS = ROOT / "results"

# (run name, display label)
MODELS = [
    ("efficientnet_b0", "EfficientNet-B0"),
    ("resnet50", "ResNet-50"),
    ("ViT", "ViT-B/16"),
    ("mobilenetv2", "MobileNetV2"),
    ("custom_cnn", "Custom CNN (from scratch)"),
]


def macro_f1(run_name: str) -> float | None:
    path = RESULTS / f"{run_name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["macro"]["f1"]


def main() -> None:
    panels = []
    for run_name, label in MODELS:
        image_path = FIGURES / f"{run_name}_confusion_matrix.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing figure: {image_path}")
        score = macro_f1(run_name)
        caption = label if score is None else f"{label}  (macro-$F_1$ = {score:.4f})"
        panels.append((mpimg.imread(image_path), caption))

    fig = plt.figure(figsize=(18, 11))
    # Six columns so the bottom row of two can be centred under the top row of three.
    grid = fig.add_gridspec(2, 6, hspace=0.12, wspace=0.05)
    slots = [
        grid[0, 0:2], grid[0, 2:4], grid[0, 4:6],
        grid[1, 1:3], grid[1, 3:5],
    ]

    for (image, caption), slot in zip(panels, slots):
        axis = fig.add_subplot(slot)
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(caption, fontsize=15, pad=8)

    fig.suptitle(
        "Held-out confusion matrices (400 images, 100 per class)",
        fontsize=20,
        y=0.97,
    )

    out_path = FIGURES / "all_confusion_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
