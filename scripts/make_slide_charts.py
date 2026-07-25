"""Generate the two summary charts used in the video slides.

  slide7_pretrained_gap.png  - macro-F1 by model, pretrained vs from-scratch
  slide8_params_vs_score.png - accuracy against parameter count (log x)

Both read their numbers live from results/*.json so they cannot drift from the
report. Usage:  python scripts/make_slide_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

MODELS = [
    ("efficientnet_b0", "EfficientNet-B0", True),
    ("resnet50", "ResNet-50", True),
    ("ViT", "ViT-B/16", True),
    ("mobilenetv2", "MobileNetV2", True),
    ("custom_cnn", "Custom CNN", False),
]

TRANSFER = "#2E6DB4"
SCRATCH = "#C4441C"


def load():
    rows = []
    for run, label, pretrained in MODELS:
        held = json.loads((RESULTS / f"{run}.json").read_text(encoding="utf-8"))
        meta = json.loads((RESULTS / f"{run}_run.json").read_text(encoding="utf-8"))
        rows.append(dict(
            label=label,
            pretrained=pretrained,
            f1=held["macro"]["f1"],
            acc=held["accuracy"],
            params=meta["total_parameters"],
        ))
    return rows


def slide7(rows) -> None:
    rows = sorted(rows, key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    colours = [TRANSFER if r["pretrained"] else SCRATCH for r in rows]
    bars = ax.barh([r["label"] for r in rows], [r["f1"] for r in rows],
                   color=colours, height=0.62)

    for bar, row in zip(bars, rows):
        ax.text(row["f1"] + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{row['f1']:.4f}   ({row['params']/1e6:.1f}M params)",
                va="center", fontsize=13)

    worst_transfer = min(r["f1"] for r in rows if r["pretrained"])
    scratch = next(r["f1"] for r in rows if not r["pretrained"])
    ax.axvline(scratch, color=SCRATCH, ls=":", lw=1.4, alpha=0.7)
    ax.axvline(worst_transfer, color=TRANSFER, ls=":", lw=1.4, alpha=0.7)
    # Bars are sorted ascending, so index 0 is the from-scratch model and index 1
    # the weakest pretrained one. Put the gap marker in the space between them.
    ax.annotate("", xy=(scratch, 0.5), xytext=(worst_transfer, 0.5),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.8))
    ax.annotate(
        f"{(worst_transfer - scratch)*100:.2f} points",
        xy=((scratch + worst_transfer) / 2, 0.5), ha="center", va="center",
        fontsize=15, color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999"),
    )

    ax.set_xlim(0.80, 1.03)
    ax.set_xlabel("Held-out macro-$F_1$", fontsize=14)
    ax.set_title("Pretrained backbones vs. training from scratch",
                 fontsize=19, pad=14)
    ax.tick_params(labelsize=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=TRANSFER),
               plt.Rectangle((0, 0), 1, 1, color=SCRATCH)]
    ax.legend(handles, ["Pretrained (frozen backbone)", "Trained from scratch"],
              fontsize=13, loc="lower right", frameon=False)

    fig.tight_layout()
    out = FIGURES / "slide7_pretrained_gap.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Saved {out}")


def slide8(rows) -> None:
    """Accuracy vs parameter count, with the Pareto frontier marked.

    Frontier = models not dominated on both axes (fewer parameters AND higher
    accuracy). Used in the video and in the report's Accuracy versus Cost section.
    """
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Pareto-optimal: no other model has both fewer params and higher accuracy.
    frontier = [r for r in rows
                if not any(o["params"] < r["params"] and o["acc"] > r["acc"]
                           for o in rows)]
    frontier.sort(key=lambda r: r["params"])
    ax.plot([r["params"] for r in frontier], [r["acc"] * 100 for r in frontier],
            ls="--", lw=1.8, color="#666666", zorder=2,
            label="Pareto frontier (accuracy vs. size)")

    for row in rows:
        colour = TRANSFER if row["pretrained"] else SCRATCH
        on_front = row in frontier
        ax.scatter(row["params"], row["acc"] * 100, s=260, color=colour, zorder=3,
                   edgecolors="black" if on_front else "none",
                   linewidths=1.8 if on_front else 0)
        ax.annotate(row["label"], (row["params"], row["acc"] * 100),
                    textcoords="offset points", xytext=(0, 18),
                    ha="center", fontsize=13)

    ax.set_xscale("log")
    ax.set_xlabel("Total parameters (log scale)", fontsize=14)
    ax.set_ylabel("Held-out accuracy (%)", fontsize=14)
    ax.set_title("Held-out accuracy versus model size", fontsize=19, pad=14)
    ax.grid(alpha=0.25, zorder=0)
    ax.tick_params(labelsize=13)
    ax.set_ylim(82, 100)
    ax.legend(fontsize=12, loc="lower left", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    out = FIGURES / "slide8_params_vs_score.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Saved {out}  (frontier: {', '.join(r['label'] for r in frontier)})")


def slide9() -> None:
    """Per-class F1 heatmap: classes ordered worst-to-best average, models by rank."""
    import numpy as np

    order = ["efficientnet_b0", "resnet50", "ViT", "mobilenetv2", "custom_cnn"]
    labels = ["EfficientNet-B0", "ResNet-50", "ViT-B/16", "MobileNetV2", "Custom CNN"]
    per_class = {}
    for run in order:
        held = json.loads((RESULTS / f"{run}.json").read_text(encoding="utf-8"))
        per_class[run] = {c: v["f1"] for c, v in held["per_class"].items()}

    classes = sorted(
        next(iter(per_class.values())).keys(),
        key=lambda c: -sum(per_class[r][c] for r in order),
    )
    matrix = np.array([[per_class[r][c] for r in order] for c in classes])

    fig, ax = plt.subplots(figsize=(13, 5.6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.70, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(order)), labels, fontsize=14)
    ax.set_yticks(range(len(classes)),
                  [c.replace("_", " ") for c in classes], fontsize=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=15, color="black")

    ax.set_title("Per-class $F_1$: the density classes are hardest for every model",
                 fontsize=18, pad=14)
    fig.colorbar(im, ax=ax, label="Held-out $F_1$", shrink=0.85)
    fig.tight_layout()
    out = FIGURES / "slide9_per_class_f1.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load()
    slide7(data)
    slide8(data)
    slide9()
