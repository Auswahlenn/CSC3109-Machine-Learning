"""Create a checksum-backed manifest for the selected deployable model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CHECKPOINT = RESULTS_DIR / "efficientnet_b0_best.keras"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run = load_json(RESULTS_DIR / "efficientnet_b0_run.json")
    metrics = load_json(RESULTS_DIR / "efficientnet_b0.json")
    manifest = {
        "model_name": "efficientnet_b0",
        "checkpoint": CHECKPOINT.name,
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "class_names": metrics["class_names"],
        "image_size": run["image_size"],
        "channels": 3,
        "input_range": [0, 255],
        "model_config": run["model_config"],
        "training_git_commit": run["git_commit"],
        "held_out_accuracy": metrics["accuracy"],
        "held_out_macro_f1": metrics["macro"]["f1"],
        "metrics_file": "efficientnet_b0.json",
        "run_metadata_file": "efficientnet_b0_run.json",
    }
    output = RESULTS_DIR / "efficientnet_b0_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
