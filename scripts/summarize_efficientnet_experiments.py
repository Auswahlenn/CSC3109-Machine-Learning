"""Create deterministic JSON and Markdown summaries of tuning experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RUNS = (
    "efficientnet_b0_baseline",
    "efficientnet_b0_lr3e4",
    "efficientnet_b0_drop04",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    experiments: list[dict[str, Any]] = []
    for run_name in RUNS:
        metadata = load_json(RESULTS_DIR / f"{run_name}_run.json")
        metrics = load_json(RESULTS_DIR / f"{run_name}_tuning.json")
        experiments.append(
            {
                "run_name": run_name,
                "dropout": metadata["model_config"]["dropout"],
                "learning_rate": metadata["model_config"]["learning_rate"],
                "epochs_completed": metadata["epochs_completed"],
                "best_epoch": metadata["best_epoch"],
                "duration_seconds": metadata["duration_seconds"],
                "tuning_accuracy": metrics["accuracy"],
                "tuning_macro_f1": metrics["macro"]["f1"],
                "per_class_f1": {
                    name: class_metrics["f1"]
                    for name, class_metrics in metrics["per_class"].items()
                },
                "run_metadata": f"{run_name}_run.json",
                "history": f"{run_name}_history.json",
                "tuning_metrics": f"{run_name}_tuning.json",
                "confusion_matrix": f"{run_name}_tuning_confusion_matrix.png",
            }
        )

    ranking = sorted(
        experiments,
        key=lambda experiment: experiment["tuning_macro_f1"],
        reverse=True,
    )
    summary = {
        "selection_split": "internal_tuning",
        "selection_metric": "macro_f1",
        "held_out_used_for_selection": False,
        "selected_run": ranking[0]["run_name"],
        "experiments": ranking,
    }
    json_path = RESULTS_DIR / "efficientnet_b0_experiment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown_lines = [
        "# EfficientNet-B0 controlled experiment summary",
        "",
        "Selection used internal tuning macro F1 only. The held-out split was not used.",
        "",
        "| Rank | Run | Dropout | Learning rate | Best epoch | Tuning accuracy | Tuning macro F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, experiment in enumerate(ranking, start=1):
        markdown_lines.append(
            "| {rank} | `{run}` | {dropout:.1f} | {learning_rate:g} | "
            "{best_epoch} | {accuracy:.4f} | {macro_f1:.4f} |".format(
                rank=rank,
                run=experiment["run_name"],
                dropout=experiment["dropout"],
                learning_rate=experiment["learning_rate"],
                best_epoch=experiment["best_epoch"],
                accuracy=experiment["tuning_accuracy"],
                macro_f1=experiment["tuning_macro_f1"],
            )
        )
    markdown_lines.extend(
        [
            "",
            f"Selected run: `{ranking[0]['run_name']}`.",
            "",
            "[Inference] The lower learning rate underfit within the shared epoch budget. Increasing "
            "dropout to 0.4 remained competitive but did not exceed the 0.3 baseline.",
        ]
    )
    markdown_path = RESULTS_DIR / "efficientnet_b0_experiment_summary.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
