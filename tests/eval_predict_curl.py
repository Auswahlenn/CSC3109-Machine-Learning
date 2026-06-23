#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(text: str) -> str:
    """
    Convert labels / filenames into comparable form.
    Example:
      dense_residential -> denseresidential
      denseresidential761.jpg -> denseresidential
    """
    text = text.lower()
    text = Path(text).stem
    text = "".join(ch for ch in text if ch.isalpha() or ch == "_")
    return text.replace("_", "")


def get_expected_label(image_path: Path, root: Path, known_labels: list[str]) -> str | None:
    """
    Priority:
    1. Use first folder under root:
       dataset/val 23/dense_residential/image.jpg -> dense_residential

    2. Fallback to filename:
       denseresidential761.jpg -> dense_residential
    """
    rel_parts = image_path.relative_to(root).parts

    if len(rel_parts) >= 2:
        folder_label = rel_parts[0]
        if folder_label in known_labels:
            return folder_label

    filename_norm = normalize_label(image_path.name)

    for label in known_labels:
        if filename_norm.startswith(normalize_label(label)):
            return label

    return None


def predict_with_curl(endpoint: str, image_path: Path, timeout: int) -> dict:
    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        endpoint,
        "-F",
        f"file=@{image_path}",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="dataset/val 23",
        help="Root folder containing validation images",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/predict",
        help="Prediction API endpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold. Must be greater than this value to pass.",
    )
    parser.add_argument(
        "--output",
        default="prediction_results.csv",
        help="CSV output file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Curl timeout per image in seconds",
    )
    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        print(f"ERROR: root folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    known_labels = sorted([
        p.name for p in root.iterdir()
        if p.is_dir()
    ])

    if not known_labels:
        print(f"ERROR: no label folders found inside: {root}", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])

    if not image_paths:
        print(f"ERROR: no images found inside: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    total = 0
    passed = 0
    failed = 0
    errors = 0

    for image_path in image_paths:
        total += 1

        expected_label = get_expected_label(image_path, root, known_labels)

        row = {
            "file": str(image_path),
            "expected_label": expected_label or "",
            "predicted_label": "",
            "confidence": "",
            "status": "FAIL",
            "error": "",
        }

        if expected_label is None:
            row["error"] = "Could not determine expected label"
            failed += 1
            rows.append(row)
            print(f"[FAIL] {image_path} -> expected label unknown")
            continue

        try:
            response = predict_with_curl(args.endpoint, image_path, args.timeout)

            predicted_label = response.get("predicted_label")
            confidence = float(response.get("confidence", 0.0))

            is_pass = (
                predicted_label == expected_label
                and confidence > args.threshold
            )

            row["predicted_label"] = predicted_label
            row["confidence"] = confidence
            row["status"] = "PASS" if is_pass else "FAIL"

            if is_pass:
                passed += 1
                print(
                    f"[PASS] {image_path} -> "
                    f"expected={expected_label}, predicted={predicted_label}, "
                    f"confidence={confidence:.4f}"
                )
            else:
                failed += 1
                print(
                    f"[FAIL] {image_path} -> "
                    f"expected={expected_label}, predicted={predicted_label}, "
                    f"confidence={confidence:.4f}"
                )

        except Exception as e:
            errors += 1
            failed += 1
            row["error"] = str(e)
            print(f"[ERROR] {image_path} -> {e}")

        rows.append(row)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "expected_label",
                "predicted_label",
                "confidence",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    accuracy = passed / total * 100 if total else 0

    print()
    print("Summary")
    print("-------")
    print(f"Total images : {total}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")
    print(f"Errors       : {errors}")
    print(f"Accuracy     : {accuracy:.2f}%")
    print(f"CSV saved to : {args.output}")


if __name__ == "__main__":
    main()
