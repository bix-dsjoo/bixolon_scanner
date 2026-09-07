from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.evaluation.onnx_detector import load_records
from bixolon_scanner.runtime.onnx import OrtRunner, prepare_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ONNX exact-count verifier")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", default="multi_object_instances.json")
    parser.add_argument("--annotation-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    contract = load_json_config(args.training_report)
    records = load_records(
        args.dataset_root,
        args.annotation,
        annotation_path=args.annotation_path,
    )
    runner = OrtRunner(args.model, args.provider)
    image_size = int(contract["image_size"])
    classes = np.asarray(contract["count_labels"], dtype=np.int64)
    temperature = float(contract.get("temperature", 1.0))
    rows = []
    for record in records:
        with Image.open(record["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            tensor = prepare_rgb(
                image,
                (image_size, image_size),
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                reducing_gap=1.0,
            )[None]
        (logits,) = runner.run(["logits"], "pixel_values", tensor)
        values = np.asarray(logits[0], dtype=np.float64) / temperature
        values -= values.max()
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        expected = len(record["annotations"])
        rows.append(
            {
                "image_id": int(record["image_id"]),
                "expected_count": expected,
                "predicted_count": int(classes[index]),
                "confidence": float(probabilities[index]),
                "correct": int(classes[index]) == expected,
            }
        )
    confidence_rows = []
    for threshold in np.linspace(0.5, 0.99, 50):
        trusted = [row for row in rows if row["confidence"] >= threshold]
        correct = sum(row["correct"] for row in trusted)
        confidence_rows.append(
            {
                "confidence_threshold": float(threshold),
                "trusted_count": len(trusted),
                "trusted_coverage": len(trusted) / len(rows),
                "trusted_correct_count": correct,
                "trusted_error_count": len(trusted) - correct,
                "trusted_accuracy": correct / len(trusted) if trusted else None,
            }
        )
    exact = sum(row["correct"] for row in rows)
    report = {
        "evaluation": "exact_count_verifier",
        "model": args.model.name,
        "provider": args.provider,
        "image_count": len(rows),
        "exact_correct_count": exact,
        "exact_accuracy": exact / len(rows),
        "confidence_selection": confidence_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
