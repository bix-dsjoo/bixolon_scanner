from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import read_manifest
from .evaluate_detector import _metrics


def _read_predictions(paths: list[Path]) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prediction = json.loads(line)
            image_id = str(prediction["image_id"])
            if image_id in predictions:
                raise ValueError(f"duplicate image_id in OOF predictions: {image_id}")
            predictions[image_id] = prediction
    return predictions


def aggregate(args: argparse.Namespace) -> None:
    manifest_metadata = json.loads(
        (args.manifest.parent / "metadata.json").read_text(encoding="utf-8")
    )
    records = [
        record
        for record in read_manifest(args.manifest)
        if record["record_type"] == "detection"
        and record["split"] == "development"
        and not record.get("exclude_from_detector_training", False)
    ]
    by_image_id = _read_predictions(args.predictions)
    expected = {str(record["image_id"]) for record in records}
    missing = sorted(expected - set(by_image_id))
    unexpected = sorted(set(by_image_id) - expected)
    if missing or unexpected:
        raise ValueError(
            f"OOF prediction coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    ordered_predictions = [by_image_id[str(record["image_id"])] for record in records]
    fixed_threshold = getattr(args, "score_threshold", None)
    thresholds = (
        np.asarray([fixed_threshold], dtype=np.float64)
        if fixed_threshold is not None
        else np.linspace(args.min_score_threshold, args.max_score_threshold, args.threshold_steps)
    )
    candidates = []
    for threshold in thresholds:
        metrics = _metrics(
            records,
            ordered_predictions,
            score_threshold=float(threshold),
            nms_iou_threshold=args.nms_threshold,
            match_iou_threshold=args.match_iou_threshold,
            max_queries=args.max_queries,
        )
        metrics["score_threshold"] = float(threshold)
        candidates.append(metrics)
    eligible = [candidate for candidate in candidates if candidate["recall"] >= args.target_recall]
    selected = max(
        eligible if eligible else candidates,
        key=(
            (lambda item: (item["count_accuracy"], item["precision"], item["score_threshold"]))
            if eligible
            else (lambda item: (item["recall"], item["count_accuracy"], item["precision"]))
        ),
    )
    report = {
        "model_version": getattr(args, "model_version", "0.1.0"),
        "dataset_version": manifest_metadata["dataset_version"],
        "mode": "oof-development",
        "fold": None,
        "checkpoint": "three-fold-oof",
        "prediction_files": [path.name for path in args.predictions],
        "match_iou_threshold": args.match_iou_threshold,
        "nms_iou_threshold": args.nms_threshold,
        "target_recall": args.target_recall,
        "threshold_policy": "fixed"
        if fixed_threshold is not None
        else "selected_on_oof-development",
        "selected_score_threshold": selected["score_threshold"],
        "target_recall_satisfied": selected["recall"] >= args.target_recall,
        "metrics": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one detector threshold from three OOF folds"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Evaluate one pre-committed threshold instead of selecting a new threshold.",
    )
    parser.add_argument("--min-score-threshold", type=float, default=0.05)
    parser.add_argument("--max-score-threshold", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument("--max-queries", type=int, default=300)
    aggregate(parser.parse_args())


if __name__ == "__main__":
    main()
