from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..training.data import read_manifest
from .detector import _metrics, _metrics_grid, detection_error_rows

_EXPECTED_RECAPTURE_STATUSES = {"RECAPTURE", "IMAGE_RECAPTURE"}


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
    fold = getattr(args, "fold", None)
    if fold is not None:
        records = [record for record in records if int(record["fold"]) == fold]
    expected_status = getattr(args, "expected_status", None)
    if expected_status is not None:
        records = [
            record for record in records if record.get("expected_image_status") == expected_status
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
    separate_recapture = getattr(args, "separate_recapture", False)
    if separate_recapture:
        annotated_pairs = [
            (record, prediction)
            for record, prediction in zip(records, ordered_predictions)
            if record.get("expected_image_status") not in _EXPECTED_RECAPTURE_STATUSES
        ]
        recapture_pairs = [
            (record, prediction)
            for record, prediction in zip(records, ordered_predictions)
            if record.get("expected_image_status") in _EXPECTED_RECAPTURE_STATUSES
        ]
    else:
        annotated_pairs = list(zip(records, ordered_predictions))
        recapture_pairs = []
    annotated_records = [record for record, _ in annotated_pairs]
    annotated_predictions = [prediction for _, prediction in annotated_pairs]
    fixed_threshold = getattr(args, "score_threshold", None)
    thresholds = (
        np.asarray([fixed_threshold], dtype=np.float64)
        if fixed_threshold is not None
        else np.linspace(args.min_score_threshold, args.max_score_threshold, args.threshold_steps)
    )
    candidates = _metrics_grid(
        annotated_records,
        annotated_predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_queries=args.max_queries,
        nms_containment_threshold=getattr(args, "nms_containment_threshold", None),
        nms_class_aware_containment=getattr(args, "nms_class_aware_containment", False),
    )
    strict_zero_errors = getattr(args, "strict_zero_errors", False)
    if strict_zero_errors:
        eligible = [
            candidate
            for candidate in candidates
            if candidate["false_positive_count"] == 0 and candidate["false_negative_count"] == 0
        ]
        selected = max(
            eligible if eligible else candidates,
            key=(
                (lambda item: (item["score_threshold"],))
                if eligible
                else (
                    lambda item: (
                        -item["false_positive_count"] - item["false_negative_count"],
                        item["exact_image_rate"],
                        -item["false_negative_count"],
                        -item["false_positive_count"],
                        item["score_threshold"],
                    )
                )
            ),
        )
    else:
        eligible = [
            candidate for candidate in candidates if candidate["recall"] >= args.target_recall
        ]
        selected = max(
            eligible if eligible else candidates,
            key=(
                (
                    lambda item: (
                        item["count_accuracy"],
                        item["precision"],
                        item["score_threshold"],
                    )
                )
                if eligible
                else (lambda item: (item["recall"], item["count_accuracy"], item["precision"]))
            ),
        )
    recapture_metrics = None
    if recapture_pairs:
        recapture_metrics = _metrics(
            [record for record, _ in recapture_pairs],
            [prediction for _, prediction in recapture_pairs],
            score_threshold=float(selected["score_threshold"]),
            nms_iou_threshold=args.nms_threshold,
            match_iou_threshold=args.match_iou_threshold,
            max_queries=args.max_queries,
            nms_containment_threshold=getattr(args, "nms_containment_threshold", None),
            nms_class_aware_containment=getattr(args, "nms_class_aware_containment", False),
        )
        recapture_metrics["detection_positive_image_count"] = (
            recapture_metrics["image_count"] - recapture_metrics["exact_image_count"]
        )
        recapture_metrics["detection_positive_image_rate"] = (
            recapture_metrics["detection_positive_image_count"] / recapture_metrics["image_count"]
            if recapture_metrics["image_count"]
            else 0.0
        )
    report = {
        "model_version": getattr(args, "model_version", "0.1.0"),
        "dataset_version": manifest_metadata["dataset_version"],
        "mode": "oof-development",
        "fold": fold,
        "checkpoint": "three-fold-oof",
        "prediction_files": [path.name for path in args.predictions],
        "match_iou_threshold": args.match_iou_threshold,
        "nms_iou_threshold": args.nms_threshold,
        "nms_containment_threshold": getattr(args, "nms_containment_threshold", None),
        "nms_class_aware_containment": getattr(args, "nms_class_aware_containment", False),
        "target_recall": args.target_recall,
        "strict_zero_errors": strict_zero_errors,
        "separate_recapture": separate_recapture,
        "threshold_policy": "fixed"
        if fixed_threshold is not None
        else "selected_on_oof-development",
        "selected_score_threshold": selected["score_threshold"],
        "target_recall_satisfied": selected["recall"] >= args.target_recall,
        "zero_false_positive_satisfied": selected["false_positive_count"] == 0,
        "zero_false_negative_satisfied": selected["false_negative_count"] == 0,
        "metrics": selected,
        "expected_recapture_raw_detection_metrics": recapture_metrics,
    }
    if getattr(args, "include_error_details", False):
        report["error_images"] = detection_error_rows(
            annotated_records,
            annotated_predictions,
            score_threshold=float(selected["score_threshold"]),
            nms_iou_threshold=args.nms_threshold,
            match_iou_threshold=args.match_iou_threshold,
            nms_containment_threshold=getattr(args, "nms_containment_threshold", None),
            nms_class_aware_containment=getattr(args, "nms_class_aware_containment", False),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one detector threshold from three OOF folds"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--expected-status")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--nms-containment-threshold", type=float)
    parser.add_argument("--nms-class-aware-containment", action="store_true")
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument(
        "--strict-zero-errors",
        action="store_true",
        help="Select a threshold only as a zero-FP/zero-FN candidate when possible.",
    )
    parser.add_argument(
        "--separate-recapture",
        action="store_true",
        help="Exclude expected IMAGE_RECAPTURE rows from raw annotated-object FP/FN.",
    )
    parser.add_argument("--include-error-details", action="store_true")
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
