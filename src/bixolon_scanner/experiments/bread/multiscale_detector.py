from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import box_iou, nms
from ...training.data import read_manifest


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["image_id"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    }


def _selected_detections(
    row: dict[str, Any], threshold: float, nms_iou_threshold: float
) -> list[Detection]:
    return nms(
        [
            Detection(*box, score)
            for box, score in zip(row["boxes_xyxy"], row["scores"])
            if score >= threshold
        ],
        nms_iou_threshold,
    )


def consensus_filter_predictions(
    primary: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    *,
    primary_threshold: float,
    primary_singleton_threshold: float,
    recovery_threshold: float,
    agreement_iou_threshold: float,
    nms_iou_threshold: float,
) -> list[dict[str, Any]]:
    if len(primary) != len(recovery):
        raise ValueError("multi-scale prediction image counts differ")
    results = []
    for primary_row, recovery_row in zip(primary, recovery):
        if primary_row["image_id"] != recovery_row["image_id"]:
            raise ValueError("multi-scale prediction image ids differ")
        primary_detections = _selected_detections(primary_row, primary_threshold, nms_iou_threshold)
        recovery_detections = _selected_detections(
            recovery_row, recovery_threshold, nms_iou_threshold
        )
        retained = [
            detection
            for detection in primary_detections
            if detection.score >= primary_singleton_threshold
            or any(
                box_iou(detection, other) >= agreement_iou_threshold
                for other in recovery_detections
            )
        ]
        results.append(
            {
                "image_id": primary_row["image_id"],
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in retained],
                "scores": [item.score for item in retained],
            }
        )
    return results


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) == args.fold
        and row.get("expected_image_status") == args.expected_status
    ]
    primary_by_id = _read_predictions(args.primary_predictions)
    recovery_by_id = _read_predictions(args.recovery_predictions)
    expected = {str(row["image_id"]) for row in records}
    if set(primary_by_id) != expected or set(recovery_by_id) != expected:
        raise ValueError("multi-scale prediction coverage differs from selected manifest rows")
    primary = [primary_by_id[str(row["image_id"])] for row in records]
    recovery = [recovery_by_id[str(row["image_id"])] for row in records]
    singleton_values = np.linspace(
        args.primary_threshold,
        args.maximum_primary_singleton_threshold,
        args.singleton_steps,
    )
    recovery_values = np.linspace(
        args.minimum_recovery_threshold,
        args.maximum_recovery_threshold,
        args.recovery_steps,
    )
    candidates = []
    for singleton, recovery_threshold, agreement_iou in product(
        singleton_values, recovery_values, args.agreement_ious
    ):
        predictions = consensus_filter_predictions(
            primary,
            recovery,
            primary_threshold=args.primary_threshold,
            primary_singleton_threshold=float(singleton),
            recovery_threshold=float(recovery_threshold),
            agreement_iou_threshold=float(agreement_iou),
            nms_iou_threshold=args.nms_threshold,
        )
        metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=args.nms_threshold,
            match_iou_threshold=args.match_iou_threshold,
            max_queries=300,
        )
        candidates.append(
            {
                "primary_singleton_threshold": float(singleton),
                "recovery_threshold": float(recovery_threshold),
                "agreement_iou_threshold": float(agreement_iou),
                "metrics": metrics,
            }
        )
    eligible = [
        row
        for row in candidates
        if row["metrics"]["false_positive_count"] == 0
        and row["metrics"]["false_negative_count"] == 0
    ]
    selected = max(
        eligible or candidates,
        key=(
            (
                lambda row: (
                    row["primary_singleton_threshold"],
                    row["recovery_threshold"],
                    row["agreement_iou_threshold"],
                )
            )
            if eligible
            else (
                lambda row: (
                    -row["metrics"]["false_positive_count"]
                    - row["metrics"]["false_negative_count"],
                    row["metrics"]["exact_image_rate"],
                    -row["metrics"]["false_negative_count"],
                    -row["metrics"]["false_positive_count"],
                )
            )
        ),
    )
    selected_predictions = consensus_filter_predictions(
        primary,
        recovery,
        primary_threshold=args.primary_threshold,
        primary_singleton_threshold=selected["primary_singleton_threshold"],
        recovery_threshold=selected["recovery_threshold"],
        agreement_iou_threshold=selected["agreement_iou_threshold"],
        nms_iou_threshold=args.nms_threshold,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_multiscale_detector_consensus_filter",
        "fold": args.fold,
        "primary_predictions": args.primary_predictions.name,
        "recovery_predictions": args.recovery_predictions.name,
        "primary_threshold": args.primary_threshold,
        "nms_iou_threshold": args.nms_threshold,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(eligible),
        "selected": selected,
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=args.nms_threshold,
            match_iou_threshold=args.match_iou_threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in selected_predictions),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter primary detector boxes using multi-scale agreement"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--expected-status", default="ANNOTATED")
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--recovery-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--primary-threshold", type=float, required=True)
    parser.add_argument("--maximum-primary-singleton-threshold", type=float, default=0.5)
    parser.add_argument("--singleton-steps", type=int, default=20)
    parser.add_argument("--minimum-recovery-threshold", type=float, default=0.03)
    parser.add_argument("--maximum-recovery-threshold", type=float, default=0.3)
    parser.add_argument("--recovery-steps", type=int, default=15)
    parser.add_argument(
        "--agreement-ious", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7]
    )
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
