from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ....evaluation.detector import _allowed_aspect_ratio, _metrics
from ....evaluation.onnx_detector import load_records
from ....pipeline.ports import Detection
from ....runtime.onnx import box_iou, nms


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select(
    prediction: dict[str, Any], threshold: float, nms_iou: float, aspect: float | None
) -> list[Detection]:
    return nms(
        [
            Detection(*box, score)
            for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
            if score >= threshold and _allowed_aspect_ratio(box, aspect)
        ],
        nms_iou,
    )


def ensemble_predictions(
    primary: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    *,
    primary_threshold: float,
    primary_singleton_threshold: float,
    recovery_threshold: float,
    agreement_iou: float,
    nms_iou: float = 0.7,
    max_object_aspect_ratio: float | None = 5.0,
) -> list[dict[str, Any]]:
    if len(primary) != len(recovery):
        raise ValueError("ensemble prediction image counts differ")
    results = []
    for primary_row, recovery_row in zip(primary, recovery):
        if primary_row["image_id"] != recovery_row["image_id"]:
            raise ValueError("ensemble prediction image ids differ")
        primary_detections = _select(
            primary_row, primary_threshold, nms_iou, max_object_aspect_ratio
        )
        recovery_detections = _select(
            recovery_row, recovery_threshold, nms_iou, max_object_aspect_ratio
        )
        retained = [
            detection
            for detection in primary_detections
            if detection.score >= primary_singleton_threshold
            or any(box_iou(detection, other) >= agreement_iou for other in recovery_detections)
        ]
        for detection in recovery_detections:
            if not any(box_iou(detection, other) >= agreement_iou for other in retained):
                retained.append(detection)
        retained = nms(retained, nms_iou)
        results.append(
            {
                "image_id": primary_row["image_id"],
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in retained],
                "scores": [item.score for item in retained],
            }
        )
    return results


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    multi_records = load_records(args.dataset_root, "multi_object_instances.json")
    scan_records = [
        record
        for record in load_records(args.dataset_root, "scan_log_instances.json")
        if record["expected_image_status"] == "ANNOTATED"
    ]
    records = multi_records + scan_records
    primary = _read_jsonl(args.primary_multi) + _read_jsonl(args.primary_scan)
    recovery = _read_jsonl(args.recovery_multi) + _read_jsonl(args.recovery_scan)
    candidates = []
    singleton_values = np.linspace(
        args.primary_threshold, args.maximum_primary_singleton_threshold, args.singleton_steps
    )
    recovery_values = np.linspace(
        args.minimum_recovery_threshold, args.maximum_recovery_threshold, args.recovery_steps
    )
    for singleton, recovery_threshold, agreement_iou in product(
        singleton_values, recovery_values, args.agreement_ious
    ):
        predictions = ensemble_predictions(
            primary,
            recovery,
            primary_threshold=args.primary_threshold,
            primary_singleton_threshold=float(singleton),
            recovery_threshold=float(recovery_threshold),
            agreement_iou=float(agreement_iou),
        )
        metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=0.7,
            match_iou_threshold=0.5,
            max_queries=600,
            max_object_aspect_ratio=5.0,
        )
        candidates.append(
            metrics
            | {
                "primary_singleton_threshold": float(singleton),
                "recovery_threshold": float(recovery_threshold),
                "agreement_iou": float(agreement_iou),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["recall"] >= args.target_recall]
    selected = max(
        eligible or candidates,
        key=lambda row: (row["precision"], row["recall"], row["count_accuracy"]),
    )
    report = {
        "evaluation": "detector_ensemble_diagnostic_only",
        "selection_set": "multi_development_plus_non_independent_scan_annotated",
        "candidate_count": len(candidates),
        "target_recall": args.target_recall,
        "target_recall_satisfied": selected["recall"] >= args.target_recall,
        "selected": selected,
        "passes_joint_segmentation_gate": (
            selected["recall"] >= 0.99 and selected["precision"] >= 0.99
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe D-FINE/RT-DETR box-level complementarity")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--primary-multi", type=Path, required=True)
    parser.add_argument("--primary-scan", type=Path, required=True)
    parser.add_argument("--recovery-multi", type=Path, required=True)
    parser.add_argument("--recovery-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-threshold", type=float, default=0.485)
    parser.add_argument("--maximum-primary-singleton-threshold", type=float, default=0.95)
    parser.add_argument("--singleton-steps", type=int, default=11)
    parser.add_argument("--minimum-recovery-threshold", type=float, default=0.10)
    parser.add_argument("--maximum-recovery-threshold", type=float, default=0.90)
    parser.add_argument("--recovery-steps", type=int, default=33)
    parser.add_argument("--agreement-ious", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--target-recall", type=float, default=0.99)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
