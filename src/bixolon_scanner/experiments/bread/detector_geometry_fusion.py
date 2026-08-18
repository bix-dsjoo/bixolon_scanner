from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import box_iou, nms
from ...training.data import read_manifest


def replace_with_recovery_geometry(
    primary: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    *,
    recovery_score_threshold: float,
    agreement_iou_threshold: float,
    recovery_nms_threshold: float,
    require_same_class: bool,
) -> list[dict[str, Any]]:
    if len(primary) != len(recovery):
        raise ValueError("geometry-fusion prediction image counts differ")
    outputs = []
    for primary_row, recovery_row in zip(primary, recovery):
        if primary_row["image_id"] != recovery_row["image_id"]:
            raise ValueError("geometry-fusion prediction image ids differ")
        primary_detections = [
            Detection(*box, score, int(class_id))
            for box, score, class_id in zip(
                primary_row["boxes_xyxy"],
                primary_row["scores"],
                primary_row.get("class_ids", [None] * len(primary_row["scores"])),
            )
        ]
        recovery_detections = nms(
            [
                Detection(*box, score, int(class_id))
                for box, score, class_id in zip(
                    recovery_row["boxes_xyxy"],
                    recovery_row["scores"],
                    recovery_row.get("class_ids", [None] * len(recovery_row["scores"])),
                )
                if score >= recovery_score_threshold
            ],
            recovery_nms_threshold,
        )
        used_recovery: set[int] = set()
        replacements: dict[int, Detection] = {}
        for primary_index in sorted(
            range(len(primary_detections)),
            key=lambda index: primary_detections[index].score,
            reverse=True,
        ):
            candidate = primary_detections[primary_index]
            compatible = [
                (index, other)
                for index, other in enumerate(recovery_detections)
                if index not in used_recovery
                and box_iou(candidate, other) >= agreement_iou_threshold
                and (
                    not require_same_class
                    or candidate.class_id is not None
                    and candidate.class_id == other.class_id
                )
            ]
            if compatible:
                recovery_index, replacement = max(compatible, key=lambda item: item[1].score)
                used_recovery.add(recovery_index)
                replacements[primary_index] = replacement
        fused = [
            replacements.get(index, detection) for index, detection in enumerate(primary_detections)
        ]
        outputs.append(
            {
                "image_id": primary_row["image_id"],
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in fused],
                "scores": [primary_detections[index].score for index in range(len(fused))],
                "class_ids": [item.class_id for item in fused],
            }
        )
    return outputs


def _read_prediction_files(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(row["image_id"]): row
        for row in (
            json.loads(line)
            for path in paths
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    selected_folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in selected_folds
        and row.get("expected_image_status") == args.expected_status
    ]
    primary_by_id = _read_prediction_files(args.primary_predictions)
    recovery_by_id = _read_prediction_files(args.recovery_predictions)
    expected_ids = {str(row["image_id"]) for row in records}
    if set(primary_by_id) != expected_ids or set(recovery_by_id) != expected_ids:
        raise ValueError("geometry-fusion coverage differs from selected manifest rows")
    primary = [primary_by_id[str(row["image_id"])] for row in records]
    recovery = [recovery_by_id[str(row["image_id"])] for row in records]

    candidates = []
    same_class_values = (
        args.require_same_class
        if isinstance(args.require_same_class, list)
        else [args.require_same_class]
    )
    for recovery_threshold, agreement_iou, same_class in product(
        args.recovery_score_thresholds,
        args.agreement_iou_thresholds,
        same_class_values,
    ):
        predictions = replace_with_recovery_geometry(
            primary,
            recovery,
            recovery_score_threshold=recovery_threshold,
            agreement_iou_threshold=agreement_iou,
            recovery_nms_threshold=args.recovery_nms_threshold,
            require_same_class=same_class,
        )
        metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=args.match_iou_threshold,
            max_queries=300,
        )
        candidates.append(
            {
                "recovery_score_threshold": recovery_threshold,
                "agreement_iou_threshold": agreement_iou,
                "require_same_class": same_class,
                "metrics": metrics,
            }
        )
    zero_error = [
        row
        for row in candidates
        if row["metrics"]["false_positive_count"] == 0
        and row["metrics"]["false_negative_count"] == 0
    ]
    selected = max(
        zero_error or candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["agreement_iou_threshold"],
            row["recovery_score_threshold"],
        ),
    )
    selected_predictions = replace_with_recovery_geometry(
        primary,
        recovery,
        recovery_score_threshold=selected["recovery_score_threshold"],
        agreement_iou_threshold=selected["agreement_iou_threshold"],
        recovery_nms_threshold=args.recovery_nms_threshold,
        require_same_class=selected["require_same_class"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_geometry_fusion",
        "folds": sorted(selected_folds),
        "primary_prediction_files": [path.name for path in args.primary_predictions],
        "recovery_prediction_files": [path.name for path in args.recovery_predictions],
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(zero_error),
        "selected": selected,
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
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
        description="Replace primary detector box geometry using recovery-scale agreement"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-status", default="ANNOTATED")
    parser.add_argument("--primary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--recovery-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--recovery-score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--agreement-iou-thresholds", type=float, nargs="+", required=True)
    parser.add_argument(
        "--require-same-class",
        action=argparse.BooleanOptionalAction,
        default=[True, False],
    )
    parser.add_argument("--recovery-nms-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
