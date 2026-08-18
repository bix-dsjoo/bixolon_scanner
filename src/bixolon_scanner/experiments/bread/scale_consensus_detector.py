from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import box_iou
from ...training.data import read_manifest
from .hierarchical_detector import hierarchical_containment_nms


def scale_consensus_predictions(
    primary: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    *,
    agreement_iou_threshold: float,
    consensus_score_threshold: float,
    nms_iou_threshold: float,
    containment_threshold: float,
    group_minimum: int,
    class_match: str = "top1",
) -> list[dict[str, Any]]:
    candidates = scale_consensus_candidates(
        primary,
        recovery,
        agreement_iou_threshold=agreement_iou_threshold,
        class_match=class_match,
    )
    return select_scale_consensus_candidates(
        candidates,
        consensus_score_threshold=consensus_score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        containment_threshold=containment_threshold,
        group_minimum=group_minimum,
    )


def scale_consensus_candidates(
    primary: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    *,
    agreement_iou_threshold: float,
    class_match: str = "top1",
) -> list[dict[str, Any]]:
    """Pair cross-scale queries once, before thresholding and suppression."""
    if class_match not in {"top1", "top3", "none"}:
        raise ValueError("class_match must be top1, top3, or none")
    if len(primary) != len(recovery):
        raise ValueError("scale-consensus prediction image counts differ")
    outputs = []
    for primary_row, recovery_row in zip(primary, recovery):
        if primary_row["image_id"] != recovery_row["image_id"]:
            raise ValueError("scale-consensus prediction image ids differ")
        primary_detections = [
            Detection(*box, score, int(class_id))
            for box, score, class_id in zip(
                primary_row["boxes_xyxy"],
                primary_row["scores"],
                primary_row["class_ids"],
            )
        ]
        recovery_detections = [
            Detection(*box, score, int(class_id))
            for box, score, class_id in zip(
                recovery_row["boxes_xyxy"],
                recovery_row["scores"],
                recovery_row["class_ids"],
            )
        ]

        def classes_match(primary_index: int, recovery_index: int) -> bool:
            if class_match == "none":
                return True
            if class_match == "top1":
                return (
                    primary_detections[primary_index].class_id
                    == recovery_detections[recovery_index].class_id
                )
            primary_top3 = primary_row.get("top3_class_ids")
            recovery_top3 = recovery_row.get("top3_class_ids")
            if primary_top3 is None or recovery_top3 is None:
                raise ValueError("top3 class matching requires top3_class_ids")
            return bool(set(primary_top3[primary_index]) & set(recovery_top3[recovery_index]))

        pairs: set[tuple[int, int]] = set()
        for primary_index, candidate in enumerate(primary_detections):
            compatible = [
                (box_iou(candidate, other), index, other)
                for index, other in enumerate(recovery_detections)
                if classes_match(primary_index, index)
                and box_iou(candidate, other) >= agreement_iou_threshold
            ]
            if compatible:
                _, recovery_index, _ = max(compatible, key=lambda item: (item[0], item[2].score))
                pairs.add((primary_index, recovery_index))
        for recovery_index, candidate in enumerate(recovery_detections):
            compatible = [
                (box_iou(candidate, other), index, other)
                for index, other in enumerate(primary_detections)
                if classes_match(index, recovery_index)
                and box_iou(candidate, other) >= agreement_iou_threshold
            ]
            if compatible:
                _, primary_index, _ = max(compatible, key=lambda item: (item[0], item[2].score))
                pairs.add((primary_index, recovery_index))

        consensus = []
        for primary_index, recovery_index in pairs:
            primary_detection = primary_detections[primary_index]
            recovery_detection = recovery_detections[recovery_index]
            score = min(primary_detection.score, recovery_detection.score)
            consensus.append(
                Detection(
                    recovery_detection.x1,
                    recovery_detection.y1,
                    recovery_detection.x2,
                    recovery_detection.y2,
                    score,
                    recovery_detection.class_id,
                )
            )
        outputs.append(
            {
                "image_id": primary_row["image_id"],
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in consensus],
                "scores": [item.score for item in consensus],
                "class_ids": [item.class_id for item in consensus],
            }
        )
    return outputs


def select_scale_consensus_candidates(
    candidates: list[dict[str, Any]],
    *,
    consensus_score_threshold: float,
    nms_iou_threshold: float,
    containment_threshold: float,
    group_minimum: int,
) -> list[dict[str, Any]]:
    """Apply the cheap score gate before hierarchical suppression."""
    outputs = []
    for row in candidates:
        selected = hierarchical_containment_nms(
            [
                Detection(*box, score, int(class_id))
                for box, score, class_id in zip(row["boxes_xyxy"], row["scores"], row["class_ids"])
                if score >= consensus_score_threshold
            ],
            iou_threshold=nms_iou_threshold,
            containment_threshold=containment_threshold,
            group_minimum=group_minimum,
        )
        outputs.append(
            {
                "image_id": row["image_id"],
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in selected],
                "scores": [item.score for item in selected],
                "class_ids": [item.class_id for item in selected],
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
    folds = set(args.folds)
    difficulties = set(args.difficulties) if args.difficulties else None
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == args.expected_status
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]
    primary_by_id = _read_prediction_files(args.primary_predictions)
    recovery_by_id = _read_prediction_files(args.recovery_predictions)
    expected_ids = {str(row["image_id"]) for row in records}
    primary_ids = set(primary_by_id)
    recovery_ids = set(recovery_by_id)
    coverage_matches = (
        primary_ids == expected_ids and recovery_ids == expected_ids
        if difficulties is None
        else expected_ids.issubset(primary_ids) and expected_ids.issubset(recovery_ids)
    )
    if not coverage_matches:
        raise ValueError("scale-consensus coverage differs from selected manifest rows")
    primary = [primary_by_id[str(row["image_id"])] for row in records]
    recovery = [recovery_by_id[str(row["image_id"])] for row in records]

    candidates = []
    candidate_cache: dict[float, list[dict[str, Any]]] = {}
    prediction_cache: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for agreement_iou in args.agreement_iou_thresholds:
        candidate_cache[agreement_iou] = scale_consensus_candidates(
            primary,
            recovery,
            agreement_iou_threshold=agreement_iou,
            class_match=args.class_match,
        )
    for agreement_iou, consensus_score in product(
        args.agreement_iou_thresholds,
        args.consensus_score_thresholds,
    ):
        selected_predictions = select_scale_consensus_candidates(
            candidate_cache[agreement_iou],
            consensus_score_threshold=consensus_score,
            nms_iou_threshold=args.nms_threshold,
            containment_threshold=args.containment_threshold,
            group_minimum=args.group_minimum,
        )
        prediction_cache[(agreement_iou, consensus_score)] = selected_predictions
        metrics = _metrics(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=args.match_iou_threshold,
            max_queries=300,
        )
        candidates.append(
            {
                "agreement_iou_threshold": agreement_iou,
                "consensus_score_threshold": consensus_score,
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
            row["consensus_score_threshold"],
            row["agreement_iou_threshold"],
        ),
    )
    selected_predictions = prediction_cache[
        (selected["agreement_iou_threshold"], selected["consensus_score_threshold"])
    ]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_scale_consensus_detector",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "primary_prediction_files": [path.name for path in args.primary_predictions],
        "recovery_prediction_files": [path.name for path in args.recovery_predictions],
        "nms_iou_threshold": args.nms_threshold,
        "containment_threshold": args.containment_threshold,
        "group_minimum": args.group_minimum,
        "class_match": args.class_match,
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
        description="Select bread detections confirmed across two inference scales"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-status", default="ANNOTATED")
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--primary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--recovery-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--agreement-iou-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--consensus-score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--containment-threshold", type=float, default=0.8)
    parser.add_argument("--group-minimum", type=int, default=2)
    parser.add_argument("--class-match", choices=["top1", "top3", "none"], default="top1")
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
