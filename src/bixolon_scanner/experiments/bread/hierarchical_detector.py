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


def _area(box: Detection) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


def _intersection(left: Detection, right: Detection) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return width * height


def hierarchical_containment_nms(
    detections: list[Detection],
    *,
    iou_threshold: float,
    containment_threshold: float,
    group_minimum: int,
) -> list[Detection]:
    """Suppress fragments and group boxes without deleting a valid nested object.

    A lower-scoring candidate is removed when ordinary NMS removes it, when it is
    mostly inside a stronger box, when a same-class stronger box is mostly inside
    it, or when it encloses at least ``group_minimum`` stronger boxes. A
    different-class outer candidate enclosing only one stronger box is retained.
    """
    if group_minimum < 2:
        raise ValueError("group_minimum must be at least 2")
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    for index, candidate in enumerate(ordered):
        candidate_area = _area(candidate)
        if candidate_area <= 0.0:
            continue
        stronger = ordered[:index]
        stronger_inside = [
            other
            for other in stronger
            if _area(other) > 0.0
            and _intersection(candidate, other) / _area(other) >= containment_threshold
        ]
        suppressed = False
        for current in kept:
            current_area = _area(current)
            intersection = _intersection(current, candidate)
            candidate_inside = intersection / candidate_area >= containment_threshold
            current_inside = (
                current_area > 0.0 and intersection / current_area >= containment_threshold
            )
            same_class = current.class_id is not None and current.class_id == candidate.class_id
            if (
                box_iou(current, candidate) > iou_threshold
                or candidate_inside
                or current_inside
                and (same_class or len(stronger_inside) >= group_minimum)
            ):
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept


def filter_predictions(
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    iou_threshold: float,
    containment_threshold: float,
    group_minimum: int,
) -> list[dict[str, Any]]:
    filtered = []
    for row in predictions:
        detections = hierarchical_containment_nms(
            [
                Detection(*box, score, int(class_id))
                for box, score, class_id in zip(row["boxes_xyxy"], row["scores"], row["class_ids"])
                if score >= score_threshold
            ],
            iou_threshold=iou_threshold,
            containment_threshold=containment_threshold,
            group_minimum=group_minimum,
        )
        filtered.append(
            {
                "image_id": row["image_id"],
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in detections],
                "scores": [item.score for item in detections],
                "class_ids": [item.class_id for item in detections],
            }
        )
    return filtered


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    selected_folds = set(args.folds)
    difficulties = set(args.difficulties) if args.difficulties else None
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in selected_folds
        and row.get("expected_image_status") == args.expected_status
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]
    predictions_by_id = {
        str(row["image_id"]): row
        for row in (
            json.loads(line)
            for path in args.predictions
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    expected_ids = {str(row["image_id"]) for row in records}
    prediction_ids = set(predictions_by_id)
    coverage_matches = (
        prediction_ids == expected_ids
        if difficulties is None
        else expected_ids.issubset(prediction_ids)
    )
    if not coverage_matches:
        raise ValueError("prediction coverage differs from selected manifest rows")
    predictions = [predictions_by_id[str(row["image_id"])] for row in records]

    candidates = []
    for score, containment, group_minimum in product(
        args.score_thresholds,
        args.containment_thresholds,
        args.group_minimums,
    ):
        filtered = filter_predictions(
            predictions,
            score_threshold=score,
            iou_threshold=args.nms_threshold,
            containment_threshold=containment,
            group_minimum=group_minimum,
        )
        metrics = _metrics(
            records,
            filtered,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=args.match_iou_threshold,
            max_queries=300,
        )
        candidates.append(
            {
                "score_threshold": score,
                "containment_threshold": containment,
                "group_minimum": group_minimum,
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
            row["score_threshold"],
        ),
    )
    selected_predictions = filter_predictions(
        predictions,
        score_threshold=selected["score_threshold"],
        iou_threshold=args.nms_threshold,
        containment_threshold=selected["containment_threshold"],
        group_minimum=selected["group_minimum"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_hierarchical_detector",
        "folds": sorted(selected_folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "prediction_files": [path.name for path in args.predictions],
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
        description="Evaluate hierarchical containment suppression for bread detection"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-status", default="ANNOTATED")
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument(
        "--containment-thresholds", type=float, nargs="+", default=[0.85, 0.9, 0.95]
    )
    parser.add_argument("--group-minimums", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
