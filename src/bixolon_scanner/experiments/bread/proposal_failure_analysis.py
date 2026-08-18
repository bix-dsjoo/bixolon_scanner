from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _xywh_to_xyxy
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions
from .proposal_ranker import _iou_matrix


def candidate_structure_rows(
    record: dict[str, Any],
    prediction: dict[str, Any],
    *,
    minimum_score: float,
    limit: int = 30,
) -> list[dict[str, Any]]:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    class_ids = np.asarray(prediction["class_ids"], dtype=np.int64)
    source_ids = np.asarray(prediction.get("source_ids", np.zeros(len(boxes))), dtype=np.int64)
    targets = np.asarray(
        [_xywh_to_xyxy(row["bbox_xywh"]) for row in record["annotations"]],
        dtype=np.float32,
    )
    target_iou = _iou_matrix(boxes, targets)
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    top_left = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=2)
    containment = np.divide(
        intersection,
        areas[None, :],
        out=np.zeros_like(intersection),
        where=areas[None, :] > 0.0,
    )
    rows = []
    for index in np.flatnonzero(scores >= minimum_score):
        smaller = areas < areas[index] * 0.8
        inner = smaller & (containment[index] >= 0.5)
        inner[index] = False
        best_target = int(np.argmax(target_iou[index]))
        inner_indices = np.flatnonzero(inner)
        rows.append(
            {
                "candidate_index": int(index),
                "score": float(scores[index]),
                "box_xyxy": boxes[index].tolist(),
                "class_id": int(class_ids[index]),
                "source_id": int(source_ids[index]),
                "best_target_annotation_id": int(
                    record["annotations"][best_target]["annotation_id"]
                ),
                "best_target_category_id": int(record["annotations"][best_target]["category_id"]),
                "best_target_iou": float(target_iou[index, best_target]),
                "inner_candidate_count_at_50": int(inner.sum()),
                "inner_candidate_count_at_70": int((smaller & (containment[index] >= 0.7)).sum()),
                "inner_candidate_count_at_90": int((smaller & (containment[index] >= 0.9)).sum()),
                "inner_source_ids": sorted(set(source_ids[inner_indices].tolist())),
                "inner_class_ids": sorted(set(class_ids[inner_indices].tolist())),
                "inner_max_score": (
                    float(scores[inner_indices].max()) if len(inner_indices) else None
                ),
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:limit]


def analyze_error_rows(
    records: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    error_images: list[dict[str, Any]],
    *,
    score_threshold: float,
) -> dict[str, Any]:
    records_by_id = {str(row["image_id"]): row for row in records}
    ranked_by_id = {str(row["image_id"]): row for row in ranked}
    false_negatives = []
    false_positives = []
    candidate_structures = {}
    for error in error_images:
        image_id = str(error["image_id"])
        record = records_by_id[image_id]
        prediction = ranked_by_id[image_id]
        boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
        scores = np.asarray(prediction["scores"], dtype=np.float32)
        targets = np.asarray(
            [_xywh_to_xyxy(row["bbox_xywh"]) for row in record["annotations"]],
            dtype=np.float32,
        )
        candidate_structures[image_id] = candidate_structure_rows(
            record,
            prediction,
            minimum_score=score_threshold,
        )
        for target in error["false_negatives"]:
            target_box = np.asarray([_xywh_to_xyxy(target["bbox_xywh"])])
            ious = _iou_matrix(boxes, target_box)[:, 0]
            positive = ious >= 0.5
            best_positive_score = float(scores[positive].max()) if positive.any() else None
            best_index = int(np.argmax(ious))
            false_negatives.append(
                {
                    "image_id": error["image_id"],
                    "fold": error["fold"],
                    "difficulty": error["difficulty"],
                    "annotation_id": target["annotation_id"],
                    "category_id": target["category_id"],
                    "best_candidate_iou": float(ious[best_index]),
                    "best_candidate_score": float(scores[best_index]),
                    "best_positive_score": best_positive_score,
                    "positive_candidate_count": int(positive.sum()),
                    "positive_above_threshold_count": int(
                        (positive & (scores >= score_threshold)).sum()
                    ),
                    "failure_stage": (
                        "score_rejected"
                        if best_positive_score is None or best_positive_score < score_threshold
                        else "nms_or_assignment"
                    ),
                }
            )
        for prediction_row in error["false_positives"]:
            box = np.asarray([prediction_row["bbox_xyxy"]], dtype=np.float32)
            max_iou = float(_iou_matrix(box, targets).max()) if len(targets) else 0.0
            false_positives.append(
                {
                    "image_id": error["image_id"],
                    "fold": error["fold"],
                    "difficulty": error["difficulty"],
                    "score": prediction_row["score"],
                    "best_target_iou": max_iou,
                    "localization_near_miss": 0.3 <= max_iou < 0.5,
                }
            )
    stage_counts = {
        stage: sum(row["failure_stage"] == stage for row in false_negatives)
        for stage in ("score_rejected", "nms_or_assignment")
    }
    return {
        "score_threshold": score_threshold,
        "false_negative_stage_counts": stage_counts,
        "false_positive_localization_near_miss_count": sum(
            row["localization_near_miss"] for row in false_positives
        ),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "candidate_structures": candidate_structures,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
    ]
    ranked = _load_predictions(args.ranked_predictions, records)
    selection_report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    if args.selection_key is not None:
        selection_report = selection_report[args.selection_key]
    selected = selection_report["selected"]
    analysis = analyze_error_rows(
        records,
        ranked,
        selection_report["error_images"],
        score_threshold=float(selected["score_threshold"]),
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_proposal_failure_analysis",
        "folds": sorted(folds),
        "selection": selected,
        **analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OOF proposal selection failures")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--selection-key")
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
