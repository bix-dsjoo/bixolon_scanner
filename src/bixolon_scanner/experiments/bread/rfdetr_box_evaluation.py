from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .proposal_ranker import proposal_iou_matrix, select_ranked_predictions


def _load_prediction_superset(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        str(row["image_id"]): row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    }
    missing = {str(record["image_id"]) for record in records} - set(by_id)
    if missing:
        raise ValueError(f"RF-DETR predictions are missing {len(missing)} validation images")
    return [by_id[str(record["image_id"])] for record in records]


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )


def proposal_coverage_metrics(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, float | int]:
    """Measure maximum one-to-one stage-1 coverage, independent of score order."""
    errors = proposal_coverage_error_rows(records, predictions)
    false_negative_count = sum(row["false_negative_count"] for row in errors)
    proposal_count = sum(len(row["scores"]) for row in predictions)
    ground_truth_count = sum(len(row["annotations"]) for row in records)
    image_count = len(records)
    missed_images = len(errors)
    covered_images = image_count - missed_images
    matched_count = ground_truth_count - false_negative_count
    return {
        "all_ground_truth_covered_image_count": covered_images,
        "all_ground_truth_covered_image_rate": (
            covered_images / image_count if image_count else 0.0
        ),
        "missed_image_count": missed_images,
        "ground_truth_count": ground_truth_count,
        "matched_count": matched_count,
        "false_negative_count": false_negative_count,
        "recall": matched_count / ground_truth_count if ground_truth_count else 0.0,
        "proposal_count": proposal_count,
        "surplus_proposal_count": proposal_count - matched_count,
        "mean_proposals_per_image": proposal_count / image_count if image_count else 0.0,
    }


def _maximum_matched_ground_truth(ious: np.ndarray, minimum_iou: float) -> set[int]:
    proposal_to_ground_truth: dict[int, int] = {}

    def assign(ground_truth_index: int, visited: set[int]) -> bool:
        candidate_indices = np.flatnonzero(ious[:, ground_truth_index] >= minimum_iou)
        candidate_indices = candidate_indices[
            np.argsort(ious[candidate_indices, ground_truth_index])[::-1]
        ]
        for proposal_index_value in candidate_indices:
            proposal_index = int(proposal_index_value)
            if proposal_index in visited:
                continue
            visited.add(proposal_index)
            previous = proposal_to_ground_truth.get(proposal_index)
            if previous is None or assign(previous, visited):
                proposal_to_ground_truth[proposal_index] = ground_truth_index
                return True
        return False

    for ground_truth_index in range(ious.shape[1]):
        assign(ground_truth_index, set())
    return set(proposal_to_ground_truth.values())


def proposal_coverage_error_rows(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record, prediction in zip(records, predictions, strict=True):
        boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
        targets = np.asarray(
            [
                [x, y, x + width, y + height]
                for x, y, width, height in (
                    annotation["bbox_xywh"] for annotation in record["annotations"]
                )
            ],
            dtype=np.float32,
        ).reshape(-1, 4)
        matched = _maximum_matched_ground_truth(proposal_iou_matrix(boxes, targets), 0.5)
        unmatched = sorted(set(range(len(targets))) - matched)
        if unmatched:
            rows.append(
                {
                    "image_id": record.get("image_id"),
                    "image_path": record.get("image_path"),
                    "fold": record.get("fold"),
                    "difficulty": record.get("difficulty"),
                    "false_negative_count": len(unmatched),
                    "false_negatives": [
                        {
                            "annotation_id": record["annotations"][index].get("annotation_id"),
                            "category_id": record["annotations"][index].get("category_id"),
                            "bbox_xywh": record["annotations"][index]["bbox_xywh"],
                        }
                        for index in unmatched
                    ],
                }
            )
    return rows


def select_proposal_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer zero proposal misses first, then minimize stage-2 workload."""
    return max(
        candidates,
        key=lambda row: (
            -row["proposal_metrics"]["false_negative_count"],
            -row["proposal_metrics"]["missed_image_count"],
            -row["proposal_metrics"]["surplus_proposal_count"],
            row["score_threshold"],
        ),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) == args.fold
        and row.get("expected_image_status") == "ANNOTATED"
    ]
    predictions = _load_prediction_superset(args.predictions, records)
    candidates: list[dict[str, Any]] = []
    selected_cache: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for score_threshold, nms_threshold in product(args.score_thresholds, args.nms_thresholds):
        selected = select_ranked_predictions(
            predictions,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_threshold,
            nms_mode="class_agnostic",
        )
        selected_cache[(score_threshold, nms_threshold)] = selected
        candidates.append(
            {
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                "metrics": _metrics(
                    records,
                    selected,
                    score_threshold=0.0,
                    nms_iou_threshold=1.0,
                    match_iou_threshold=0.5,
                    max_queries=300,
                ),
                "proposal_metrics": proposal_coverage_metrics(records, selected),
            }
        )
    selected = select_candidate(candidates)
    maximum_recall = max(
        candidates,
        key=lambda row: (
            row["metrics"]["recall"],
            row["metrics"]["precision"],
            -row["metrics"]["false_positive_count"],
            row["score_threshold"],
        ),
    )
    full_recall = [row for row in candidates if row["metrics"]["false_negative_count"] == 0]
    full_proposal_recall = [
        row for row in candidates if row["proposal_metrics"]["false_negative_count"] == 0
    ]
    zero_false_positive = [row for row in candidates if row["metrics"]["false_positive_count"] == 0]
    selected_predictions = selected_cache[
        (selected["score_threshold"], selected["nms_iou_threshold"])
    ]
    stage1_proposal_best = select_proposal_candidate(candidates)
    stage1_predictions = selected_cache[
        (
            stage1_proposal_best["score_threshold"],
            stage1_proposal_best["nms_iou_threshold"],
        )
    ]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_rfdetr_box_only_fold_diagnostic",
        "fold": args.fold,
        "selection_scope": "single held-out development fold diagnostic only",
        "class_labels_used_for_matching": False,
        "match_iou_threshold": 0.5,
        "image_count": len(records),
        "ground_truth_count": sum(len(row["annotations"]) for row in records),
        "candidate_count": len(candidates),
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
        "stage1_proposal_best": stage1_proposal_best,
        "stage1_proposal_error_images": proposal_coverage_error_rows(records, stage1_predictions),
        "maximum_recall_best": maximum_recall,
        "full_recall_best": select_candidate(full_recall) if full_recall else None,
        "full_proposal_recall_best": (
            select_proposal_candidate(full_proposal_recall) if full_proposal_recall else None
        ),
        "zero_false_positive_best": (
            select_candidate(zero_false_positive) if zero_false_positive else None
        ),
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR box-only validation coverage")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
