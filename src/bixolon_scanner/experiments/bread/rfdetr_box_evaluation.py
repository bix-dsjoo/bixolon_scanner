from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .proposal_ranker import select_ranked_predictions


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
    zero_false_positive = [row for row in candidates if row["metrics"]["false_positive_count"] == 0]
    selected_predictions = selected_cache[
        (selected["score_threshold"], selected["nms_iou_threshold"])
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
        "maximum_recall_best": maximum_recall,
        "full_recall_best": select_candidate(full_recall) if full_recall else None,
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
