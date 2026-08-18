from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .proposal_count_selector import count_constrained_select


def ambiguity_recapture_mask(
    available: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    minimum_selected_count: int,
    extra_candidate_count: int,
    extra_count_mode: str,
    next_score_threshold: float,
) -> np.ndarray:
    if extra_count_mode not in {"exact", "at_least"}:
        raise ValueError("extra count mode must be exact or at_least")
    decisions = []
    for available_row, selected_row in zip(available, selected):
        selected_count = len(selected_row["scores"])
        extra_count = len(available_row["scores"]) - selected_count
        next_score = float(available_row["scores"][selected_count]) if extra_count > 0 else -1.0
        extra_matches = (
            extra_count == extra_candidate_count
            if extra_count_mode == "exact"
            else extra_count >= extra_candidate_count
        )
        decisions.append(
            selected_count >= minimum_selected_count
            and extra_matches
            and next_score >= next_score_threshold
        )
    return np.asarray(decisions, dtype=bool)


def _read_predictions(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {int(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate prediction image ids in {path}")
    return by_id


def _image_error_counts(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[int, int]:
    fp_images = 0
    fn_images = 0
    for record, prediction in zip(records, predictions):
        metrics = _metrics(
            [record],
            [prediction],
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        fp_images += metrics["false_positive_count"] > 0
        fn_images += metrics["false_negative_count"] > 0
    return int(fp_images), int(fn_images)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    difficulties = set(args.difficulties) if args.difficulties else None
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]
    raw_by_id = _read_predictions(args.raw_predictions)
    selected_by_id = _read_predictions(args.selected_predictions)
    expected_ids = {int(row["image_id"]) for row in records}
    if not expected_ids.issubset(raw_by_id) or not expected_ids.issubset(selected_by_id):
        raise ValueError("ambiguity gate predictions do not cover the selected scope")
    raw = [raw_by_id[int(record["image_id"])] for record in records]
    selected = [selected_by_id[int(record["image_id"])] for record in records]
    baseline_report = json.loads(args.baseline_recapture_report.read_text(encoding="utf-8"))
    baseline_ids = {
        int(image_id) for image_id in baseline_report["selected"]["recaptured_image_ids"]
    }
    if not baseline_ids <= expected_ids:
        raise ValueError("baseline recapture report contains images outside the selected scope")
    baseline_mask = np.asarray(
        [int(record["image_id"]) in baseline_ids for record in records], dtype=bool
    )
    available = [
        count_constrained_select(
            prediction,
            predicted_count=600,
            score_threshold=args.availability_score_threshold,
            nms_iou_threshold=args.availability_nms_threshold,
            containment_threshold=args.availability_containment_threshold,
            group_minimum=args.availability_group_minimum,
        )
        for prediction in raw
    ]
    next_scores = args.next_score_thresholds or sorted(
        {
            float(available_row["scores"][len(selected_row["scores"])])
            for available_row, selected_row in zip(available, selected)
            if len(available_row["scores"]) > len(selected_row["scores"])
        }
    )
    candidates = []
    masks: dict[tuple[int, int, str, float], np.ndarray] = {}
    for minimum_count, extra_count, mode, next_score in product(
        args.minimum_selected_counts,
        args.extra_candidate_counts,
        args.extra_count_modes,
        next_scores,
    ):
        ambiguity = ambiguity_recapture_mask(
            available,
            selected,
            minimum_selected_count=minimum_count,
            extra_candidate_count=extra_count,
            extra_count_mode=mode,
            next_score_threshold=next_score,
        )
        recaptured = baseline_mask | ambiguity
        accepted_count = int((~recaptured).sum())
        segmentation_rate = accepted_count / len(records) if records else 0.0
        if segmentation_rate < args.minimum_segmentation_rate:
            continue
        accepted_records = [row for row, flag in zip(records, recaptured) if not flag]
        accepted_predictions = [row for row, flag in zip(selected, recaptured) if not flag]
        metrics = _metrics(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        fp_images, fn_images = _image_error_counts(accepted_records, accepted_predictions)
        key = (minimum_count, extra_count, mode, next_score)
        masks[key] = recaptured
        candidates.append(
            {
                "minimum_selected_count": minimum_count,
                "extra_candidate_count": extra_count,
                "extra_count_mode": mode,
                "next_score_threshold_inclusive": next_score,
                "segmentation_image_count": accepted_count,
                "image_recapture_count": int(recaptured.sum()),
                "segmentation_rate": segmentation_rate,
                "false_positive_image_count": fp_images,
                "false_negative_image_count": fn_images,
                "metrics": metrics,
            }
        )
    if not candidates:
        raise ValueError("no ambiguity candidates satisfy the segmentation-rate floor")
    zero_error = [
        row
        for row in candidates
        if row["false_positive_image_count"] == 0 and row["false_negative_image_count"] == 0
    ]
    selected_candidate = max(
        zero_error or candidates,
        key=lambda row: (
            -(row["false_positive_image_count"] + row["false_negative_image_count"]),
            row["segmentation_rate"],
            row["next_score_threshold_inclusive"],
        ),
    )
    key = (
        selected_candidate["minimum_selected_count"],
        selected_candidate["extra_candidate_count"],
        selected_candidate["extra_count_mode"],
        selected_candidate["next_score_threshold_inclusive"],
    )
    recaptured = masks[key]
    accepted_records = [row for row, flag in zip(records, recaptured) if not flag]
    accepted_predictions = [row for row, flag in zip(selected, recaptured) if not flag]
    recaptured_ids = [int(row["image_id"]) for row, flag in zip(records, recaptured) if flag]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_postprocessing_ambiguity_gate",
        "selection_scope": "prediction-only proposal availability after fixed postprocessing",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "image_count": len(records),
        "baseline_recapture_count": int(baseline_mask.sum()),
        "availability_policy": {
            "score_threshold": args.availability_score_threshold,
            "nms_iou_threshold": args.availability_nms_threshold,
            "containment_threshold": args.availability_containment_threshold,
            "group_minimum": args.availability_group_minimum,
        },
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(zero_error),
        "selected": {
            **selected_candidate,
            "recaptured_image_ids": recaptured_ids,
        },
        "error_images": detection_error_rows(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a proposal-availability ambiguity gate after detector postprocessing"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--selected-predictions", type=Path, required=True)
    parser.add_argument("--baseline-recapture-report", type=Path, required=True)
    parser.add_argument("--availability-score-threshold", type=float, required=True)
    parser.add_argument("--availability-nms-threshold", type=float, default=0.5)
    parser.add_argument("--availability-containment-threshold", type=float, default=0.9)
    parser.add_argument("--availability-group-minimum", type=int, default=2)
    parser.add_argument("--minimum-selected-counts", type=int, nargs="+", default=range(1, 8))
    parser.add_argument("--extra-candidate-counts", type=int, nargs="+", default=range(1, 6))
    parser.add_argument("--extra-count-modes", nargs="+", default=("exact", "at_least"))
    parser.add_argument("--next-score-thresholds", type=float, nargs="+")
    parser.add_argument("--minimum-segmentation-rate", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
