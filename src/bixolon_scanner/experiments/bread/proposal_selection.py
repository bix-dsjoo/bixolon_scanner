from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions
from .proposal_ranker import select_ranked_predictions


def evaluate(args: argparse.Namespace) -> dict:
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
    ranked = _load_predictions(args.ranked_predictions, records, allow_superset=True)
    candidates = []
    for (
        score,
        nms_threshold,
        nms_mode,
        center_threshold,
        containment_threshold,
        group_minimum,
    ) in product(
        args.score_thresholds,
        args.nms_thresholds,
        args.nms_modes,
        args.center_distance_thresholds,
        args.containment_thresholds,
        args.group_minimums,
    ):
        predictions = select_ranked_predictions(
            ranked,
            score_threshold=score,
            nms_iou_threshold=nms_threshold,
            nms_mode=nms_mode,
            nms_center_distance_threshold=center_threshold,
            containment_threshold=containment_threshold,
            group_minimum=group_minimum,
        )
        metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        candidates.append(
            {
                "score_threshold": score,
                "nms_iou_threshold": nms_threshold,
                "nms_mode": nms_mode,
                "center_distance_threshold": center_threshold,
                "containment_threshold": containment_threshold,
                "group_minimum": group_minimum,
                "metrics": metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )
    selected_predictions = select_ranked_predictions(
        ranked,
        score_threshold=selected["score_threshold"],
        nms_iou_threshold=selected["nms_iou_threshold"],
        nms_mode=selected["nms_mode"],
        nms_center_distance_threshold=selected["center_distance_threshold"],
        containment_threshold=selected["containment_threshold"],
        group_minimum=selected["group_minimum"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_ranked_proposal_selection",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
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
    parser = argparse.ArgumentParser(description="Select postprocessing for OOF ranked proposals")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument(
        "--nms-modes",
        choices=["class_agnostic", "class_aware", "center_aware"],
        nargs="+",
        required=True,
    )
    parser.add_argument("--center-distance-thresholds", type=float, nargs="+", default=[0.3])
    parser.add_argument("--containment-thresholds", type=float, nargs="+", default=[0.8])
    parser.add_argument("--group-minimums", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
