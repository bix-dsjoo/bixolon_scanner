from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest
from .detector_disagreement_gate import prediction_rows_agree


def policy_agreement_count(
    primary: dict[str, Any],
    policies: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> int:
    """Count policies with one-to-one count and geometry agreement with ``primary``."""
    return sum(prediction_rows_agree(primary, row, iou_threshold=iou_threshold) for row in policies)


def _read_predictions(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {int(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate image ids in {path}")
    return by_id


def _image_error_counts(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[int, int]:
    false_positive_images = 0
    false_negative_images = 0
    for record, prediction in zip(records, predictions):
        metrics = _metrics(
            [record],
            [prediction],
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        false_positive_images += metrics["false_positive_count"] > 0
        false_negative_images += metrics["false_negative_count"] > 0
    return int(false_positive_images), int(false_negative_images)


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
    policy_maps = [_read_predictions(path) for path in args.policy_predictions]
    expected_ids = {int(record["image_id"]) for record in records}
    if any(not expected_ids.issubset(policy) for policy in policy_maps):
        raise ValueError("policy predictions do not cover the selected scope")
    policies = [[policy[int(record["image_id"])] for record in records] for policy in policy_maps]
    policy_count = len(policies)
    minimum_agreements = args.minimum_agreeing_policies or list(range(1, policy_count + 1))
    if any(value < 1 or value > policy_count for value in minimum_agreements):
        raise ValueError("minimum agreeing policy counts must be within the policy count")

    candidates = []
    cache: dict[
        tuple[int, float, int],
        tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]],
    ] = {}
    for primary_index, threshold, minimum_agreement in product(
        range(policy_count), args.agreement_iou_thresholds, minimum_agreements
    ):
        accepted_records = []
        accepted_predictions = []
        recaptured_ids = []
        for row_index, record in enumerate(records):
            primary = policies[primary_index][row_index]
            rows = [policy[row_index] for policy in policies]
            agreement_count = policy_agreement_count(
                primary,
                rows,
                iou_threshold=float(threshold),
            )
            if agreement_count >= minimum_agreement:
                accepted_records.append(record)
                accepted_predictions.append(primary)
            else:
                recaptured_ids.append(int(record["image_id"]))
        segmentation_rate = len(accepted_records) / len(records) if records else 0.0
        if segmentation_rate < args.minimum_segmentation_rate:
            continue
        metrics = _metrics(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        fp_images, fn_images = _image_error_counts(accepted_records, accepted_predictions)
        key = (primary_index, float(threshold), minimum_agreement)
        cache[key] = (accepted_records, accepted_predictions, recaptured_ids)
        candidates.append(
            {
                "primary_policy_index": primary_index,
                "agreement_iou_threshold": float(threshold),
                "minimum_agreeing_policy_count": minimum_agreement,
                "segmentation_image_count": len(accepted_records),
                "image_recapture_count": len(recaptured_ids),
                "segmentation_rate": segmentation_rate,
                "false_positive_image_count": fp_images,
                "false_negative_image_count": fn_images,
                "metrics": metrics,
            }
        )
    if not candidates:
        raise ValueError("no policy-consensus candidate satisfies the segmentation-rate floor")
    zero_error = [
        row
        for row in candidates
        if row["false_positive_image_count"] == 0 and row["false_negative_image_count"] == 0
    ]
    selected = max(
        zero_error or candidates,
        key=lambda row: (
            -(row["false_positive_image_count"] + row["false_negative_image_count"]),
            row["segmentation_rate"],
            row["minimum_agreeing_policy_count"],
            row["agreement_iou_threshold"],
        ),
    )
    key = (
        selected["primary_policy_index"],
        selected["agreement_iou_threshold"],
        selected["minimum_agreeing_policy_count"],
    )
    accepted_records, accepted_predictions, recaptured_ids = cache[key]
    recaptured = set(recaptured_ids)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_fixed_policy_consensus_gate",
        "selection_scope": "prediction-only one-to-one count and geometry agreement",
        "selection_status": "development_validation_only_not_locked_test",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "image_count": len(records),
        "policy_predictions": [str(path) for path in args.policy_predictions],
        "policy_count": policy_count,
        "minimum_segmentation_rate": args.minimum_segmentation_rate,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(zero_error),
        "selected": {
            **selected,
            "primary_predictions": str(args.policy_predictions[selected["primary_policy_index"]]),
            "recaptured_image_ids": recaptured_ids,
            "recaptured_by_difficulty": dict(
                sorted(
                    Counter(
                        str(record["difficulty"])
                        for record in records
                        if int(record["image_id"]) in recaptured
                    ).items()
                )
            ),
        },
        "error_images": detection_error_rows(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in accepted_predictions),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed multi-policy detector consensus IMAGE_RECAPTURE gate"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--policy-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--agreement-iou-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--minimum-agreeing-policies", type=int, nargs="+")
    parser.add_argument("--minimum-segmentation-rate", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
