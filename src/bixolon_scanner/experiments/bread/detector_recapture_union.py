from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ...evaluation.detector import _metrics, detection_error_rows
from ...training.data import read_manifest


def union_recapture_ids(reports: list[dict[str, Any]]) -> set[int]:
    output: set[int] = set()
    for report in reports:
        selected = report.get("selected")
        if not isinstance(selected, dict) or "recaptured_image_ids" not in selected:
            raise ValueError("recapture report is missing selected.recaptured_image_ids")
        output.update(int(value) for value in selected["recaptured_image_ids"])
    return output


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


def _scope_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    recaptured_ids: set[int],
) -> dict[str, Any]:
    accepted_records = [row for row in records if int(row["image_id"]) not in recaptured_ids]
    accepted_predictions = [
        prediction
        for row, prediction in zip(records, predictions)
        if int(row["image_id"]) not in recaptured_ids
    ]
    metrics = _metrics(
        accepted_records,
        accepted_predictions,
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=0.5,
        max_queries=600,
    )
    fp_images, fn_images = _image_error_counts(accepted_records, accepted_predictions)
    image_count = len(records)
    segmentation_count = len(accepted_records)
    return {
        "image_count": image_count,
        "segmentation_image_count": segmentation_count,
        "image_recapture_count": image_count - segmentation_count,
        "segmentation_rate": segmentation_count / image_count if image_count else 0.0,
        "false_positive_image_count": fp_images,
        "false_negative_image_count": fn_images,
        "false_positive_image_rate": fp_images / segmentation_count if segmentation_count else 0.0,
        "false_negative_image_rate": fn_images / segmentation_count if segmentation_count else 0.0,
        "metrics": metrics,
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
        and row.get("expected_image_status") == "ANNOTATED"
        and (difficulties is None or row.get("difficulty") in difficulties)
    ]
    predictions_by_id = _read_predictions(args.selected_predictions)
    expected_ids = {int(row["image_id"]) for row in records}
    if not expected_ids.issubset(predictions_by_id):
        raise ValueError("selected predictions do not cover the selected scope")
    predictions = [predictions_by_id[int(row["image_id"])] for row in records]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    recaptured_ids = union_recapture_ids(reports)
    if not recaptured_ids.issubset(expected_ids):
        raise ValueError("recapture report contains images outside the selected scope")
    overall = _scope_metrics(records, predictions, recaptured_ids)
    by_difficulty = {}
    for difficulty in sorted({str(row["difficulty"]) for row in records}):
        mask = [str(row["difficulty"]) == difficulty for row in records]
        subset_records = [row for row, include in zip(records, mask) if include]
        subset_predictions = [row for row, include in zip(predictions, mask) if include]
        by_difficulty[difficulty] = _scope_metrics(
            subset_records,
            subset_predictions,
            recaptured_ids,
        )
    accepted_records = [row for row in records if int(row["image_id"]) not in recaptured_ids]
    accepted_predictions = [
        prediction
        for row, prediction in zip(records, predictions)
        if int(row["image_id"]) not in recaptured_ids
    ]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_fixed_recapture_rule_union",
        "selection_scope": "fixed OR of prediction-only recapture rules",
        "selection_status": "development_validation_only_not_locked_test",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "selected_predictions": str(args.selected_predictions),
        "recapture_reports": [str(path) for path in args.reports],
        "recaptured_image_ids": sorted(recaptured_ids),
        "recaptured_by_difficulty": dict(
            sorted(
                Counter(
                    str(record["difficulty"])
                    for record in records
                    if int(record["image_id"]) in recaptured_ids
                ).items()
            )
        ),
        "overall": overall,
        "by_difficulty": by_difficulty,
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
        description="Evaluate the union of fixed detector IMAGE_RECAPTURE rules"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--selected-predictions", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
