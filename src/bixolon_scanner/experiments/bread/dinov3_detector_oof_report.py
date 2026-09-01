from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...training.data import read_manifest
from .rfdetr_box_evaluation import (
    proposal_coverage_error_rows,
    proposal_coverage_metrics,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def count_gate_catches_proposal_miss(
    *,
    expected_count: int,
    missed_count: int,
    count_prediction: int,
    count_confidence: float,
    minimum_confidence: float,
) -> bool:
    """Upper-bound gate assuming stage2 has removed every surplus proposal."""
    filtered_proposal_count = expected_count - missed_count
    return count_confidence < minimum_confidence or count_prediction != filtered_proposal_count


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records = [
        record
        for record in read_manifest(args.manifest)
        if record["record_type"] == "detection"
        and record["split"] == "development"
        and not record.get("exclude_from_detector_training", False)
    ]
    count_predictions = {int(row["image_id"]): row for row in _jsonl(args.count_predictions)}
    fold_reports = []
    all_errors = []
    total_proposals = 0
    for fold in (0, 1, 2):
        fold_records = [record for record in records if int(record["fold"]) == fold]
        prediction_path = Path(str(args.predictions_template).format(fold=fold))
        predictions = _jsonl(prediction_path)
        metrics = proposal_coverage_metrics(fold_records, predictions)
        errors = proposal_coverage_error_rows(fold_records, predictions)
        all_errors.extend(errors)
        total_proposals += int(metrics["proposal_count"])
        fold_reports.append(
            {
                "fold": fold,
                "prediction_path": prediction_path.as_posix(),
                "proposal_metrics": metrics,
            }
        )

    safety_rows = []
    for error in all_errors:
        image_id = int(error["image_id"])
        count = count_predictions[image_id]
        missed_count = int(error["false_negative_count"])
        expected_count = int(count["expected_count"])
        caught = count_gate_catches_proposal_miss(
            expected_count=expected_count,
            missed_count=missed_count,
            count_prediction=int(count["predicted_count"]),
            count_confidence=float(count["confidence"]),
            minimum_confidence=args.count_confidence_minimum,
        )
        safety_rows.append(
            {
                "image_id": image_id,
                "fold": int(error["fold"]),
                "difficulty": error.get("difficulty"),
                "missed_count": missed_count,
                "expected_count": expected_count,
                "count_prediction": int(count["predicted_count"]),
                "count_confidence": float(count["confidence"]),
                "upper_bound_gate_caught": caught,
            }
        )

    ground_truth_count = sum(len(record["annotations"]) for record in records)
    false_negative_count = sum(row["missed_count"] for row in safety_rows)
    matched_count = ground_truth_count - false_negative_count
    caught_count = sum(row["upper_bound_gate_caught"] for row in safety_rows)
    report = {
        "schema_version": "1.0",
        "experiment": "bread-dinov3-objectness-group-oof-summary",
        "selection_scope": "development group-aware OOF; not independent test",
        "prohibited_detector_inputs": {
            "existing_detector_used": False,
            "yolo_family_used": False,
            "rfdetr_used": False,
        },
        "stage1": {
            "image_count": len(records),
            "ground_truth_count": ground_truth_count,
            "matched_count": matched_count,
            "false_negative_count": false_negative_count,
            "missed_image_count": len(safety_rows),
            "recall": matched_count / ground_truth_count,
            "proposal_count": total_proposals,
            "mean_proposals_per_image": total_proposals / len(records),
        },
        "folds": fold_reports,
        "count_safety_upper_bound": {
            "count_confidence_minimum": args.count_confidence_minimum,
            "missed_image_count": len(safety_rows),
            "count_prediction_exact_on_missed_image_count": sum(
                row["count_prediction"] == row["expected_count"] for row in safety_rows
            ),
            "upper_bound_gate_caught_image_count": caught_count,
            "upper_bound_gate_missed_image_count": len(safety_rows) - caught_count,
            "minimum_count_confidence_on_missed_images": min(
                row["count_confidence"] for row in safety_rows
            ),
            "assumption": (
                "stage2 perfectly removes surplus proposals without removing a covered object; "
                "this is not yet measured"
            ),
        },
        "missed_images": safety_rows,
        "decision": {
            "stage1_runtime_eligible": False,
            "reason": (
                "proposal recall is below one and stage2 false-positive filtering plus accepted-route "
                "safety have not been measured end to end"
            ),
            "active_runtime_modified": False,
        },
        "limitations": [
            "checkpoint epochs differ across folds and were selected on their validation folds",
            "the count head and detector share a DINOv3 family representation and are not fully independent",
            "count-gate coverage is an oracle-filtered upper bound, not an accepted-path result",
            "a separate capture-session calibration set and locked test are still required",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate DINOv3 detector group-OOF evidence")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-template", type=Path, required=True)
    parser.add_argument("--count-predictions", type=Path, required=True)
    parser.add_argument("--count-confidence-minimum", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0.0 <= args.count_confidence_minimum <= 1.0:
        raise ValueError("count confidence minimum must be between zero and one")
    build_report(args)


if __name__ == "__main__":
    main()
