from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def object_metrics(
    *,
    mask: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    top3: np.ndarray,
    approved: np.ndarray,
    unknown: np.ndarray,
) -> dict[str, Any]:
    sample_count = int(np.count_nonzero(mask))
    denominator = max(1, sample_count)
    approved_mask = mask & approved
    unknown_mask = mask & unknown
    segment_recapture = mask & (~approved) & (~unknown)
    top3_correct = np.any(top3 == targets[:, None], axis=1)
    approved_count = int(np.count_nonzero(approved_mask))
    unknown_count = int(np.count_nonzero(unknown_mask))
    recapture_count = int(np.count_nonzero(segment_recapture))
    misrecognition_count = int(np.count_nonzero(approved_mask & (predictions != targets)))
    candidate_out_count = int(np.count_nonzero(unknown_mask & (~top3_correct)))
    return {
        "sample_count": sample_count,
        "approved_count": approved_count,
        "approved_rate": approved_count / denominator,
        "approved_misrecognition_count": misrecognition_count,
        "approved_misrecognition_rate": misrecognition_count / denominator,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_count / denominator,
        "unknown_top3_candidate_out_count": candidate_out_count,
        "unknown_top3_candidate_out_rate": candidate_out_count / denominator,
        "segment_recapture_count": recapture_count,
        "segment_recapture_rate": recapture_count / denominator,
    }


def gates(
    detector: dict[str, Any],
    classifier: dict[str, Any],
    *,
    minimum_segmentation_rate: float,
    minimum_approved_rate: float,
    maximum_error_rate: float,
) -> dict[str, bool]:
    values = {
        "segmentation_rate": detector["segmentation_rate"] >= minimum_segmentation_rate,
        "segmentation_image_false_negative_rate": detector["false_negative_image_rate"]
        <= maximum_error_rate,
        "segmentation_image_false_positive_rate": detector["false_positive_image_rate"]
        <= maximum_error_rate,
        "approved_rate": classifier["approved_rate"] >= minimum_approved_rate,
        "approved_misrecognition_rate": classifier["approved_misrecognition_rate"]
        <= maximum_error_rate,
        "unknown_top3_candidate_out_rate": classifier["unknown_top3_candidate_out_rate"]
        <= maximum_error_rate,
    }
    return {**values, "all_met": all(values.values())}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    detector_report = json.loads(args.detector_report.read_text(encoding="utf-8"))
    if detector_report.get("error_images"):
        raise ValueError("detector report contains accepted FP/FN error images")
    rows = [
        json.loads(line)
        for line in args.classifier_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    decisions = np.load(args.classifier_decisions)
    if len(rows) != len(decisions["targets"]):
        raise ValueError("classifier records and decisions are not aligned")
    manifest = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    difficulties = np.asarray([str(manifest[int(row["image_id"])]["difficulty"]) for row in rows])
    unexpected = set(difficulties.tolist()) - set(args.difficulties)
    if unexpected:
        raise ValueError(f"classifier decisions include excluded difficulties: {unexpected}")
    approved_key = f"{args.decision_scope}_approved"
    unknown_key = f"{args.decision_scope}_unknown"
    approved = decisions[approved_key].astype(bool)
    unknown = decisions[unknown_key].astype(bool)
    targets = decisions["targets"].astype(np.int64)
    predictions = decisions["predictions"].astype(np.int64)
    top3 = decisions["top3"].astype(np.int64)
    full_mask = np.ones(len(targets), dtype=bool)
    overall_classifier = object_metrics(
        mask=full_mask,
        targets=targets,
        predictions=predictions,
        top3=top3,
        approved=approved,
        unknown=unknown,
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_fixed_detector_classifier_pipeline_candidate",
        "selection_scope": "development_grouped_oof_not_locked_test",
        "difficulties": list(args.difficulties),
        "scan_log_excluded": True,
        "decision_scope": args.decision_scope,
        "overall": {
            "images": detector_report["overall"],
            "objects": overall_classifier,
        },
        "by_difficulty": {},
        "gates": gates(
            detector_report["overall"],
            overall_classifier,
            minimum_segmentation_rate=args.minimum_segmentation_rate,
            minimum_approved_rate=args.minimum_approved_rate,
            maximum_error_rate=args.maximum_error_rate,
        ),
        "promotion_ready": False,
        "promotion_blockers": [
            "independent locked test is unavailable",
            "deployable ONNX package and Worker parity/latency are not yet validated",
        ],
    }
    for difficulty in args.difficulties:
        mask = difficulties == difficulty
        report["by_difficulty"][difficulty] = {
            "images": detector_report["by_difficulty"][difficulty],
            "objects": object_metrics(
                mask=mask,
                targets=targets,
                predictions=predictions,
                top3=top3,
                approved=approved,
                unknown=unknown,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine fixed detector and classifier decisions into promotion metrics"
    )
    parser.add_argument("--detector-report", type=Path, required=True)
    parser.add_argument("--classifier-records", type=Path, required=True)
    parser.add_argument("--classifier-decisions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decision-scope",
        choices=("grouped_oof", "final"),
        default="grouped_oof",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=("EASY", "MEDIUM", "HARD"),
    )
    parser.add_argument("--minimum-segmentation-rate", type=float, default=0.90)
    parser.add_argument("--minimum-approved-rate", type=float, default=0.90)
    parser.add_argument("--maximum-error-rate", type=float, default=0.001)
    build_report(parser.parse_args())


if __name__ == "__main__":
    main()
