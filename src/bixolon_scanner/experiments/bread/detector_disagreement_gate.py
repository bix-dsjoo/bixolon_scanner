from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import box_iou
from ...training.data import read_manifest


def has_perfect_iou_matching(
    primary: list[Detection], recovery: list[Detection], *, iou_threshold: float
) -> bool:
    """Return whether every primary box has a distinct recovery box above the IoU gate."""
    if len(primary) != len(recovery):
        return False
    if not primary:
        return True
    adjacency = [
        [index for index, other in enumerate(recovery) if box_iou(box, other) >= iou_threshold]
        for box in primary
    ]
    matched_primary = [-1] * len(recovery)

    def augment(primary_index: int, seen: set[int]) -> bool:
        for recovery_index in adjacency[primary_index]:
            if recovery_index in seen:
                continue
            seen.add(recovery_index)
            previous = matched_primary[recovery_index]
            if previous == -1 or augment(previous, seen):
                matched_primary[recovery_index] = primary_index
                return True
        return False

    return all(augment(index, set()) for index in range(len(primary)))


def prediction_rows_agree(
    primary_row: dict[str, Any],
    recovery_row: dict[str, Any],
    *,
    iou_threshold: float,
) -> bool:
    if primary_row["image_id"] != recovery_row["image_id"]:
        raise ValueError("detector disagreement image ids differ")
    primary = [
        Detection(*box, float(score))
        for box, score in zip(primary_row["boxes_xyxy"], primary_row["scores"])
    ]
    recovery = [
        Detection(*box, float(score))
        for box, score in zip(recovery_row["boxes_xyxy"], recovery_row["scores"])
    ]
    return has_perfect_iou_matching(primary, recovery, iou_threshold=iou_threshold)


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
    primary_by_id = _read_predictions(args.primary_predictions)
    recovery_by_id = _read_predictions(args.recovery_predictions)
    expected_ids = {int(row["image_id"]) for row in records}
    if not expected_ids.issubset(primary_by_id) or not expected_ids.issubset(recovery_by_id):
        raise ValueError("detector disagreement predictions do not cover the selected scope")
    primary = [primary_by_id[int(record["image_id"])] for record in records]
    recovery = [recovery_by_id[int(record["image_id"])] for record in records]
    raw_metrics = _metrics(
        records,
        primary,
        score_threshold=0.0,
        nms_iou_threshold=1.0,
        match_iou_threshold=0.5,
        max_queries=600,
    )
    raw_fp_images, raw_fn_images = _image_error_counts(records, primary)
    candidates = []
    accepted_cache: dict[float, tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]] = {}
    for threshold in args.agreement_iou_thresholds:
        accepted_records = []
        accepted_predictions = []
        recaptured_ids = []
        for record, primary_row, recovery_row in zip(records, primary, recovery):
            if prediction_rows_agree(
                primary_row,
                recovery_row,
                iou_threshold=float(threshold),
            ):
                accepted_records.append(record)
                accepted_predictions.append(primary_row)
            else:
                recaptured_ids.append(int(record["image_id"]))
        metrics = _metrics(
            accepted_records,
            accepted_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        fp_images, fn_images = _image_error_counts(accepted_records, accepted_predictions)
        segmentation_rate = len(accepted_records) / len(records) if records else 0.0
        candidate = {
            "agreement_iou_threshold": float(threshold),
            "segmentation_image_count": len(accepted_records),
            "image_recapture_count": len(recaptured_ids),
            "segmentation_rate": segmentation_rate,
            "false_positive_image_count": fp_images,
            "false_negative_image_count": fn_images,
            "metrics": metrics,
        }
        candidates.append(candidate)
        accepted_cache[float(threshold)] = (
            accepted_records,
            accepted_predictions,
            recaptured_ids,
        )
    eligible = [
        row for row in candidates if row["segmentation_rate"] >= args.minimum_segmentation_rate
    ]
    if not eligible:
        eligible = candidates
    zero_error = [
        row
        for row in eligible
        if row["false_positive_image_count"] == 0 and row["false_negative_image_count"] == 0
    ]
    selected = max(
        zero_error or eligible,
        key=lambda row: (
            -(row["false_positive_image_count"] + row["false_negative_image_count"]),
            row["segmentation_rate"],
            row["agreement_iou_threshold"],
        ),
    )
    accepted_records, accepted_predictions, recaptured_ids = accepted_cache[
        selected["agreement_iou_threshold"]
    ]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_detector_cross_scale_disagreement_gate",
        "selection_scope": "prediction-only count and one-to-one geometry agreement",
        "folds": sorted(folds),
        "difficulties": sorted(difficulties) if difficulties is not None else None,
        "image_count": len(records),
        "minimum_segmentation_rate": args.minimum_segmentation_rate,
        "raw_primary": {
            **raw_metrics,
            "false_positive_image_count": raw_fp_images,
            "false_negative_image_count": raw_fn_images,
        },
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(zero_error),
        "selected": {
            **selected,
            "recaptured_image_ids": recaptured_ids,
            "recaptured_by_difficulty": dict(
                sorted(
                    Counter(
                        str(record["difficulty"])
                        for record in records
                        if int(record["image_id"]) in set(recaptured_ids)
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
        description="Evaluate a label-free cross-scale detector disagreement IMAGE_RECAPTURE gate"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--recovery-predictions", type=Path, required=True)
    parser.add_argument("--agreement-iou-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--minimum-segmentation-rate", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
