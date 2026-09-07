from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bixolon_scanner.evaluation.detector import _iou, _metrics_grid
from bixolon_scanner.evaluation.onnx_detector import (
    _preselect_view_prediction,
    load_records,
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _selected(row: dict, *, threshold: float, nms_threshold: float) -> dict:
    prediction = {key: row[key] for key in ("boxes_xyxy", "scores", "class_ids", "top3_class_ids")}
    result = _preselect_view_prediction(
        prediction,
        minimum_score=threshold,
        nms_iou_threshold=nms_threshold,
        max_object_aspect_ratio=5.0,
    )
    return result


def _containment_fraction(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0


def _append_recovery(
    primary: dict,
    backup: dict,
    *,
    minimum_support_iou: float,
    maximum_support_iou: float,
    maximum_containment_fraction: float = 1.0,
) -> dict | None:
    primary_boxes = [np.asarray(box, dtype=np.float32) for box in primary["boxes_xyxy"]]
    backup_boxes = [np.asarray(box, dtype=np.float32) for box in backup["boxes_xyxy"]]
    candidates = []
    for backup_index, backup_box in enumerate(backup_boxes):
        support = max(
            (
                (_iou(backup_box, box), _containment_fraction(backup_box, box))
                for box in primary_boxes
            ),
            default=(0.0, 0.0),
        )
        maximum_iou, containment_fraction = support
        if (
            minimum_support_iou <= maximum_iou <= maximum_support_iou
            and containment_fraction <= maximum_containment_fraction
        ):
            candidates.append(
                (
                    float(backup["scores"][backup_index]),
                    backup_index,
                    maximum_iou,
                    containment_fraction,
                )
            )
    if not candidates:
        return None
    _score, backup_index, _maximum_iou, _containment_fraction_value = max(candidates)
    result = {
        key: list(primary[key]) for key in ("boxes_xyxy", "scores", "class_ids", "top3_class_ids")
    }
    for key in ("boxes_xyxy", "class_ids", "top3_class_ids"):
        result[key].append(backup[key][backup_index])
    result["scores"].append(1.0)
    return result


def _split_recovery(
    primary: dict,
    backup: dict,
    *,
    support_iou: float,
    maximum_pair_iou: float,
) -> dict | None:
    primary_boxes = [np.asarray(box, dtype=np.float32) for box in primary["boxes_xyxy"]]
    backup_boxes = [np.asarray(box, dtype=np.float32) for box in backup["boxes_xyxy"]]
    choices = []
    for primary_index, primary_box in enumerate(primary_boxes):
        supporters = [
            index for index, box in enumerate(backup_boxes) if _iou(primary_box, box) >= support_iou
        ]
        for first_offset, first in enumerate(supporters):
            for second in supporters[first_offset + 1 :]:
                pair_iou = _iou(backup_boxes[first], backup_boxes[second])
                if pair_iou <= maximum_pair_iou:
                    choices.append(
                        (
                            _iou(primary_box, backup_boxes[first])
                            + _iou(primary_box, backup_boxes[second]),
                            primary_index,
                            first,
                            second,
                        )
                    )
    if not choices:
        return None
    _score, primary_index, first, second = max(choices)
    keep_primary = [index for index in range(len(primary_boxes)) if index != primary_index]
    result = {}
    for key in ("boxes_xyxy", "scores", "class_ids", "top3_class_ids"):
        result[key] = [primary[key][index] for index in keep_primary] + [
            backup[key][first],
            backup[key][second],
        ]
    result["scores"] = [1.0] * len(result["scores"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective detector count recovery")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", default="multi_object_instances.json")
    parser.add_argument("--annotation-path", type=Path)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--primary-threshold", type=float, required=True)
    parser.add_argument("--primary-nms", type=float, required=True)
    parser.add_argument("--backup-nms", type=float, required=True)
    parser.add_argument("--backup-threshold", type=float, nargs="+", required=True)
    parser.add_argument("--minimum-primary-count", type=int, nargs="+", default=[0])
    parser.add_argument("--mode", choices=("replace", "split", "append"), default="replace")
    parser.add_argument(
        "--allow-append-without-count-increase",
        action="store_true",
        help=(
            "For append mode, inspect independently supported backup hypotheses without "
            "requiring the backup detector to have exactly one more selected box"
        ),
    )
    parser.add_argument("--support-iou", type=float, nargs="+", default=[0.3])
    parser.add_argument("--maximum-support-iou", type=float, nargs="+", default=[1.0])
    parser.add_argument("--maximum-containment-fraction", type=float, nargs="+", default=[1.0])
    parser.add_argument("--maximum-pair-iou", type=float, nargs="+", default=[0.4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(
        args.dataset_root,
        args.annotation,
        annotation_path=args.annotation_path,
    )
    primary_rows = _rows(args.primary)
    backup_rows = _rows(args.backup)
    if len(records) != len(primary_rows) or len(records) != len(backup_rows):
        raise ValueError("prediction source count does not match evaluation records")
    primary = [
        _selected(row, threshold=args.primary_threshold, nms_threshold=args.primary_nms)
        for row in primary_rows
    ]
    candidates = []
    for threshold in args.backup_threshold:
        backup = [
            _selected(row, threshold=threshold, nms_threshold=args.backup_nms)
            for row in backup_rows
        ]
        for minimum_count in args.minimum_primary_count:
            for support_iou in args.support_iou:
                for maximum_support_iou in args.maximum_support_iou:
                    for maximum_containment_fraction in args.maximum_containment_fraction:
                        for maximum_pair_iou in args.maximum_pair_iou:
                            recovered = []
                            recovery_count = 0
                            recovery_image_ids = []
                            for record, primary_row, backup_row in zip(
                                records, primary, backup, strict=True
                            ):
                                count_candidate = (
                                    len(primary_row["scores"]) >= minimum_count
                                    and len(backup_row["scores"]) == len(primary_row["scores"]) + 1
                                )
                                append_candidate = (
                                    args.mode == "append"
                                    and args.allow_append_without_count_increase
                                    and len(primary_row["scores"]) >= minimum_count
                                )
                                selected = None
                                if count_candidate and args.mode == "replace":
                                    selected = backup_row
                                elif count_candidate and args.mode == "split":
                                    selected = _split_recovery(
                                        primary_row,
                                        backup_row,
                                        support_iou=support_iou,
                                        maximum_pair_iou=maximum_pair_iou,
                                    )
                                elif count_candidate or append_candidate:
                                    selected = _append_recovery(
                                        primary_row,
                                        backup_row,
                                        minimum_support_iou=support_iou,
                                        maximum_support_iou=maximum_support_iou,
                                        maximum_containment_fraction=(maximum_containment_fraction),
                                    )
                                selected = selected or primary_row
                                if selected is not primary_row:
                                    recovery_count += 1
                                    recovery_image_ids.append(int(record["image_id"]))
                                recovered.append(selected)
                            metrics = _metrics_grid(
                                records,
                                recovered,
                                score_thresholds=[0.5],
                                nms_iou_threshold=1.0,
                                match_iou_threshold=0.5,
                                max_queries=300,
                                max_object_aspect_ratio=5.0,
                            )[0]
                            candidates.append(
                                metrics
                                | {
                                    "backup_threshold": threshold,
                                    "minimum_primary_count": minimum_count,
                                    "support_iou": support_iou,
                                    "maximum_support_iou": maximum_support_iou,
                                    "maximum_containment_fraction": (maximum_containment_fraction),
                                    "maximum_pair_iou": maximum_pair_iou,
                                    "recovery_image_count": recovery_count,
                                    "recovery_image_ids": recovery_image_ids,
                                    "append_requires_backup_count_increase": not (
                                        args.mode == "append"
                                        and args.allow_append_without_count_increase
                                    ),
                                }
                            )
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["false_positive_count"] + row["false_negative_count"],
            row["false_negative_count"],
        ),
    )
    report = {
        "evaluation": "development_selective_detector_count_recovery",
        "best": ranked[0],
        "top_candidates": ranked[:25],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
