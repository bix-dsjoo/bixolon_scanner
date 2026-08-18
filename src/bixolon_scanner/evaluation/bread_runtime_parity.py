from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _bbox_iou(left: dict[str, int], right: dict[str, int]) -> float:
    left_x2 = left["x"] + left["width"]
    left_y2 = left["y"] + left["height"]
    right_x2 = right["x"] + right["width"]
    right_y2 = right["y"] + right["height"]
    intersection = max(0, min(left_x2, right_x2) - max(left["x"], right["x"])) * max(
        0, min(left_y2, right_y2) - max(left["y"], right["y"])
    )
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union else 0.0


def _decision(item: dict[str, Any]) -> tuple[Any, ...]:
    prediction = item["prediction"]
    return (
        item["status"],
        tuple(item["reason_codes"]),
        prediction["class_id"] if prediction else None,
        tuple(candidate["class_id"] for candidate in item["top3"]),
    )


def compare_runtime_traces(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    minimum_bbox_iou: float = 0.999,
    maximum_confidence_error: float = 0.001,
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("runtime decision trace image counts differ")
    decision_mismatch_images: list[int | str] = []
    bbox_mismatch_images: list[int | str] = []
    minimum_observed_bbox_iou = 1.0
    maximum_observed_confidence_error = 0.0
    segmentation_count = 0
    for left, right in zip(reference, candidate, strict=True):
        image_id = left["image_id"]
        if image_id != right["image_id"]:
            raise ValueError("runtime decision trace image ids differ")
        image_decision_mismatch = (
            left["status"] != right["status"]
            or left["reason_codes"] != right["reason_codes"]
            or left["worker_version"] != right["worker_version"]
            or left["detector_version"] != right["detector_version"]
            or left["classifier_version"] != right["classifier_version"]
            or len(left["segmentations"]) != len(right["segmentations"])
        )
        if not image_decision_mismatch:
            for left_item, right_item in zip(
                left["segmentations"], right["segmentations"], strict=True
            ):
                segmentation_count += 1
                image_decision_mismatch |= _decision(left_item) != _decision(right_item)
                overlap = _bbox_iou(left_item["bbox"], right_item["bbox"])
                minimum_observed_bbox_iou = min(minimum_observed_bbox_iou, overlap)
                if overlap < minimum_bbox_iou and image_id not in bbox_mismatch_images:
                    bbox_mismatch_images.append(image_id)
                maximum_observed_confidence_error = max(
                    maximum_observed_confidence_error,
                    abs(left_item["confidence"] - right_item["confidence"]),
                )
                for left_candidate, right_candidate in zip(
                    left_item["top3"], right_item["top3"], strict=True
                ):
                    maximum_observed_confidence_error = max(
                        maximum_observed_confidence_error,
                        abs(left_candidate["confidence"] - right_candidate["confidence"]),
                    )
        if image_decision_mismatch:
            decision_mismatch_images.append(image_id)
    passes = (
        not decision_mismatch_images
        and not bbox_mismatch_images
        and maximum_observed_confidence_error <= maximum_confidence_error
    )
    return {
        "schema_version": "1.0",
        "evaluation": "bread_1_1_runtime_cpu_cuda_parity",
        "image_count": len(reference),
        "segmentation_count": segmentation_count,
        "decision_mismatch_image_count": len(decision_mismatch_images),
        "decision_mismatch_image_ids": decision_mismatch_images,
        "bbox_mismatch_image_count": len(bbox_mismatch_images),
        "bbox_mismatch_image_ids": bbox_mismatch_images,
        "minimum_observed_bbox_iou": minimum_observed_bbox_iou,
        "minimum_bbox_iou_tolerance": minimum_bbox_iou,
        "maximum_observed_confidence_error": maximum_observed_confidence_error,
        "maximum_confidence_error_tolerance": maximum_confidence_error,
        "final_status_class_rank_parity_exact": not decision_mismatch_images,
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Bread 1.1 CPU/CUDA runtime decisions")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-bbox-iou", type=float, default=0.999)
    parser.add_argument("--maximum-confidence-error", type=float, default=0.001)
    args = parser.parse_args()
    report = compare_runtime_traces(
        _read_jsonl(args.reference),
        _read_jsonl(args.candidate),
        minimum_bbox_iou=args.minimum_bbox_iou,
        maximum_confidence_error=args.maximum_confidence_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
