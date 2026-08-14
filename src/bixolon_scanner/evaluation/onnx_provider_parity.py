from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline.ports import Detection
from ..runtime.onnx import box_iou, nms
from .detector import _allowed_aspect_ratio


def _selected(
    prediction: dict[str, Any],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    max_object_aspect_ratio: float | None,
) -> list[Detection]:
    candidates = [
        Detection(*box, score)
        for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
        if score >= score_threshold and _allowed_aspect_ratio(box, max_object_aspect_ratio)
    ]
    return nms(candidates, nms_iou_threshold)


def provider_parity_report(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    score_threshold: float,
    nms_iou_threshold: float = 0.7,
    max_object_aspect_ratio: float | None = 5.0,
    minimum_pair_iou: float = 0.999,
    maximum_score_error: float = 0.02,
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("provider prediction image counts differ")
    score_error = 0.0
    raw_box_error = 0.0
    class_mismatches = 0
    query_count = 0
    final_mismatch_images = 0
    minimum_final_iou = 1.0
    final_score_error = 0.0
    for left, right in zip(reference, candidate):
        if left["image_id"] != right["image_id"]:
            raise ValueError("provider prediction image ids differ")
        left_scores = np.asarray(left["scores"], dtype=np.float64)
        right_scores = np.asarray(right["scores"], dtype=np.float64)
        left_boxes = np.asarray(left["boxes_xyxy"], dtype=np.float64)
        right_boxes = np.asarray(right["boxes_xyxy"], dtype=np.float64)
        if left_scores.shape != right_scores.shape or left_boxes.shape != right_boxes.shape:
            raise ValueError("provider raw output shapes differ")
        score_error = max(score_error, float(np.max(np.abs(left_scores - right_scores))))
        raw_box_error = max(raw_box_error, float(np.max(np.abs(left_boxes - right_boxes))))
        left_classes = np.asarray(left["class_ids"], dtype=np.int64)
        right_classes = np.asarray(right["class_ids"], dtype=np.int64)
        class_mismatches += int(np.count_nonzero(left_classes != right_classes))
        query_count += len(left_classes)

        left_selected = _selected(
            left,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_object_aspect_ratio=max_object_aspect_ratio,
        )
        right_selected = _selected(
            right,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_object_aspect_ratio=max_object_aspect_ratio,
        )
        if len(left_selected) != len(right_selected):
            final_mismatch_images += 1
            continue
        image_matches = True
        remaining = set(range(len(right_selected)))
        for left_detection in left_selected:
            pairs = [(index, box_iou(left_detection, right_selected[index])) for index in remaining]
            if not pairs:
                image_matches = False
                continue
            index, overlap = max(pairs, key=lambda item: item[1])
            remaining.remove(index)
            right_detection = right_selected[index]
            minimum_final_iou = min(minimum_final_iou, overlap)
            final_score_error = max(
                final_score_error, abs(left_detection.score - right_detection.score)
            )
            if overlap < minimum_pair_iou:
                image_matches = False
        final_mismatch_images += int(not image_matches)
    passes = final_mismatch_images == 0 and final_score_error <= maximum_score_error
    return {
        "image_count": len(reference),
        "raw_query_count": query_count,
        "permutation_sensitive_raw_score_max_abs_error_diagnostic": score_error,
        "permutation_sensitive_raw_box_pixel_max_abs_error_diagnostic": raw_box_error,
        "permutation_sensitive_raw_top1_class_mismatch_count_diagnostic": class_mismatches,
        "final_detection_mismatch_image_count": final_mismatch_images,
        "minimum_final_detection_pair_iou": minimum_final_iou,
        "final_detection_score_max_abs_error": final_score_error,
        "score_tolerance": maximum_score_error,
        "minimum_pair_iou_tolerance": minimum_pair_iou,
        "passes": passes,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU and CUDA detector predictions")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--max-object-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--minimum-pair-iou", type=float, default=0.999)
    parser.add_argument("--maximum-score-error", type=float, default=0.02)
    args = parser.parse_args()
    report = provider_parity_report(
        _read_jsonl(args.reference),
        _read_jsonl(args.candidate),
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_threshold,
        max_object_aspect_ratio=args.max_object_aspect_ratio,
        minimum_pair_iou=args.minimum_pair_iou,
        maximum_score_error=args.maximum_score_error,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
