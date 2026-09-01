from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...evaluation.detector import _iou, _xywh_to_xyxy
from ...pipeline.ports import Detection
from ...runtime.onnx import nms
from ...training.models import require_torch
from ...training.ssdlite_objectness_detector import (
    CachedObjectnessDataset,
    build_ssdlite_objectness,
)
from .ssdlite_objectness_detector import _evaluate_dataset, _standardize_records


def _analyze_dataset(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    threshold: float,
    nms_threshold: float,
) -> dict[str, Any]:
    failures = []
    matched_total = 0
    predicted_total = 0
    gt_total = 0
    for record, prediction in zip(records, predictions, strict=True):
        selected = nms(
            [
                Detection(*box, score)
                for box, score in zip(prediction["boxes_xyxy"], prediction["scores"], strict=True)
                if score >= threshold
            ],
            nms_threshold,
        )
        ground_truth = [
            _xywh_to_xyxy(annotation["bbox_xywh"]) for annotation in record["annotations"]
        ]
        unmatched_gt = set(range(len(ground_truth)))
        unmatched_prediction = []
        matches = []
        for detection in sorted(selected, key=lambda item: item.score, reverse=True):
            box = np.asarray(
                [detection.x1, detection.y1, detection.x2, detection.y2],
                dtype=np.float32,
            )
            candidates = [(index, _iou(box, ground_truth[index])) for index in unmatched_gt]
            if not candidates:
                unmatched_prediction.append(detection)
                continue
            gt_index, overlap = max(candidates, key=lambda item: item[1])
            if overlap < 0.5:
                unmatched_prediction.append(detection)
                continue
            unmatched_gt.remove(gt_index)
            matches.append(
                {
                    "gt_index": gt_index,
                    "score": detection.score,
                    "iou": overlap,
                }
            )
        matched_total += len(matches)
        predicted_total += len(selected)
        gt_total += len(ground_truth)
        if unmatched_gt or unmatched_prediction:
            failures.append(
                {
                    "image_id": record["image_id"],
                    "ground_truth_count": len(ground_truth),
                    "prediction_count": len(selected),
                    "false_negative_count": len(unmatched_gt),
                    "false_positive_count": len(unmatched_prediction),
                    "missed_ground_truth": [
                        {
                            "gt_index": index,
                            "bbox_xywh": record["annotations"][index]["bbox_xywh"],
                        }
                        for index in sorted(unmatched_gt)
                    ],
                    "unmatched_predictions": [
                        {
                            "bbox_xyxy": [
                                detection.x1,
                                detection.y1,
                                detection.x2,
                                detection.y2,
                            ],
                            "score": detection.score,
                            "maximum_gt_iou": max(
                                (
                                    _iou(
                                        np.asarray(
                                            [
                                                detection.x1,
                                                detection.y1,
                                                detection.x2,
                                                detection.y2,
                                            ],
                                            dtype=np.float32,
                                        ),
                                        box,
                                    )
                                    for box in ground_truth
                                ),
                                default=None,
                            ),
                        }
                        for detection in unmatched_prediction
                    ],
                    "matches": matches,
                }
            )
    return {
        "image_count": len(records),
        "ground_truth_count": gt_total,
        "prediction_count": predicted_total,
        "matched_count": matched_total,
        "false_positive_count": predicted_total - matched_total,
        "false_negative_count": gt_total - matched_total,
        "failure_image_count": len(failures),
        "empty_ground_truth_failure_count": sum(row["ground_truth_count"] == 0 for row in failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SSDLite objectness detector failures")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-cache", type=Path, required=True)
    parser.add_argument("--operational-manifest", type=Path, required=True)
    parser.add_argument("--operational-root", type=Path, required=True)
    parser.add_argument("--operational-cache", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--nms-threshold", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch = require_torch()
    real_dataset = CachedObjectnessDataset(
        args.real_manifest, args.real_root, args.real_cache, training=False
    )
    operational_dataset = CachedObjectnessDataset(
        args.operational_manifest,
        args.operational_root,
        args.operational_cache,
        training=False,
    )
    model = build_ssdlite_objectness(device="cuda")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    report = {
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "real": _analyze_dataset(
            _standardize_records(real_dataset),
            _evaluate_dataset(model, real_dataset, batch_size=args.batch_size),
            threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
        ),
        "operational": _analyze_dataset(
            _standardize_records(operational_dataset),
            _evaluate_dataset(model, operational_dataset, batch_size=args.batch_size),
            threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
