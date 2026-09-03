from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bixolon_scanner.evaluation.detector import _metrics_grid
from bixolon_scanner.experiments.bread.ssdlite_objectness_detector import (
    _evaluate_dataset,
    _standardize_records,
)
from bixolon_scanner.training.dinov3_objectness_detector import (
    build_dinov3_fasterrcnn,
    build_dinov3_fcos,
)
from bixolon_scanner.training.models import require_torch
from bixolon_scanner.training.ssdlite_objectness_detector import (
    CachedObjectnessDataset,
    build_ssdlite_objectness,
)


def _match_details(records, predictions, *, score_threshold: float, nms_threshold: float):
    torch = require_torch()
    from torchvision.ops import box_iou, nms

    confusion: Counter[str] = Counter()
    missed_by_class: Counter[str] = Counter()
    false_positive_by_class: Counter[str] = Counter()
    failures = []
    matched_count = 0
    correct_count = 0
    for record, prediction in zip(records, predictions, strict=True):
        boxes = torch.as_tensor(prediction["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(prediction["scores"], dtype=torch.float32)
        class_ids = torch.as_tensor(prediction["class_ids"], dtype=torch.int64)
        eligible = scores >= score_threshold
        boxes = boxes[eligible]
        scores = scores[eligible]
        class_ids = class_ids[eligible]
        if len(boxes):
            keep = nms(boxes, scores, nms_threshold)
            boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        annotations = record["annotations"]
        ground_truth = torch.as_tensor(
            [
                [
                    row["bbox_xywh"][0],
                    row["bbox_xywh"][1],
                    row["bbox_xywh"][0] + row["bbox_xywh"][2],
                    row["bbox_xywh"][1] + row["bbox_xywh"][3],
                ]
                for row in annotations
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        overlaps = box_iou(boxes, ground_truth) if len(boxes) and len(ground_truth) else None
        unmatched_gt = set(range(len(annotations)))
        unmatched_detection = []
        image_matches = []
        for detection_index in range(len(boxes)):
            if not unmatched_gt:
                unmatched_detection.extend(range(detection_index, len(boxes)))
                break
            best = max(
                unmatched_gt,
                key=lambda index: float(overlaps[detection_index, index]),
            )
            overlap = float(overlaps[detection_index, best])
            if overlap < 0.5:
                unmatched_detection.append(detection_index)
                continue
            unmatched_gt.remove(best)
            truth = int(annotations[best]["category_id"])
            predicted = int(class_ids[detection_index])
            matched_count += 1
            correct_count += predicted == truth
            confusion[f"{truth}->{predicted}"] += 1
            image_matches.append(
                {
                    "gt_index": best,
                    "truth_class": truth,
                    "predicted_class": predicted,
                    "score": float(scores[detection_index]),
                    "iou": overlap,
                }
            )
        for gt_index in unmatched_gt:
            missed_by_class[str(int(annotations[gt_index]["category_id"]))] += 1
        for detection_index in unmatched_detection:
            false_positive_by_class[str(int(class_ids[detection_index]))] += 1
        if (
            unmatched_gt
            or unmatched_detection
            or any(row["truth_class"] != row["predicted_class"] for row in image_matches)
        ):
            failures.append(
                {
                    "image_id": int(record["image_id"]),
                    "missed_gt_indices": sorted(unmatched_gt),
                    "false_positive_count": len(unmatched_detection),
                    "matches": image_matches,
                }
            )
    return {
        "matched_count": matched_count,
        "class_correct_count": correct_count,
        "class_error_count": matched_count - correct_count,
        "matched_class_accuracy": correct_count / matched_count if matched_count else 0.0,
        "confusion": dict(sorted(confusion.items())),
        "missed_by_class": dict(sorted(missed_by_class.items())),
        "false_positive_by_class": dict(sorted(false_positive_by_class.items())),
        "failure_image_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fresh store-2 detector")
    parser.add_argument(
        "--architecture",
        choices=("ssdlite", "dinov3-fcos", "dinov3-fasterrcnn"),
        default="ssdlite",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.735)
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--foreground-class-count", type=int, default=20)
    parser.add_argument("--dinov3-weights", type=Path)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    torch = require_torch()
    dataset = CachedObjectnessDataset(
        args.manifest,
        args.dataset_root,
        args.cache,
        training=False,
        class_aware=args.foreground_class_count > 1,
    )
    if args.architecture == "ssdlite":
        model = build_ssdlite_objectness(
            foreground_class_count=args.foreground_class_count,
            device="cuda",
        )
    elif args.architecture == "dinov3-fcos":
        if args.dinov3_weights is None:
            parser.error("--dinov3-weights is required for dinov3-fcos")
        model = build_dinov3_fcos(
            args.dinov3_weights,
            image_size=dataset.image_size,
            score_threshold=0.01,
            nms_threshold=args.nms_threshold,
            detections_per_image=100,
            topk_candidates=1000,
            fpn_channels=args.fpn_channels,
            unfreeze_last_stages=0,
            foreground_class_count=args.foreground_class_count,
            device="cuda",
        )
    else:
        if args.dinov3_weights is None:
            parser.error("--dinov3-weights is required for dinov3-fasterrcnn")
        if args.foreground_class_count != 1:
            parser.error("dinov3-fasterrcnn requires --foreground-class-count 1")
        model = build_dinov3_fasterrcnn(
            args.dinov3_weights,
            image_size=dataset.image_size,
            score_threshold=0.001,
            nms_threshold=args.nms_threshold,
            detections_per_image=100,
            fpn_channels=args.fpn_channels,
            unfreeze_last_stages=0,
            device="cuda",
        )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    predictions = _evaluate_dataset(model, dataset, batch_size=args.batch_size)
    if args.architecture == "dinov3-fcos" and args.foreground_class_count > 1:
        for prediction in predictions:
            prediction["class_ids"] = [int(class_id) + 1 for class_id in prediction["class_ids"]]
    elif args.architecture == "dinov3-fasterrcnn":
        for prediction in predictions:
            prediction["class_ids"] = [0] * len(prediction["class_ids"])
    records = _standardize_records(dataset)
    metrics = _metrics_grid(
        records,
        predictions,
        score_thresholds=[args.score_threshold],
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=0.5,
        max_queries=300,
    )[0]
    details = _match_details(
        records,
        predictions,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    prediction_body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in predictions
    )
    args.predictions.write_text(prediction_body, encoding="utf-8", newline="\n")
    report = {
        "policy_version": "0.1.13",
        "architecture": args.architecture,
        "score_threshold": args.score_threshold,
        "nms_iou_threshold": args.nms_threshold,
        "match_iou_threshold": 0.5,
        "image_count": len(records),
        "ground_truth_count": sum(len(row["annotations"]) for row in records),
        "metrics": metrics,
        "class_metrics": details,
        "checkpoint": str(args.checkpoint),
        "predictions": str(args.predictions),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {**report, "class_metrics": {k: v for k, v in details.items() if k != "failures"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
