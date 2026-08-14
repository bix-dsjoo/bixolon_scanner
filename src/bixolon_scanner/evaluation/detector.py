from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..pipeline.ports import Detection
from ..runtime.onnx import nms as _nms
from ..training.data import DetectionDataset
from ..training.models import require_torch


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = box
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _allowed_aspect_ratio(box: list[float], maximum: float | None) -> bool:
    if maximum is None:
        return True
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    if width <= 0.0 or height <= 0.0:
        return False
    return max(width / height, height / width) <= maximum


def _metrics(
    records: list[dict],
    predictions: list[dict],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    match_iou_threshold: float,
    max_queries: int,
    max_object_aspect_ratio: float | None = None,
) -> dict[str, float | int]:
    matched_total = 0
    gt_total = 0
    predicted_total = 0
    count_correct = 0
    capacity_saturated = 0
    for record, prediction in zip(records, predictions):
        selected = [
            Detection(*box, score)
            for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
            if score >= score_threshold and _allowed_aspect_ratio(box, max_object_aspect_ratio)
        ]
        if len(selected) >= max_queries:
            capacity_saturated += 1
        selected = _nms(selected, nms_iou_threshold)
        gt_boxes = [_xywh_to_xyxy(annotation["bbox_xywh"]) for annotation in record["annotations"]]
        unmatched = set(range(len(gt_boxes)))
        matched = 0
        for detection in sorted(selected, key=lambda item: item.score, reverse=True):
            box = np.asarray([detection.x1, detection.y1, detection.x2, detection.y2])
            candidates = [(index, _iou(box, gt_boxes[index])) for index in unmatched]
            if not candidates:
                continue
            index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= match_iou_threshold:
                unmatched.remove(index)
                matched += 1
        gt_total += len(gt_boxes)
        predicted_total += len(selected)
        matched_total += matched
        count_correct += int(len(selected) == len(gt_boxes))
    precision = matched_total / predicted_total if predicted_total else 0.0
    recall = matched_total / gt_total if gt_total else 0.0
    return {
        "image_count": len(records),
        "ground_truth_count": gt_total,
        "prediction_count": predicted_total,
        "matched_count": matched_total,
        "recall": recall,
        "precision": precision,
        "count_accuracy": count_correct / len(records) if records else 0.0,
        "capacity_saturated_images": capacity_saturated,
    }


def _metrics_grid(
    records: list[dict],
    predictions: list[dict],
    *,
    score_thresholds,
    nms_iou_threshold: float,
    match_iou_threshold: float,
    max_queries: int,
    max_object_aspect_ratio: float | None = None,
) -> list[dict[str, float | int]]:
    """Evaluate a score grid with one greedy-NMS pass per image.

    Greedy NMS visits detections in descending score order. A detection below a
    later threshold cannot suppress a survivor at that threshold, because it is
    visited after every qualifying detection. Therefore NMS at the minimum grid
    threshold, followed by score-filtering its survivors, is exactly equivalent
    to filtering first and independently running NMS at every threshold. Python's
    stable sort also preserves the original tie order in both paths.
    """
    thresholds = [float(value) for value in score_thresholds]
    if not thresholds:
        raise ValueError("detector metric threshold grid must not be empty")
    minimum_threshold = min(thresholds)
    totals = [
        {
            "ground_truth_count": 0,
            "prediction_count": 0,
            "matched_count": 0,
            "count_correct": 0,
            "capacity_saturated_images": 0,
        }
        for _ in thresholds
    ]
    for record, prediction in zip(records, predictions):
        raw = [
            Detection(*box, score)
            for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
            if score >= minimum_threshold and _allowed_aspect_ratio(box, max_object_aspect_ratio)
        ]
        nms_survivors = _nms(raw, nms_iou_threshold)
        gt_boxes = [_xywh_to_xyxy(annotation["bbox_xywh"]) for annotation in record["annotations"]]
        for threshold, total in zip(thresholds, totals):
            raw_count = sum(detection.score >= threshold for detection in raw)
            selected = [detection for detection in nms_survivors if detection.score >= threshold]
            unmatched = set(range(len(gt_boxes)))
            matched = 0
            for detection in selected:
                box = np.asarray([detection.x1, detection.y1, detection.x2, detection.y2])
                candidates = [(index, _iou(box, gt_boxes[index])) for index in unmatched]
                if not candidates:
                    continue
                index, overlap = max(candidates, key=lambda item: item[1])
                if overlap >= match_iou_threshold:
                    unmatched.remove(index)
                    matched += 1
            total["ground_truth_count"] += len(gt_boxes)
            total["prediction_count"] += len(selected)
            total["matched_count"] += matched
            total["count_correct"] += int(len(selected) == len(gt_boxes))
            total["capacity_saturated_images"] += int(raw_count >= max_queries)
    results: list[dict[str, float | int]] = []
    for threshold, total in zip(thresholds, totals):
        predicted_total = int(total["prediction_count"])
        gt_total = int(total["ground_truth_count"])
        matched_total = int(total["matched_count"])
        results.append(
            {
                "image_count": len(records),
                "ground_truth_count": gt_total,
                "prediction_count": predicted_total,
                "matched_count": matched_total,
                "recall": matched_total / gt_total if gt_total else 0.0,
                "precision": matched_total / predicted_total if predicted_total else 0.0,
                "count_accuracy": (int(total["count_correct"]) / len(records) if records else 0.0),
                "capacity_saturated_images": int(total["capacity_saturated_images"]),
                "score_threshold": threshold,
            }
        )
    return results


def select_release_threshold_candidate(
    candidates: list[dict[str, float | int]], target_recall: float
) -> dict[str, float | int]:
    eligible = [item for item in candidates if float(item["recall"]) >= target_recall]
    if eligible:
        return max(
            eligible,
            key=lambda item: (
                float(item["precision"]),
                float(item["count_accuracy"]),
                float(item["score_threshold"]),
            ),
        )
    return max(
        candidates,
        key=lambda item: (
            float(item["recall"]),
            float(item["precision"]),
            float(item["count_accuracy"]),
        ),
    )


def evaluate(args: argparse.Namespace) -> None:
    torch = require_torch()
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = RTDetrV2ForObjectDetection.from_pretrained(args.checkpoint).to(device).eval()
    dataset = DetectionDataset(args.manifest, args.dataset_root, mode=args.mode, fold=args.fold)
    predictions: list[dict] = []
    for start in range(0, len(dataset.records), args.batch_size):
        batch_records = dataset.records[start : start + args.batch_size]
        images = []
        sizes = []
        for record in batch_records:
            images.append(_open_rgb(args.dataset_root / record["image_path"]))
            sizes.append([record["height"], record["width"]])
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            outputs = model(**inputs)
        processed = processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=torch.asarray(sizes, device=device),
        )
        for record, result in zip(batch_records, processed):
            predictions.append(
                {
                    "image_id": record["image_id"],
                    "boxes_xyxy": result["boxes"].float().cpu().numpy().tolist(),
                    "scores": result["scores"].float().cpu().numpy().tolist(),
                }
            )
    thresholds = (
        np.asarray([args.score_threshold], dtype=np.float64)
        if args.score_threshold is not None
        else np.linspace(args.min_score_threshold, args.max_score_threshold, args.threshold_steps)
    )
    candidates = _metrics_grid(
        dataset.records,
        predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_queries=int(model.config.num_queries),
        max_object_aspect_ratio=args.max_object_aspect_ratio,
    )
    selected = select_release_threshold_candidate(candidates, args.target_recall)
    manifest_metadata = json.loads(
        (args.manifest.parent / "metadata.json").read_text(encoding="utf-8")
    )
    report = {
        "model_version": getattr(args, "model_version", "0.1.0"),
        "dataset_version": manifest_metadata["dataset_version"],
        "mode": args.mode,
        "fold": args.fold,
        "checkpoint": args.checkpoint.name,
        "match_iou_threshold": args.match_iou_threshold,
        "nms_iou_threshold": args.nms_threshold,
        "max_object_aspect_ratio": args.max_object_aspect_ratio,
        "target_recall": args.target_recall,
        "threshold_policy": "fixed"
        if args.score_threshold is not None
        else "recall_floor_then_precision_on_evaluation_set",
        "selected_score_threshold": selected["score_threshold"],
        "target_recall_satisfied": selected["recall"] >= args.target_recall,
        "metrics": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(item) + "\n" for item in predictions), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RT-DETRv2 recall and count accuracy")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("validation", "evaluation", "test"), default="validation"
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--max-object-aspect-ratio", type=float)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Use a preselected score threshold (required for an unbiased final test)",
    )
    parser.add_argument("--min-score-threshold", type=float, default=0.05)
    parser.add_argument("--max-score-threshold", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=91)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
