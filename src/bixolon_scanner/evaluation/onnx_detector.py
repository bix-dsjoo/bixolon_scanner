from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..pipeline.ports import Detection
from ..runtime.imaging import decode_image, image_original_size
from ..runtime.onnx import OrtRunner, nms, prepare_rgb, sigmoid
from .detector import (
    _allowed_aspect_ratio,
    _iou,
    _metrics_grid,
    _xywh_to_xyxy,
    select_release_threshold_candidate,
)


def raw_outputs_to_prediction(
    logits: np.ndarray,
    boxes: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, list[Any]]:
    """Collapse class logits to object scores and convert normalized boxes to pixels."""
    values = np.asarray(logits, dtype=np.float32)
    normalized_boxes = np.asarray(boxes, dtype=np.float32)
    if values.ndim == 1:
        scores = sigmoid(values)
        ranks = np.zeros((len(values), 1), dtype=np.int64)
    elif values.ndim == 2:
        scores = sigmoid(values).max(axis=-1)
        ranks = np.argsort(-values, axis=-1, kind="stable")
    else:
        raise ValueError("detector logits must have shape [queries] or [queries, classes]")
    if normalized_boxes.shape != (len(scores), 4):
        raise ValueError("detector boxes must have shape [queries, 4]")

    converted_boxes: list[list[float]] = []
    converted_scores: list[float] = []
    converted_class_ids: list[int] = []
    converted_top3: list[list[int]] = []
    for score, box, rank in zip(scores, normalized_boxes, ranks):
        cx, cy, width, height = [float(value) for value in box]
        x1 = max(0.0, (cx - width * 0.5) * image_width)
        y1 = max(0.0, (cy - height * 0.5) * image_height)
        x2 = min(float(image_width), (cx + width * 0.5) * image_width)
        y2 = min(float(image_height), (cy + height * 0.5) * image_height)
        if x2 > x1 and y2 > y1:
            converted_boxes.append([x1, y1, x2, y2])
            converted_scores.append(float(score))
            converted_class_ids.append(int(rank[0]))
            converted_top3.append([int(value) for value in rank[:3]])
    return {
        "boxes_xyxy": converted_boxes,
        "scores": converted_scores,
        "class_ids": converted_class_ids,
        "top3_class_ids": converted_top3,
    }


def load_records(
    dataset_root: Path,
    annotation_name: str,
    *,
    annotation_path: Path | None = None,
) -> list[dict[str, Any]]:
    dataset_root = dataset_root.resolve()
    explicit_annotation = annotation_path is not None
    annotation_path = (
        annotation_path.resolve()
        if explicit_annotation
        else dataset_root / "annotations" / annotation_name
    )
    image_base = dataset_root if explicit_annotation else annotation_path.parent
    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    annotations: dict[int, list[dict[str, Any]]] = {}
    for row in payload["annotations"]:
        annotations.setdefault(int(row["image_id"]), []).append(
            {
                "bbox_xywh": [float(value) for value in row["bbox"]],
                "category_id": int(row["category_id"]),
            }
        )
    records = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        path = (image_base / str(image["file_name"])).resolve()
        path.relative_to(dataset_root)
        records.append(
            {
                "image_id": int(image["id"]),
                "image_path": path,
                "expected_image_status": str(image.get("status", "ANNOTATED")),
                "annotations": annotations.get(int(image["id"]), []),
            }
        )
    return records


def detector_classification_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    match_iou_threshold: float,
    max_object_aspect_ratio: float | None,
) -> dict[str, float | int | None]:
    matched = 0
    top1_correct = 0
    top3_correct = 0
    for record, prediction in zip(records, predictions):
        candidates = [
            (index, Detection(*box, score))
            for index, (box, score) in enumerate(
                zip(prediction["boxes_xyxy"], prediction["scores"])
            )
            if score >= score_threshold and _allowed_aspect_ratio(box, max_object_aspect_ratio)
        ]
        index_by_detection = {detection: index for index, detection in candidates}
        selected = nms([detection for _, detection in candidates], nms_iou_threshold)
        remaining = set(range(len(record["annotations"])))
        for detection in selected:
            prediction_index = index_by_detection[detection]
            box = np.asarray(
                [detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32
            )
            overlaps = [
                (gt_index, _iou(box, _xywh_to_xyxy(record["annotations"][gt_index]["bbox_xywh"])))
                for gt_index in remaining
            ]
            if not overlaps:
                continue
            gt_index, overlap = max(overlaps, key=lambda item: item[1])
            if overlap < match_iou_threshold:
                continue
            remaining.remove(gt_index)
            target = int(record["annotations"][gt_index]["category_id"]) - 1
            matched += 1
            top1_correct += int(prediction["class_ids"][prediction_index] == target)
            top3_correct += int(target in prediction["top3_class_ids"][prediction_index])
    return {
        "matched_sample_count": matched,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / matched if matched else None,
        "top3_correct": top3_correct,
        "top3_accuracy": top3_correct / matched if matched else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(
        args.dataset_root.resolve(),
        args.annotation,
        annotation_path=getattr(args, "annotation_path", None),
    )
    if args.expected_status is not None:
        records = [
            record for record in records if record["expected_image_status"] == args.expected_status
        ]
    runner = OrtRunner(args.model, args.provider, args.cuda_dll_dir)
    predictions = []
    query_count: int | None = None
    for record in records:
        jpeg_draft_size = getattr(args, "jpeg_draft_size", None)
        if jpeg_draft_size is None:
            with Image.open(record["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            image = decode_image(
                record["image_path"].read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=jpeg_draft_size,
            )
        try:
            width, height = image_original_size(image)
            tensor = prepare_rgb(
                image,
                (args.input_height, args.input_width),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                reducing_gap=1.0,
            )[None]
        finally:
            image.close()
        logits, boxes = runner.run([args.logits_output, args.boxes_output], args.input_name, tensor)
        logits = np.asarray(logits)[0]
        boxes = np.asarray(boxes)[0]
        query_count = len(logits) if query_count is None else query_count
        prediction = raw_outputs_to_prediction(
            logits,
            boxes,
            image_width=width,
            image_height=height,
        )
        if getattr(args, "include_class_logits", False):
            prediction["class_logits"] = np.asarray(logits, dtype=np.float32).tolist()
        prediction["image_id"] = record["image_id"]
        predictions.append(prediction)

    thresholds = np.linspace(
        args.min_score_threshold,
        args.max_score_threshold,
        args.threshold_steps,
        dtype=np.float64,
    )
    candidates = _metrics_grid(
        records,
        predictions,
        score_thresholds=thresholds,
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_queries=query_count or args.max_queries,
        max_object_aspect_ratio=args.max_object_aspect_ratio,
    )
    selected = select_release_threshold_candidate(candidates, args.target_recall)
    class_metrics = detector_classification_metrics(
        records,
        predictions,
        score_threshold=float(selected["score_threshold"]),
        nms_iou_threshold=args.nms_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_object_aspect_ratio=args.max_object_aspect_ratio,
    )
    report = {
        "evaluation": "onnx_detector_threshold_selection",
        "model": args.model.name,
        "annotation": args.annotation,
        "expected_status_filter": args.expected_status,
        "provider": args.provider,
        "jpeg_draft_size": getattr(args, "jpeg_draft_size", None),
        "threshold_policy": "recall_floor_then_precision_on_development_set",
        "target_recall": args.target_recall,
        "target_recall_satisfied": float(selected["recall"]) >= args.target_recall,
        "selected_score_threshold": selected["score_threshold"],
        "metrics": selected,
        "detector_classification_on_matched": class_metrics,
        "query_count": query_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an ONNX detector threshold")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", default="multi_object_instances.json")
    parser.add_argument(
        "--annotation-path",
        type=Path,
        help="Use an explicit COCO annotation while resolving file_name under --dataset-root",
    )
    parser.add_argument("--expected-status")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--logits-output", default="logits")
    parser.add_argument("--boxes-output", default="pred_boxes")
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--jpeg-draft-size", type=int)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--max-object-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--min-score-threshold", type=float, default=0.01)
    parser.add_argument("--max-score-threshold", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=197)
    parser.add_argument(
        "--include-class-logits",
        action="store_true",
        help="Include raw per-query class logits in diagnostic prediction output",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
