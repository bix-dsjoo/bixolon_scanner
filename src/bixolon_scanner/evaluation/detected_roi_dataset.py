from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..pipeline.ports import Detection
from ..runtime.imaging import decode_image, image_original_size
from ..runtime.onnx import box_iou, classifier_crop_box, nms, prepare_rgb
from .onnx_detector import load_records


def _allowed_box(box: list[float], maximum_aspect_ratio: float) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width > 0 and height > 0 and max(width / height, height / width) <= maximum_aspect_ratio


def select_detections(
    prediction: dict[str, Any],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    maximum_aspect_ratio: float,
) -> list[Detection]:
    candidates = [
        Detection(*box, float(score))
        for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
        if score >= score_threshold and _allowed_box(box, maximum_aspect_ratio)
    ]
    return sorted(nms(candidates, nms_iou_threshold), key=lambda item: (item.y1, item.x1))


def match_detections(
    detections: list[Detection],
    annotations: list[dict[str, Any]],
    *,
    match_iou_threshold: float,
) -> dict[int, tuple[int, float]]:
    remaining = set(range(len(annotations)))
    matches: dict[int, tuple[int, float]] = {}
    for detection_index, detection in sorted(
        enumerate(detections), key=lambda item: item[1].score, reverse=True
    ):
        overlaps = []
        for annotation_index in remaining:
            x, y, width, height = annotations[annotation_index]["bbox_xywh"]
            target = Detection(x, y, x + width, y + height, 1.0)
            overlaps.append((annotation_index, box_iou(detection, target)))
        if not overlaps:
            continue
        annotation_index, overlap = max(overlaps, key=lambda item: item[1])
        if overlap >= match_iou_threshold:
            remaining.remove(annotation_index)
            matches[detection_index] = (annotation_index, float(overlap))
    return matches


def crop_tensor(
    image: Image.Image,
    detection: Detection,
    *,
    crop_margin_ratio: float,
    input_size: int,
) -> np.ndarray:
    original_width, original_height = image_original_size(image)
    x1, y1, x2, y2 = classifier_crop_box(
        detection,
        original_width,
        original_height,
        margin_ratio=crop_margin_ratio,
        crop_mode="box_resize",
    )
    scale_x = image.width / original_width
    scale_y = image.height / original_height
    crop = image.crop(
        (
            int(np.floor(x1 * scale_x)),
            int(np.floor(y1 * scale_y)),
            int(np.ceil(x2 * scale_x)),
            int(np.ceil(y2 * scale_y)),
        )
    )
    if crop.width <= 0 or crop.height <= 0:
        raise ValueError("detector crop is empty")
    return prepare_rgb(
        crop,
        (input_size, input_size),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        reducing_gap=1.0,
    )


def _manifest_index(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {int(row["image_id"]): row for row in rows if row.get("record_type") == "detection"}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.dataset_root.resolve(), args.annotation)
    predictions = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    manifest = _manifest_index(args.evaluation_manifest)
    output_records: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    per_image: list[dict[str, Any]] = []
    class_counts: defaultdict[int, int] = defaultdict(int)
    total_truth = total_predictions = total_matches = 0
    for record in records:
        image_id = int(record["image_id"])
        detections = select_detections(
            predictions[image_id],
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_threshold,
            maximum_aspect_ratio=args.maximum_aspect_ratio,
        )
        matches = match_detections(
            detections,
            record["annotations"],
            match_iou_threshold=args.match_iou_threshold,
        )
        total_truth += len(record["annotations"])
        total_predictions += len(detections)
        total_matches += len(matches)
        manifest_row = manifest[image_id]
        if args.jpeg_draft_size is None:
            with Image.open(record["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            image = decode_image(
                record["image_path"].read_bytes(),
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=args.jpeg_draft_size,
            )
        try:
            for detection_index, detection in enumerate(detections):
                if detection_index not in matches and not getattr(args, "include_unmatched", False):
                    continue
                if detection_index in matches:
                    annotation_index, overlap = matches[detection_index]
                    target = int(record["annotations"][annotation_index]["category_id"]) - 1
                else:
                    overlap = 0.0
                    target = -1
                tensors.append(
                    crop_tensor(
                        image,
                        detection,
                        crop_margin_ratio=args.crop_margin_ratio,
                        input_size=args.input_size,
                    )
                )
                if target >= 0:
                    class_counts[target] += 1
                output_records.append(
                    {
                        "tensor_index": len(tensors) - 1,
                        "image_id": image_id,
                        "fold": int(manifest_row["fold"]),
                        "group_id": str(manifest_row["capture_session_id"]),
                        "detection_index": detection_index,
                        "target": target,
                        "match_iou": overlap,
                    }
                )
        finally:
            image.close()
        per_image.append(
            {
                "image_id": image_id,
                "ground_truth_count": len(record["annotations"]),
                "prediction_count": len(detections),
                "matched_count": len(matches),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "evaluation_tensors.npy", np.stack(tensors).astype(np.float32))
    (args.output_dir / "evaluation_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_records),
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "detected_roi_dataset",
        "annotation": args.annotation,
        "prediction_source": args.predictions.name,
        "score_threshold": args.score_threshold,
        "nms_iou_threshold": args.nms_threshold,
        "match_iou_threshold": args.match_iou_threshold,
        "jpeg_draft_size": args.jpeg_draft_size,
        "unmatched_predictions_included": getattr(args, "include_unmatched", False),
        "image_count": len(records),
        "ground_truth_count": total_truth,
        "prediction_count": total_predictions,
        "matched_count": total_matches,
        "recall": total_matches / total_truth,
        "precision": total_matches / total_predictions,
        "matched_class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "per_image": per_image,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_image"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare matched classifier ROIs from detections")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", default="multi_object_instances.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.485)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--maximum-aspect-ratio", type=float, default=5.0)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.05)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--jpeg-draft-size", type=int)
    parser.add_argument("--include-unmatched", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
