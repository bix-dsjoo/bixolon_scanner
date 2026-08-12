from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..inference import Detection, OnnxDetector, OrtRunner, _nms, _prepare_rgb, _sigmoid
from ..package import load_model_package, sha256_file
from ..pipeline import _softmax
from .models import build_dino_classifier, require_torch


def _detector_checkpoint_sha256(checkpoint: Path) -> str:
    candidates = [
        path
        for path in (
            checkpoint / "model.safetensors",
            checkpoint / "pytorch_model.bin",
        )
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError("detector checkpoint weights are missing or ambiguous")
    return sha256_file(candidates[0])


def _postprocess_detector(logits, boxes, metadata, image_shape) -> list[Detection]:
    height, width = image_shape[:2]
    scores = _sigmoid(np.asarray(logits)[0]).max(axis=-1)
    detections = []
    for index in np.flatnonzero(scores >= metadata.score_threshold):
        cx, cy, box_width, box_height = [float(value) for value in boxes[0, index]]
        detections.append(
            Detection(
                max(0.0, (cx - box_width / 2.0) * width),
                max(0.0, (cy - box_height / 2.0) * height),
                min(float(width), (cx + box_width / 2.0) * width),
                min(float(height), (cy + box_height / 2.0) * height),
                float(scores[index]),
            )
        )
    return _nms(detections, metadata.nms_iou_threshold)


def _matched_detection_errors(reference, candidate, image_shape):
    if len(reference) != len(candidate):
        return 0.0, float("inf"), float("inf")
    height, width = image_shape[:2]
    remaining = set(range(len(candidate)))
    overlaps = []
    coordinate_errors = []
    score_errors = []
    for left in reference:
        best_index = max(
            remaining,
            key=lambda index: _detection_iou(left, candidate[index]),
        )
        right = candidate[best_index]
        remaining.remove(best_index)
        overlaps.append(_detection_iou(left, right))
        coordinate_errors.extend(
            [
                abs(left.x1 - right.x1) / width,
                abs(left.x2 - right.x2) / width,
                abs(left.y1 - right.y1) / height,
                abs(left.y2 - right.y2) / height,
            ]
        )
        score_errors.append(abs(left.score - right.score))
    return min(overlaps), max(coordinate_errors), max(score_errors)


def _detection_iou(left: Detection, right: Detection) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _classifier_batch(image: np.ndarray, detections, metadata) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    crops = []
    for detection in detections:
        margin_x = (detection.x2 - detection.x1) * metadata.crop_margin_ratio
        margin_y = (detection.y2 - detection.y1) * metadata.crop_margin_ratio
        x1 = max(0, int(np.floor(detection.x1 - margin_x)))
        y1 = max(0, int(np.floor(detection.y1 - margin_y)))
        x2 = min(image_width, int(np.ceil(detection.x2 + margin_x)))
        y2 = min(image_height, int(np.ceil(detection.y2 + margin_y)))
        crops.append(
            _prepare_rgb(
                image[y1:y2, x1:x2],
                metadata.input_size,
                metadata.mean,
                metadata.std,
                reducing_gap=metadata.resize_reducing_gap,
            )
        )
    return np.stack(crops).astype(np.float32, copy=False)


def parity(args: argparse.Namespace) -> None:
    if not math.isfinite(args.detector_min_iou) or not 0.0 <= args.detector_min_iou <= 1.0:
        raise ValueError("detector_min_iou must be finite and in [0, 1]")
    for name in (
        "detector_coordinate_tolerance",
        "detector_score_tolerance",
        "classifier_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    torch = require_torch()
    from transformers import RTDetrV2ForObjectDetection

    package = load_model_package(args.package_dir)
    metadata = package.metadata
    provider = "cpu" if args.cpu else "cuda"
    device = torch.device(provider)
    with Image.open(args.image) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)

    detector_tensor = _prepare_rgb(
        image,
        metadata.detector.input_size,
        metadata.detector.mean,
        metadata.detector.std,
        reducing_gap=metadata.detector.resize_reducing_gap,
    )[None].astype(np.float32, copy=False)
    detector_reference = RTDetrV2ForObjectDetection.from_pretrained(
        args.detector_checkpoint
    ).to(device).eval()
    with torch.inference_mode():
        reference_output = detector_reference(
            pixel_values=torch.from_numpy(detector_tensor).to(device)
        )
    reference_logits = reference_output.logits.float().cpu().numpy()
    reference_boxes = reference_output.pred_boxes.float().cpu().numpy()
    detector_runner = OrtRunner(package.detector_path, provider, args.cuda_dll_dir)
    onnx_logits, onnx_boxes = detector_runner.run(
        [metadata.detector.logits_output, metadata.detector.boxes_output],
        metadata.detector.input_name,
        detector_tensor,
    )
    reference_detections = _postprocess_detector(
        reference_logits, reference_boxes, metadata.detector, image.shape
    )
    onnx_detections = _postprocess_detector(
        onnx_logits, onnx_boxes, metadata.detector, image.shape
    )
    minimum_iou, box_error, score_error = _matched_detection_errors(
        reference_detections, onnx_detections, image.shape
    )
    detector = OnnxDetector(
        package.detector_path, metadata.detector, provider, args.cuda_dll_dir
    )
    detection_result = detector.detect(image)
    if not detection_result.detections:
        raise RuntimeError("parity image produced no detections")

    checkpoint = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=False)
    classifier_reference = build_dino_classifier(
        checkpoint["backbone_kind"],
        checkpoint["num_classes"],
        pretrained_name=checkpoint["pretrained_name"],
        hub_repository=f"facebookresearch/dinov3:{checkpoint['source_revision']}",
    )
    classifier_reference.load_state_dict(checkpoint["state_dict"])
    classifier_reference.to(device).eval()
    classifier_tensor = _classifier_batch(
        image, detection_result.detections, metadata.classifier
    )
    with torch.inference_mode():
        classifier_reference_logits = (
            classifier_reference(torch.from_numpy(classifier_tensor).to(device)).float().cpu().numpy()
        )
    classifier_runner = OrtRunner(package.classifier_path, provider, args.cuda_dll_dir)
    (classifier_onnx_logits,) = classifier_runner.run(
        [metadata.classifier.logits_output],
        metadata.classifier.input_name,
        classifier_tensor,
    )
    reference_probabilities = _softmax(
        classifier_reference_logits, metadata.classifier.temperature
    )
    onnx_probabilities = _softmax(
        np.asarray(classifier_onnx_logits), metadata.classifier.temperature
    )
    reference_ranks = np.argsort(-reference_probabilities, axis=1)[:, :3]
    onnx_ranks = np.argsort(-onnx_probabilities, axis=1)[:, :3]
    reference_status = (
        reference_probabilities.max(axis=1) >= metadata.classifier.approval_threshold
    )
    onnx_status = onnx_probabilities.max(axis=1) >= metadata.classifier.approval_threshold

    detector_logits_error = float(np.max(np.abs(reference_logits - onnx_logits)))
    detector_boxes_error = float(np.max(np.abs(reference_boxes - onnx_boxes)))
    classifier_logits_error = float(
        np.max(np.abs(classifier_reference_logits - classifier_onnx_logits))
    )
    report = {
        "package_version": metadata.package_version,
        "dataset_version": metadata.dataset_version,
        "detector_version": metadata.detector.version,
        "classifier_version": metadata.classifier.version,
        "detector_checkpoint_sha256": _detector_checkpoint_sha256(
            args.detector_checkpoint
        ),
        "classifier_checkpoint_sha256": sha256_file(args.classifier_checkpoint),
        "package_artifact_sha256": {
            "metadata.json": sha256_file(args.package_dir / "metadata.json"),
            metadata.detector.filename: sha256_file(package.detector_path),
            metadata.classifier.filename: sha256_file(package.classifier_path),
        },
        "provider": provider,
        "image": args.image.name,
        "item_count": len(detection_result.detections),
        "detector": {
            "raw_query_logits_max_abs_error_diagnostic": detector_logits_error,
            "raw_query_boxes_max_abs_error_diagnostic": detector_boxes_error,
            "reference_count": len(reference_detections),
            "onnx_count": len(onnx_detections),
            "matched_min_iou": minimum_iou,
            "normalized_coordinate_max_abs_error": box_error,
            "score_max_abs_error": score_error,
            "minimum_iou_tolerance": args.detector_min_iou,
            "coordinate_tolerance": args.detector_coordinate_tolerance,
            "score_tolerance": args.detector_score_tolerance,
            "passes": len(reference_detections) == len(onnx_detections)
            and minimum_iou >= args.detector_min_iou
            and box_error <= args.detector_coordinate_tolerance
            and score_error <= args.detector_score_tolerance,
        },
        "classifier": {
            "logits_max_abs_error": classifier_logits_error,
            "tolerance": args.classifier_tolerance,
            "top3_equal": bool(np.array_equal(reference_ranks, onnx_ranks)),
            "status_equal": bool(np.array_equal(reference_status, onnx_status)),
            "passes": classifier_logits_error <= args.classifier_tolerance
            and bool(np.array_equal(reference_ranks, onnx_ranks))
            and bool(np.array_equal(reference_status, onnx_status)),
        },
    }
    report["passes"] = report["detector"]["passes"] and report["classifier"]["passes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passes"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare packaged ONNX outputs with PyTorch")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--detector-min-iou", type=float, default=0.99)
    parser.add_argument("--detector-coordinate-tolerance", type=float, default=0.01)
    parser.add_argument("--detector-score-tolerance", type=float, default=0.02)
    parser.add_argument("--classifier-tolerance", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true")
    parity(parser.parse_args())


if __name__ == "__main__":
    main()
