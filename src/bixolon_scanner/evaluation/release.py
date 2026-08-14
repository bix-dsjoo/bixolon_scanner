from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ItemStatus, Status
from ..contracts.model_package import load_model_package
from ..pipeline import DecisionPipeline
from ..pipeline.ports import Detection, DetectionResult
from ..runtime.imaging import decode_image
from ..runtime.onnx import build_onnx_adapters
from ..training.bread_dataset import audit_bread_dataset, audit_bread_evaluation_set
from ..training.calibration import binomial_rate_upper_bound
from .detector import _iou, _xywh_to_xyxy


@dataclass
class RecordingDetector:
    detector: Any
    last_result: DetectionResult | None = None
    last_ms: float = 0.0

    @property
    def version(self) -> str:
        return self.detector.version

    def detect(self, image: Any) -> DetectionResult:
        started = time.perf_counter()
        self.last_result = self.detector.detect(image)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return self.last_result


@dataclass
class RecordingClassifier:
    classifier: Any
    last_ms: float = 0.0

    @property
    def version(self) -> str:
        return self.classifier.version

    def classify(self, image: Any, detections: list[Detection]) -> Any:
        started = time.perf_counter()
        result = self.classifier.classify(image, detections)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return result


@dataclass
class Counts:
    images: int = 0
    ground_truth: int = 0
    predictions: int = 0
    matched: int = 0
    image_recapture: int = 0
    segment_recapture: int = 0
    classified_matched: int = 0
    top1_correct: int = 0
    recognized_correct: int = 0
    approved: int = 0
    approved_correct: int = 0
    approved_wrong: int = 0
    approved_classification_wrong: int = 0
    approved_false_segmentation: int = 0
    unknown: int = 0
    unknown_matched: int = 0
    unknown_top3_correct: int = 0
    exact_segmentation_images: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    scored_predictions: list[tuple[float, bool]] = field(default_factory=list)


def _load_coco_records(
    root: Path, annotation_name: str, *, dataset_name: str
) -> list[dict[str, Any]]:
    annotation_path = root / "annotations" / annotation_name
    coco = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    records: list[dict[str, Any]] = []
    for image in sorted(coco["images"], key=lambda row: int(row["id"])):
        image_path = (annotation_path.parent / str(image["file_name"])).resolve()
        image_path.relative_to(root)
        relative = image_path.relative_to(root)
        difficulty = (
            relative.parts[1].upper()
            if dataset_name == "multi_object_scenes" and len(relative.parts) > 1
            else "SCAN_LOG"
        )
        records.append(
            {
                "dataset": dataset_name,
                "difficulty": difficulty,
                "image_id": int(image["id"]),
                "image_path": image_path,
                "expected_image_status": str(image.get("status", "ANNOTATED")),
                "expected_reason_codes": [str(value) for value in image.get("reason_codes", [])],
                "annotations": sorted(
                    annotations_by_image.get(int(image["id"]), []),
                    key=lambda row: int(row["id"]),
                ),
            }
        )
    return records


def _match(
    detections: list[Detection], annotations: list[dict[str, Any]], threshold: float
) -> tuple[dict[int, int], set[int]]:
    gt_boxes = [_xywh_to_xyxy(row["bbox"]) for row in annotations]
    remaining = set(range(len(gt_boxes)))
    matches: dict[int, int] = {}
    for detection_index, detection in sorted(
        enumerate(detections), key=lambda item: item[1].score, reverse=True
    ):
        box = np.asarray([detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32)
        candidates = [(index, _iou(box, gt_boxes[index])) for index in remaining]
        if candidates:
            gt_index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= threshold:
                matches[detection_index] = gt_index
                remaining.remove(gt_index)
    return matches, remaining


def nearest_iou(box: np.ndarray, candidates: list[np.ndarray]) -> float:
    return max((_iou(box, candidate) for candidate in candidates), default=0.0)


def average_precision(scored_predictions: list[tuple[float, bool]], ground_truth: int) -> float:
    if ground_truth <= 0:
        return 0.0
    ordered = sorted(scored_predictions, key=lambda row: row[0], reverse=True)
    if not ordered:
        return 0.0
    true_positive = np.cumsum([int(row[1]) for row in ordered], dtype=np.float64)
    false_positive = np.cumsum([int(not row[1]) for row in ordered], dtype=np.float64)
    recall = true_positive / ground_truth
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changing = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changing + 1] - recall[changing]) * precision[changing + 1]))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return (
        float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else None
    )


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _metrics(counts: Counts) -> dict[str, Any]:
    unresolved = counts.ground_truth - counts.recognized_correct
    return {
        "image_count": counts.images,
        "ground_truth_segmentation_count": counts.ground_truth,
        "predicted_segmentation_count": counts.predictions,
        "matched_segmentation_count": counts.matched,
        "segmentation_recall_iou50": _rate(counts.matched, counts.ground_truth),
        "segmentation_precision_iou50": _rate(counts.matched, counts.predictions),
        "segmentation_ap50": average_precision(counts.scored_predictions, counts.ground_truth),
        "exact_segmentation_image_rate": _rate(counts.exact_segmentation_images, counts.images),
        "recognition_accuracy_all_ground_truth": _rate(
            counts.recognized_correct, counts.ground_truth
        ),
        "recognition_error_count_all_ground_truth": unresolved,
        "classifier_top1_accuracy_on_resolved_matches": _rate(
            counts.top1_correct, counts.classified_matched
        ),
        "approved_count": counts.approved,
        "approved_correct": counts.approved_correct,
        "approved_misrecognition_count": counts.approved_wrong,
        "approved_classification_misrecognition_count": counts.approved_classification_wrong,
        "approved_false_segmentation_count": counts.approved_false_segmentation,
        "approved_misrecognition_rate": _rate(counts.approved_wrong, counts.approved),
        "approved_misrecognition_rate_upper_95": binomial_rate_upper_bound(
            counts.approved_wrong, counts.approved
        ),
        "unknown_count": counts.unknown,
        "unknown_top3_accuracy": _rate(counts.unknown_top3_correct, counts.unknown_matched),
        "image_recapture_count": counts.image_recapture,
        "image_recapture_rate": _rate(counts.image_recapture, counts.images),
        "segment_recapture_count": counts.segment_recapture,
        "segment_recapture_rate_all_ground_truth": _rate(
            counts.segment_recapture, counts.ground_truth
        ),
        "resolved_match_coverage": _rate(counts.classified_matched, counts.ground_truth),
        "latency_ms": _latency_summary(counts.latencies_ms),
    }


def minimum_zero_error_samples(maximum_rate: float, confidence_level: float = 0.95) -> int:
    if not 0 < maximum_rate < 1 or not 0 < confidence_level < 1:
        raise ValueError("rates must be strictly between zero and one")
    return math.ceil(math.log(1.0 - confidence_level) / math.log(1.0 - maximum_rate))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    gate_dataset = getattr(args, "gate_dataset", "all_annotated")
    dataset_metadata_path = getattr(args, "dataset_metadata", None)
    if gate_dataset == "multi_object_scenes" and dataset_metadata_path is not None:
        dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
        expected = dataset_metadata["evaluation_sets"]["multi_object_scenes"]
        actual = audit_bread_evaluation_set(dataset_root, "multi_object_scenes")
        for field in (
            "annotation_sha256",
            "image_content_sha256",
            "image_count",
            "annotation_count",
        ):
            if actual[field] != expected[field]:
                raise ValueError(f"locked multi_object_scenes {field} mismatch")
    else:
        _, dataset_metadata = audit_bread_dataset(dataset_root)
    package = load_model_package(args.package_dir)
    jpeg_draft_size = getattr(args, "jpeg_draft_size", None)
    if jpeg_draft_size is None:
        jpeg_draft_size = package.metadata.input.jpeg_draft_size
    approval_threshold = getattr(args, "approval_threshold", None)
    if approval_threshold is not None:
        if not 0.0 <= approval_threshold <= 1.0:
            raise ValueError("approval threshold must be between zero and one")
        package.metadata.classifier.approval_threshold = approval_threshold
    detector, classifier, provider = build_onnx_adapters(
        package, args.provider, cuda_dll_dir=args.cuda_dll_dir
    )
    recording_detector = RecordingDetector(detector)
    recording_classifier = RecordingClassifier(classifier)
    pipeline = DecisionPipeline(
        recording_detector,
        recording_classifier,
        package.metadata.classifier,
        package.metadata.quality,
        package.metadata.count_verifier,
        worker_version=package.metadata.worker_version,
    )
    multi_records = _load_coco_records(
        dataset_root,
        "multi_object_instances.json",
        dataset_name="multi_object_scenes",
    )
    if gate_dataset == "multi_object_scenes":
        records = multi_records
    else:
        scan_records = _load_coco_records(
            dataset_root,
            "scan_log_instances.json",
            dataset_name="scan_log_samples",
        )
        records = multi_records + scan_records
    if records and args.warmup_count:
        warmup_encoded = records[0]["image_path"].read_bytes()
        warmup_image = decode_image(
            warmup_encoded,
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=jpeg_draft_size,
        )
        for index in range(args.warmup_count):
            pipeline.scan(warmup_image, request_id=f"release-warmup-{index + 1:04d}")
    buckets: dict[str, Counts] = {"ALL_ANNOTATED": Counts(), "MULTI_ALL": Counts()}
    all_latencies: list[float] = []
    full_path_latencies: list[float] = []
    early_exit_latencies: list[float] = []
    stage_latencies: dict[str, list[float]] = {
        "decode": [],
        "detector": [],
        "classifier": [],
        "decision_overhead": [],
    }
    recapture_expected = 0
    recapture_detected = 0
    recapture_by_reason: dict[str, dict[str, int]] = {}
    annotated_scan_images = 0
    annotated_scan_recaptured = 0
    details: list[dict[str, Any]] = []

    for ordinal, record in enumerate(records, start=1):
        encoded = record["image_path"].read_bytes()
        started = time.perf_counter()
        decode_started = time.perf_counter()
        image = decode_image(
            encoded,
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=jpeg_draft_size,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        response = pipeline.scan(image, request_id=f"release-eval-{ordinal:06d}")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stage_latencies["decode"].append(decode_ms)
        stage_latencies["detector"].append(recording_detector.last_ms)
        if response.classifier_version is not None:
            stage_latencies["classifier"].append(recording_classifier.last_ms)
            stage_latencies["decision_overhead"].append(
                max(
                    0.0,
                    elapsed_ms
                    - decode_ms
                    - recording_detector.last_ms
                    - recording_classifier.last_ms,
                )
            )
        all_latencies.append(elapsed_ms)
        if response.classifier_version is None:
            early_exit_latencies.append(elapsed_ms)
        else:
            full_path_latencies.append(elapsed_ms)
        result = recording_detector.last_result
        if result is None:
            raise RuntimeError("detector result was not recorded")
        is_expected_recapture = record["expected_image_status"] == "RECAPTURE"
        is_image_recapture = response.status is Status.IMAGE_RECAPTURE
        if is_expected_recapture:
            recapture_expected += 1
            recapture_detected += int(is_image_recapture)
            for reason in record["expected_reason_codes"]:
                reason_counts = recapture_by_reason.setdefault(
                    reason, {"expected": 0, "image_recapture": 0, "reason_matched": 0}
                )
                reason_counts["expected"] += 1
                reason_counts["image_recapture"] += int(is_image_recapture)
                reason_counts["reason_matched"] += int(reason in response.reason_codes)
            details.append(
                {
                    "dataset": record["dataset"],
                    "difficulty": record["difficulty"],
                    "image": record["image_path"].relative_to(dataset_root).as_posix(),
                    "expected": "IMAGE_RECAPTURE",
                    "expected_reason_codes": "|".join(record["expected_reason_codes"]),
                    "actual": response.status.value,
                    "reason_codes": "|".join(response.reason_codes),
                    "error": "" if is_image_recapture else "MISSED_IMAGE_RECAPTURE",
                }
            )
            continue

        if record["dataset"] == "scan_log_samples":
            annotated_scan_images += 1
            annotated_scan_recaptured += int(is_image_recapture)
        selected = [buckets["ALL_ANNOTATED"]]
        if record["dataset"] == "multi_object_scenes":
            selected.extend(
                [
                    buckets["MULTI_ALL"],
                    buckets.setdefault(f"MULTI_{record['difficulty']}", Counts()),
                ]
            )
        else:
            selected.append(buckets.setdefault("SCAN_LOG_ANNOTATED", Counts()))

        detections = sorted(result.detections, key=lambda value: (value.y1, value.x1))
        matches, missed = _match(detections, record["annotations"], args.match_iou_threshold)
        ground_truth_boxes = [_xywh_to_xyxy(row["bbox"]) for row in record["annotations"]]
        exact = not missed and len(matches) == len(detections)
        for detection_index, detection in enumerate(detections):
            if detection_index not in matches:
                detection_box = np.asarray(
                    [detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32
                )
                closest_iou = nearest_iou(detection_box, ground_truth_boxes)
                details.append(
                    {
                        "dataset": record["dataset"],
                        "difficulty": record["difficulty"],
                        "image": record["image_path"].relative_to(dataset_root).as_posix(),
                        "expected": "NO_SEGMENTATION",
                        "actual": "DETECTOR_PREDICTION",
                        "reason_codes": "",
                        "error": "DETECTOR_FALSE_POSITIVE",
                        "bbox": "|".join(
                            f"{value:.2f}"
                            for value in (detection.x1, detection.y1, detection.x2, detection.y2)
                        ),
                        "score": f"{detection.score:.8f}",
                        "nearest_iou": f"{closest_iou:.8f}",
                    }
                )
        for gt_index in sorted(missed):
            annotation = record["annotations"][gt_index]
            ground_truth_box = ground_truth_boxes[gt_index]
            closest_iou = nearest_iou(
                ground_truth_box,
                [
                    np.asarray(
                        [detection.x1, detection.y1, detection.x2, detection.y2],
                        dtype=np.float32,
                    )
                    for detection in detections
                ],
            )
            details.append(
                {
                    "dataset": record["dataset"],
                    "difficulty": record["difficulty"],
                    "image": record["image_path"].relative_to(dataset_root).as_posix(),
                    "expected": f"bread_{int(annotation['category_id']):02d}",
                    "actual": "NO_SEGMENTATION",
                    "reason_codes": "",
                    "error": "DETECTOR_MISSED_GT",
                    "bbox": "|".join(f"{float(value):.2f}" for value in annotation["bbox"]),
                    "score": "",
                    "nearest_iou": f"{closest_iou:.8f}",
                }
            )
        for counts in selected:
            counts.images += 1
            counts.ground_truth += len(record["annotations"])
            counts.predictions += len(detections)
            counts.matched += len(matches)
            counts.image_recapture += int(is_image_recapture)
            counts.exact_segmentation_images += int(exact)
            counts.latencies_ms.append(elapsed_ms)
            counts.scored_predictions.extend(
                (detection.score, index in matches) for index, detection in enumerate(detections)
            )
        if is_image_recapture:
            details.append(
                {
                    "dataset": record["dataset"],
                    "difficulty": record["difficulty"],
                    "image": record["image_path"].relative_to(dataset_root).as_posix(),
                    "expected": "SEGMENTATION",
                    "actual": response.status.value,
                    "reason_codes": "|".join(response.reason_codes),
                    "error": "FALSE_IMAGE_RECAPTURE",
                }
            )
            continue
        if len(response.segmentations) != len(detections):
            raise RuntimeError("Worker segmentation count does not match detector output")
        for index, segmentation in enumerate(response.segmentations):
            gt_index = matches.get(index)
            if segmentation.status is ItemStatus.SEGMENT_RECAPTURE:
                for counts in selected:
                    counts.segment_recapture += 1
                continue
            if segmentation.status is ItemStatus.APPROVED:
                for counts in selected:
                    counts.approved += 1
                predicted = segmentation.prediction.class_id
                top3: list[str] = []
            else:
                for counts in selected:
                    counts.unknown += 1
                predicted = segmentation.top3[0].class_id
                top3 = [candidate.class_id for candidate in segmentation.top3]
            if gt_index is None:
                if segmentation.status is ItemStatus.APPROVED:
                    for counts in selected:
                        counts.approved_wrong += 1
                        counts.approved_false_segmentation += 1
                detection = detections[index]
                details.append(
                    {
                        "dataset": record["dataset"],
                        "difficulty": record["difficulty"],
                        "image": record["image_path"].relative_to(dataset_root).as_posix(),
                        "expected": "NO_SEGMENTATION",
                        "actual": predicted,
                        "reason_codes": "",
                        "error": (
                            "APPROVED_FALSE_SEGMENTATION"
                            if segmentation.status is ItemStatus.APPROVED
                            else "UNKNOWN_FALSE_SEGMENTATION"
                        ),
                        "bbox": "|".join(
                            f"{value:.2f}"
                            for value in (
                                detection.x1,
                                detection.y1,
                                detection.x2,
                                detection.y2,
                            )
                        ),
                        "score": f"{detection.score:.8f}",
                        "top3": "|".join(top3),
                        "confidence": f"{segmentation.confidence:.8f}",
                    }
                )
                continue
            expected = f"bread_{int(record['annotations'][gt_index]['category_id']):02d}"
            correct = predicted == expected
            recognized = correct if segmentation.status is ItemStatus.APPROVED else expected in top3
            for counts in selected:
                counts.classified_matched += 1
                counts.top1_correct += int(correct)
                counts.recognized_correct += int(recognized)
                if segmentation.status is ItemStatus.APPROVED:
                    counts.approved_correct += int(correct)
                    counts.approved_wrong += int(not correct)
                    counts.approved_classification_wrong += int(not correct)
                else:
                    counts.unknown_matched += 1
                    counts.unknown_top3_correct += int(expected in top3)
            detection = detections[index]
            details.append(
                {
                    "dataset": record["dataset"],
                    "difficulty": record["difficulty"],
                    "image": record["image_path"].relative_to(dataset_root).as_posix(),
                    "expected": expected,
                    "actual": predicted,
                    "reason_codes": "",
                    "error": (
                        ""
                        if correct
                        else (
                            "APPROVED_MISRECOGNITION"
                            if segmentation.status is ItemStatus.APPROVED
                            else "UNKNOWN_TOP1_MISRECOGNITION"
                        )
                    ),
                    "bbox": "|".join(
                        f"{value:.2f}"
                        for value in (
                            detection.x1,
                            detection.y1,
                            detection.x2,
                            detection.y2,
                        )
                    ),
                    "score": f"{detection.score:.8f}",
                    "top3": "|".join(top3),
                    "confidence": f"{segmentation.confidence:.8f}",
                }
            )

    metrics = {name: _metrics(counts) for name, counts in sorted(buckets.items())}
    overall = metrics["MULTI_ALL" if gate_dataset == "multi_object_scenes" else "ALL_ANNOTATED"]
    recapture_recall = _rate(recapture_detected, recapture_expected)
    false_image_recapture_rate = _rate(annotated_scan_recaptured, annotated_scan_images)
    if gate_dataset == "multi_object_scenes":
        false_image_recapture_rate = overall["image_recapture_rate"]
    performance = {
        "warmup_count": args.warmup_count,
        "scope": "decode+preprocess+detector+classifier+postprocess+decision",
        "all_images": _latency_summary(all_latencies),
        "full_path": _latency_summary(full_path_latencies),
        "early_exit": _latency_summary(early_exit_latencies),
        "stages": {name: _latency_summary(values) for name, values in stage_latencies.items()},
    }
    risk_upper = overall["approved_misrecognition_rate_upper_95"]
    required_zero_error = minimum_zero_error_samples(args.maximum_misrecognition_rate)
    checks = {
        "dataset_version_matches": package.metadata.dataset_version
        == dataset_metadata["dataset_version"],
        "official_version_floor": not (
            package.metadata.worker_version.startswith("0.")
            or package.metadata.detector.version.startswith("0.")
            or package.metadata.classifier.version.startswith("0.")
        ),
        "recognition_accuracy": float(overall["recognition_accuracy_all_ground_truth"] or 0.0)
        >= args.minimum_recognition_accuracy,
        "approved_misrecognition_rate": float(overall["approved_misrecognition_rate"] or 0.0)
        <= args.maximum_misrecognition_rate,
        "approved_misrecognition_risk_upper_95": float(risk_upper)
        <= args.maximum_misrecognition_rate,
        "segmentation_recall": float(overall["segmentation_recall_iou50"] or 0.0)
        >= args.minimum_segmentation_recall,
        "segmentation_precision": float(overall["segmentation_precision_iou50"] or 0.0)
        >= args.minimum_segmentation_precision,
        "false_image_recapture_rate": float(false_image_recapture_rate or 0.0)
        <= args.maximum_false_recapture_rate,
        "segment_recapture_rate": float(overall["segment_recapture_rate_all_ground_truth"] or 0.0)
        <= args.maximum_false_recapture_rate,
        "mean_latency": float(performance["all_images"]["mean"] or float("inf"))
        <= args.maximum_latency_ms,
        "p95_latency": float(performance["all_images"]["p95"] or float("inf"))
        <= args.maximum_latency_ms,
    }
    if gate_dataset != "multi_object_scenes":
        checks["recapture_recall"] = float(recapture_recall or 0.0) >= args.minimum_recapture_recall
    baseline_report = None
    if args.baseline_report is not None:
        baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        baseline_false = float(baseline_report["recapture"]["false_image_recapture_rate"])
        checks["recapture_non_regression"] = float(false_image_recapture_rate or 0.0) <= (
            baseline_false + args.maximum_recapture_regression
        )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_worker_release_gate",
        "dataset_version": dataset_metadata["dataset_version"],
        "provider": provider,
        "gate_dataset": gate_dataset,
        "versions": {
            "worker_version": package.metadata.worker_version,
            "detector_version": package.metadata.detector.version,
            "classifier_version": package.metadata.classifier.version,
        },
        "effective_configuration": {
            "jpeg_draft_size": jpeg_draft_size,
            "approval_threshold": package.metadata.classifier.approval_threshold,
            "jpeg_draft_size_overridden": getattr(args, "jpeg_draft_size", None) is not None,
            "approval_threshold_overridden": approval_threshold is not None,
        },
        "targets": {
            "minimum_recognition_accuracy": args.minimum_recognition_accuracy,
            "maximum_misrecognition_rate": args.maximum_misrecognition_rate,
            "minimum_segmentation_recall": args.minimum_segmentation_recall,
            "minimum_segmentation_precision": args.minimum_segmentation_precision,
            "minimum_recapture_recall": args.minimum_recapture_recall,
            "maximum_false_recapture_rate": args.maximum_false_recapture_rate,
            "maximum_mean_and_p95_latency_ms": args.maximum_latency_ms,
        },
        "metrics": metrics,
        "performance": performance,
        "recapture": {
            "gate_applied_to_recall": gate_dataset != "multi_object_scenes",
            "expected_image_recapture_count": recapture_expected,
            "detected_image_recapture_count": recapture_detected,
            "recapture_recall": recapture_recall,
            "annotated_scan_image_count": annotated_scan_images,
            "false_image_recapture_count": annotated_scan_recaptured,
            "false_image_recapture_rate": false_image_recapture_rate,
            "by_expected_reason": {
                reason: counts
                | {
                    "image_recapture_recall": _rate(counts["image_recapture"], counts["expected"]),
                    "reason_match_recall": _rate(counts["reason_matched"], counts["expected"]),
                }
                for reason, counts in sorted(recapture_by_reason.items())
            },
        },
        "risk_evidence": {
            "approved_sample_count": overall["approved_count"],
            "observed_error_count": overall["approved_misrecognition_count"],
            "upper_95": risk_upper,
            "required_zero_error_samples": required_zero_error,
            "available_ground_truth_segments": overall["ground_truth_segmentation_count"],
            "sufficient": checks["approved_misrecognition_risk_upper_95"],
        },
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
        "promotion_status": "production" if all(checks.values()) else "experiment_only",
        "limitations": {
            "multi_object_scenes_derived_from_training_originals": True,
            "scan_logs_used_by_previous_experiments": True,
            "independent_test_set_available": False,
            "baseline_report": str(args.baseline_report) if baseline_report else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with args.details.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "dataset",
                    "difficulty",
                    "image",
                    "expected",
                    "expected_reason_codes",
                    "actual",
                    "reason_codes",
                    "error",
                    "bbox",
                    "score",
                    "nearest_iou",
                    "top3",
                    "confidence",
                ],
            )
            writer.writeheader()
            writer.writerows(details)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 1.0 bread Worker release gate")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--gate-dataset",
        choices=("all_annotated", "multi_object_scenes"),
        default="all_annotated",
    )
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument(
        "--jpeg-draft-size",
        type=int,
        help="Evaluation-only override; the packaged value remains authoritative for Worker runs",
    )
    parser.add_argument(
        "--approval-threshold",
        type=float,
        help="Evaluation-only override; the packaged value remains authoritative for Worker runs",
    )
    parser.add_argument("--minimum-recognition-accuracy", type=float, default=0.99)
    parser.add_argument("--maximum-misrecognition-rate", type=float, default=0.001)
    parser.add_argument("--minimum-segmentation-recall", type=float, default=0.99)
    parser.add_argument("--minimum-segmentation-precision", type=float, default=0.99)
    parser.add_argument("--minimum-recapture-recall", type=float, default=0.99)
    parser.add_argument("--maximum-false-recapture-rate", type=float, default=0.01)
    parser.add_argument("--maximum-recapture-regression", type=float, default=0.0)
    parser.add_argument("--maximum-latency-ms", type=float, default=100.0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
