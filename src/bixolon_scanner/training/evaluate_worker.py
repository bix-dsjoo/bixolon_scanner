from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ItemStatus, Status
from ..inference import build_onnx_adapters
from ..imaging import decode_image
from ..package import load_model_package
from ..pipeline import DecisionPipeline
from .calibration import binomial_rate_upper_bound
from .data import read_manifest
from .evaluate import wilson_interval
from .evaluate_detector import _iou, _xywh_to_xyxy


@dataclass
class RecordingDetector:
    detector: Any
    last_result: Any = None

    @property
    def version(self) -> str:
        return self.detector.version

    def detect(self, image: Any):
        self.last_result = self.detector.detect(image)
        return self.last_result


@dataclass
class RecordingClassifier:
    classifier: Any
    last_logits: np.ndarray | None = None

    @property
    def version(self) -> str:
        return self.classifier.version

    def classify(self, image: Any, detections: Any) -> np.ndarray:
        self.last_logits = self.classifier.classify(image, detections)
        return self.last_logits


def evaluate(args: argparse.Namespace) -> None:
    package = load_model_package(args.package_dir)
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
    )
    records = [
        record
        for record in read_manifest(args.manifest)
        if record["record_type"] == "detection"
        and (
            args.mode == "test"
            and record["split"] == "test"
            or args.mode == "validation"
            and record["split"] == "development"
            and record["fold"] == args.fold
        )
    ]
    ground_truth_count = 0
    detection_count = 0
    matched_count = 0
    count_correct = 0
    recapture_count = 0
    capacity_saturated_count = 0
    approved_count = 0
    approved_correct = 0
    approved_unmatched = 0
    classified_detection_count = 0
    classified_matched_count = 0
    matched_top1_correct = 0
    matched_top3_correct = 0
    unknown_count = 0
    unknown_matched_count = 0
    unknown_top3_correct = 0
    image_approved_count = 0
    image_approved_correct = 0

    for record in records:
        image = decode_image(
            (args.dataset_root / record["image_path"]).read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        recording_classifier.last_logits = None
        response = pipeline.scan(image, request_id=f"evaluation-{record['image_id']}")
        detection_result = recording_detector.last_result
        if detection_result is None:
            raise RuntimeError("detector result was not recorded")
        detections = sorted(
            detection_result.detections, key=lambda value: (value.y1, value.x1)
        )
        is_recapture = response.status is Status.RECAPTURE
        recapture_count += int(is_recapture)
        capacity_saturated_count += int(detection_result.capacity_saturated)
        gt_boxes = [_xywh_to_xyxy(item["bbox_xywh"]) for item in record["annotations"]]
        remaining_gt = set(range(len(gt_boxes)))
        detection_to_gt: dict[int, int] = {}
        for detection_index, detection in sorted(
            enumerate(detections), key=lambda item: item[1].score, reverse=True
        ):
            box = np.asarray(
                [detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32
            )
            candidates = [(index, _iou(box, gt_boxes[index])) for index in remaining_gt]
            if candidates:
                gt_index, overlap = max(candidates, key=lambda item: item[1])
                if overlap >= args.match_iou_threshold:
                    remaining_gt.remove(gt_index)
                    detection_to_gt[detection_index] = gt_index

        ground_truth_count += len(gt_boxes)
        detection_count += len(detections)
        matched_count += len(detection_to_gt)
        count_correct += int(len(detections) == len(gt_boxes))
        if is_recapture or not detections:
            continue
        if len(response.items) != len(detections):
            raise RuntimeError("Worker item count does not match detector output")
        if recording_classifier.last_logits is None:
            raise RuntimeError("classifier was not recorded on a non-recapture path")
        ranks = np.argsort(
            -recording_classifier.last_logits, axis=1, kind="stable"
        )
        classified_detection_count += len(detections)
        image_correct = len(detections) == len(gt_boxes)
        for detection_index, item in enumerate(response.items):
            is_approved = item.status is ItemStatus.APPROVED
            approved_count += int(is_approved)
            unknown_count += int(not is_approved)
            gt_index = detection_to_gt.get(detection_index)
            if gt_index is None:
                approved_unmatched += int(is_approved)
                image_correct = False
                continue
            classified_matched_count += 1
            target = int(record["annotations"][gt_index]["category_id"]) - 1
            indices = ranks[detection_index]
            top1_correct = int(indices[0]) == target
            top3_correct = target in {int(value) for value in indices[:3]}
            matched_top1_correct += int(top1_correct)
            matched_top3_correct += int(top3_correct)
            if is_approved:
                approved_correct += int(top1_correct)
                image_correct = image_correct and top1_correct
            else:
                unknown_matched_count += 1
                unknown_top3_correct += int(top3_correct)
        if response.status is Status.APPROVED:
            image_approved_count += 1
            image_approved_correct += int(image_correct)

    detection_recall = matched_count / ground_truth_count if ground_truth_count else 0.0
    detection_precision = matched_count / detection_count if detection_count else 0.0
    approved_precision = approved_correct / approved_count if approved_count else 1.0
    unknown_top3_gate_satisfied = (
        unknown_matched_count > 0
        and unknown_top3_correct / unknown_matched_count >= 0.95
    )
    unknown_top3_waived = bool(
        package.metadata.promotion
        and any(
            waiver.gate == "unknown_top3_accuracy"
            for waiver in package.metadata.promotion.waivers
        )
    )
    report = {
        "mode": args.mode,
        "fold": args.fold if args.mode == "validation" else None,
        "threshold_policy": "fixed-from-package-oof-metadata",
        "package_version": package.metadata.package_version,
        "dataset_version": package.metadata.dataset_version,
        "promotion_status": package.metadata.promotion_status,
        "promotion_decision": (
            package.metadata.promotion.model_dump(mode="json")
            if package.metadata.promotion
            else None
        ),
        "provider": provider,
        "image_count": len(records),
        "detector": {
            "ground_truth_count": ground_truth_count,
            "prediction_count": detection_count,
            "matched_count": matched_count,
            "recall": detection_recall,
            "precision": detection_precision,
            "count_accuracy": count_correct / len(records) if records else 0.0,
            "capacity_saturated_images": capacity_saturated_count,
            "recall_gate_satisfied": detection_recall >= 0.99,
        },
        "classifier_on_detector_crops": {
            "matched_sample_count": classified_matched_count,
            "detector_matched_sample_count": matched_count,
            "recapture_skipped_matched_sample_count": (
                matched_count - classified_matched_count
            ),
            "overall_top1_accuracy": (
                matched_top1_correct / classified_matched_count
                if classified_matched_count
                else 0.0
            ),
            "overall_top3_accuracy": (
                matched_top3_correct / classified_matched_count
                if classified_matched_count
                else 0.0
            ),
            "approved_count": approved_count,
            "approved_correct": approved_correct,
            "approved_unmatched": approved_unmatched,
            "approval_coverage_of_detections": approved_count / detection_count if detection_count else 0.0,
            "approval_coverage_of_classified_detections": (
                approved_count / classified_detection_count
                if classified_detection_count
                else 0.0
            ),
            "approved_precision": approved_precision,
            "approved_precision_95ci": list(wilson_interval(approved_correct, approved_count)),
            "approved_false_rate_upper_95": binomial_rate_upper_bound(
                approved_count - approved_correct, approved_count
            ),
            "approved_precision_gate_satisfied": approved_precision >= 0.995,
            "unknown_count": unknown_count,
            "unknown_matched_count": unknown_matched_count,
            "unknown_top3_accuracy": (
                unknown_top3_correct / unknown_matched_count
                if unknown_matched_count
                else None
            ),
            "unknown_top3_gate_satisfied": unknown_top3_gate_satisfied,
            "unknown_top3_promotion_disposition": (
                "passed" if unknown_top3_gate_satisfied else "waived" if unknown_top3_waived else "failed"
            ),
        },
        "frame_policy": {
            "recapture_count": recapture_count,
            "approved_image_count": image_approved_count,
            "approved_image_correct": image_approved_correct,
            "approved_image_precision": (
                image_approved_correct / image_approved_count if image_approved_count else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the packaged ONNX detector-crop pipeline")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("validation", "test"), default="test")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
