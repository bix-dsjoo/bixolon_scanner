from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ItemStatus, Status
from ..contracts.model_package import load_model_package
from ..pipeline import DecisionPipeline
from ..pipeline.ports import Detection, DetectionResult
from ..runtime.imaging import decode_image
from ..runtime.onnx import build_onnx_adapters
from .detector import _iou, _xywh_to_xyxy

DIFFICULTY_PATTERN = re.compile(r"_(e|m|h)_", re.IGNORECASE)
DIFFICULTIES = ("E", "M", "H")


@dataclass
class RecordingDetector:
    detector: Any
    last_result: DetectionResult | None = None

    @property
    def version(self) -> str:
        return self.detector.version

    def detect(self, image: Any) -> DetectionResult:
        self.last_result = self.detector.detect(image)
        return self.last_result


def _load_records(dataset_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    root_annotation_path = dataset_root / "annotations" / "instances.json"
    if root_annotation_path.is_file():
        coco = json.loads(root_annotation_path.read_text(encoding="utf-8-sig"))
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco["annotations"]:
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image_record in sorted(coco["images"], key=lambda item: int(item["id"])):
            filename = str(image_record["file_name"])
            relative_path = Path(filename)
            directory_difficulty = (
                relative_path.parts[0].upper() if len(relative_path.parts) > 1 else None
            )
            filename_match = DIFFICULTY_PATTERN.search(relative_path.name)
            difficulty = (
                directory_difficulty
                if directory_difficulty in DIFFICULTIES
                else filename_match.group(1).upper()
                if filename_match is not None
                else None
            )
            if difficulty is None:
                raise ValueError(f"difficulty is missing from image path: {filename}")
            image_path = dataset_root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            annotations = sorted(
                annotations_by_image.get(int(image_record["id"]), []),
                key=lambda item: int(item["id"]),
            )
            if not annotations:
                raise ValueError(f"image has no annotations: {image_path}")
            records.append(
                {
                    "group": relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else dataset_root.name,
                    "difficulty": difficulty,
                    "image_path": image_path,
                    "annotations": annotations,
                }
            )
        if not records:
            raise ValueError(f"no COCO detection records found in {root_annotation_path}")
        return records

    for group_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        annotation_path = group_dir / "annotations" / "instances.json"
        image_dir = group_dir / "images"
        if not annotation_path.is_file() or not image_dir.is_dir():
            continue
        coco = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco["annotations"]:
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image_record in sorted(coco["images"], key=lambda item: int(item["id"])):
            filename = str(image_record["file_name"])
            match = DIFFICULTY_PATTERN.search(filename)
            if match is None:
                raise ValueError(f"difficulty is missing from filename: {filename}")
            image_path = image_dir / filename
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            annotations = sorted(
                annotations_by_image.get(int(image_record["id"]), []),
                key=lambda item: int(item["id"]),
            )
            if not annotations:
                raise ValueError(f"image has no annotations: {image_path}")
            records.append(
                {
                    "group": group_dir.name,
                    "difficulty": match.group(1).upper(),
                    "image_path": image_path,
                    "annotations": annotations,
                }
            )
    if not records:
        raise ValueError(f"no COCO detection groups found under {dataset_root}")
    return records


def _match_detections(
    detections: list[Detection], annotations: list[dict[str, Any]], threshold: float
) -> tuple[dict[int, tuple[int, float]], set[int]]:
    gt_boxes = [_xywh_to_xyxy(annotation["bbox"]) for annotation in annotations]
    remaining_gt = set(range(len(gt_boxes)))
    matches: dict[int, tuple[int, float]] = {}
    for detection_index, detection in sorted(
        enumerate(detections), key=lambda item: item[1].score, reverse=True
    ):
        box = np.asarray([detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32)
        candidates = [(index, _iou(box, gt_boxes[index])) for index in remaining_gt]
        if not candidates:
            continue
        gt_index, overlap = max(candidates, key=lambda item: item[1])
        if overlap >= threshold:
            remaining_gt.remove(gt_index)
            matches[detection_index] = (gt_index, overlap)
    return matches, remaining_gt


def _empty_counts() -> dict[str, Any]:
    return {
        "images": 0,
        "response_status": Counter(),
        "recapture_reasons": Counter(),
        "segment_recapture_reasons": Counter(),
        "ground_truth_boxes": 0,
        "predicted_boxes": 0,
        "matched_boxes": 0,
        "missed_boxes": 0,
        "false_positive_boxes": 0,
        "exact_detection_images": 0,
        "failed_detection_images": 0,
        "non_recapture_images": 0,
        "approved_boxes": 0,
        "approved_correct": 0,
        "approved_wrong": 0,
        "approved_wrong_matched": 0,
        "approved_wrong_unmatched": 0,
        "unknown_boxes": 0,
        "unknown_matched_boxes": 0,
        "unknown_top3_correct": 0,
        "unknown_top3_missing": 0,
        "unknown_unmatched": 0,
        "segment_recapture_boxes": 0,
        "segment_recapture_matched_boxes": 0,
        "segment_recapture_unmatched_boxes": 0,
        "classified_matched_boxes": 0,
        "top1_correct": 0,
        "recapture_ground_truth_boxes": 0,
        "unblocked_missed_boxes": 0,
        "unblocked_false_positive_boxes": 0,
        "end_to_end_latency_ms_total": 0.0,
        "end_to_end_latency_ms_samples": [],
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _finalize_counts(counts: dict[str, Any]) -> dict[str, Any]:
    result = dict(counts)
    latency_samples = np.asarray(counts["end_to_end_latency_ms_samples"], dtype=np.float64)
    result.pop("end_to_end_latency_ms_samples", None)
    ground_truth_outcome_counts = {
        "recognized_approved_correct": counts["approved_correct"],
        "top3_candidate": counts["unknown_top3_correct"],
        "candidate_out": counts["unknown_top3_missing"],
        "approved_misclassification": counts["approved_wrong_matched"],
        "recapture": counts["recapture_ground_truth_boxes"]
        + counts["segment_recapture_matched_boxes"],
        "unblocked_segmentation_missed": counts["unblocked_missed_boxes"],
    }
    if sum(ground_truth_outcome_counts.values()) != counts["ground_truth_boxes"]:
        raise ValueError("ground-truth box outcomes do not partition all boxes")
    result["response_status"] = {
        status.value: counts["response_status"][status.value]
        for status in (Status.APPROVED, Status.UNKNOWN, Status.RECAPTURE)
    }
    result["recapture_reasons"] = dict(sorted(counts["recapture_reasons"].items()))
    result["segment_recapture_reasons"] = dict(sorted(counts["segment_recapture_reasons"].items()))
    result["rates"] = {
        "recapture_image_rate": _safe_rate(
            counts["response_status"][Status.RECAPTURE.value], counts["images"]
        ),
        "detector_box_success_rate": _safe_rate(
            counts["matched_boxes"], counts["ground_truth_boxes"]
        ),
        "detector_box_failure_rate": _safe_rate(
            counts["missed_boxes"], counts["ground_truth_boxes"]
        ),
        "detector_false_positive_rate": _safe_rate(
            counts["false_positive_boxes"], counts["predicted_boxes"]
        ),
        "exact_detection_image_rate": _safe_rate(
            counts["exact_detection_images"], counts["images"]
        ),
        "approved_wrong_rate": _safe_rate(counts["approved_wrong"], counts["approved_boxes"]),
        "approved_accuracy": _safe_rate(counts["approved_correct"], counts["approved_boxes"]),
        "unknown_top3_missing_rate": _safe_rate(
            counts["unknown_top3_missing"], counts["unknown_matched_boxes"]
        ),
        "unknown_top3_accuracy": _safe_rate(
            counts["unknown_top3_correct"], counts["unknown_matched_boxes"]
        ),
        "segment_recapture_rate_of_matched_detections": _safe_rate(
            counts["segment_recapture_matched_boxes"], counts["matched_boxes"]
        ),
        "classifier_top1_accuracy_excluding_recapture": _safe_rate(
            counts["top1_correct"], counts["classified_matched_boxes"]
        ),
        "approval_coverage_of_classified_detections": _safe_rate(
            counts["approved_boxes"], counts["classified_matched_boxes"]
        ),
        "e2e_top1_accuracy_all_ground_truth": _safe_rate(
            counts["top1_correct"], counts["ground_truth_boxes"]
        ),
        "e2e_top3_conservative_accuracy_all_ground_truth": _safe_rate(
            counts["approved_correct"] + counts["unknown_top3_correct"],
            counts["ground_truth_boxes"],
        ),
        "e2e_safe_resolution_rate_all_ground_truth": _safe_rate(
            counts["approved_correct"]
            + counts["unknown_top3_correct"]
            + counts["recapture_ground_truth_boxes"]
            + counts["segment_recapture_matched_boxes"],
            counts["ground_truth_boxes"],
        ),
    }
    result["all_ground_truth_box_outcomes"] = {
        "denominator": counts["ground_truth_boxes"],
        "counts": ground_truth_outcome_counts,
        "rates": {
            name: _safe_rate(value, counts["ground_truth_boxes"])
            for name, value in ground_truth_outcome_counts.items()
        },
        "unblocked_false_positive_boxes": counts["unblocked_false_positive_boxes"],
        "unblocked_false_positive_boxes_per_ground_truth": _safe_rate(
            counts["unblocked_false_positive_boxes"], counts["ground_truth_boxes"]
        ),
        "raw_detector_missed_boxes": counts["missed_boxes"],
        "raw_detector_false_positive_boxes": counts["false_positive_boxes"],
    }
    result["end_to_end_latency_ms"] = {
        "sample_count": counts["images"],
        "mean": float(latency_samples.mean()) if len(latency_samples) else None,
        "p50": float(np.percentile(latency_samples, 50)) if len(latency_samples) else None,
        "p95": float(np.percentile(latency_samples, 95)) if len(latency_samples) else None,
        "p99": float(np.percentile(latency_samples, 99)) if len(latency_samples) else None,
        "minimum": float(latency_samples.min()) if len(latency_samples) else None,
        "maximum": float(latency_samples.max()) if len(latency_samples) else None,
    }
    return result


def _write_details(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "difficulty",
        "image",
        "response_status",
        "reason_codes",
        "error_type",
        "item_id",
        "item_status",
        "expected_class_id",
        "predicted_class_id",
        "top3_class_ids",
        "iou",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    package = load_model_package(args.package_dir)
    if args.border_policy is not None:
        package.metadata.quality = package.metadata.quality.model_copy(
            update={"border_policy": args.border_policy}
        )
    detector, classifier, provider = build_onnx_adapters(
        package, args.provider, cuda_dll_dir=args.cuda_dll_dir
    )
    recording_detector = RecordingDetector(detector)
    pipeline = DecisionPipeline(
        recording_detector,
        classifier,
        package.metadata.classifier,
        package.metadata.quality,
        package.metadata.count_verifier,
    )
    records = _load_records(args.dataset_root)
    counts = {difficulty: _empty_counts() for difficulty in (*DIFFICULTIES, "ALL")}
    detail_rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records, start=1):
        encoded_image = record["image_path"].read_bytes()
        started_at = time.perf_counter()
        image = decode_image(
            encoded_image,
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        response = pipeline.scan(image, request_id=f"evaluation-{record_index:06d}")
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        detection_result = recording_detector.last_result
        if detection_result is None:
            raise RuntimeError("detector result was not recorded")
        ordered_detections = sorted(
            detection_result.detections, key=lambda detection: (detection.y1, detection.x1)
        )
        matches, remaining_gt = _match_detections(
            ordered_detections, record["annotations"], args.match_iou_threshold
        )
        false_positive_indices = set(range(len(ordered_detections))) - set(matches)
        exact_detection = not remaining_gt and not false_positive_indices
        selected_counts = (counts[record["difficulty"]], counts["ALL"])

        for bucket in selected_counts:
            bucket["images"] += 1
            bucket["response_status"][response.status.value] += 1
            bucket["ground_truth_boxes"] += len(record["annotations"])
            bucket["predicted_boxes"] += len(ordered_detections)
            bucket["matched_boxes"] += len(matches)
            bucket["missed_boxes"] += len(remaining_gt)
            bucket["false_positive_boxes"] += len(false_positive_indices)
            bucket["exact_detection_images"] += int(exact_detection)
            bucket["failed_detection_images"] += int(not exact_detection)
            bucket["end_to_end_latency_ms_total"] += elapsed_ms
            bucket["end_to_end_latency_ms_samples"].append(elapsed_ms)

        common = {
            "group": record["group"],
            "difficulty": record["difficulty"],
            "image": str(record["image_path"]),
            "response_status": response.status.value,
        }
        for gt_index in sorted(remaining_gt):
            detail_rows.append(
                {
                    **common,
                    "error_type": "DETECTOR_MISSED_GT",
                    "expected_class_id": f"bread_{int(record['annotations'][gt_index]['category_id']):02d}",
                }
            )
        for detection_index in sorted(false_positive_indices):
            detail_rows.append(
                {
                    **common,
                    "error_type": "DETECTOR_FALSE_POSITIVE",
                    "iou": "",
                }
            )

        if response.status is Status.RECAPTURE:
            for bucket in selected_counts:
                bucket["recapture_reasons"].update(response.reason_codes)
                bucket["recapture_ground_truth_boxes"] += len(record["annotations"])
            detail_rows.append(
                {
                    **common,
                    "reason_codes": "|".join(response.reason_codes),
                    "error_type": "RECAPTURE",
                }
            )
            continue

        if len(response.items) != len(ordered_detections):
            raise RuntimeError("Worker item count does not match detector output")
        for bucket in selected_counts:
            bucket["non_recapture_images"] += 1
            bucket["unblocked_missed_boxes"] += len(remaining_gt)
            bucket["unblocked_false_positive_boxes"] += len(false_positive_indices)

        for detection_index, item in enumerate(response.items):
            match = matches.get(detection_index)
            if item.status is ItemStatus.APPROVED:
                for bucket in selected_counts:
                    bucket["approved_boxes"] += 1
                if match is None:
                    for bucket in selected_counts:
                        bucket["approved_wrong"] += 1
                        bucket["approved_wrong_unmatched"] += 1
                    detail_rows.append(
                        {
                            **common,
                            "error_type": "APPROVED_WRONG_UNMATCHED",
                            "item_id": item.item_id,
                            "item_status": item.status.value,
                            "predicted_class_id": item.prediction.class_id,
                        }
                    )
                    continue
                gt_index, overlap = match
                expected = f"bread_{int(record['annotations'][gt_index]['category_id']):02d}"
                predicted = item.prediction.class_id
                correct = predicted == expected
                for bucket in selected_counts:
                    bucket["classified_matched_boxes"] += 1
                    bucket["top1_correct"] += int(correct)
                    bucket["approved_correct"] += int(correct)
                    bucket["approved_wrong"] += int(not correct)
                    bucket["approved_wrong_matched"] += int(not correct)
                if not correct:
                    detail_rows.append(
                        {
                            **common,
                            "error_type": "APPROVED_WRONG_CLASS",
                            "item_id": item.item_id,
                            "item_status": item.status.value,
                            "expected_class_id": expected,
                            "predicted_class_id": predicted,
                            "iou": round(overlap, 6),
                        }
                    )
            elif item.status is ItemStatus.SEGMENT_RECAPTURE:
                for bucket in selected_counts:
                    bucket["segment_recapture_boxes"] += 1
                    bucket["segment_recapture_reasons"].update(item.reason_codes)
                if match is None:
                    for bucket in selected_counts:
                        bucket["segment_recapture_unmatched_boxes"] += 1
                    detail_rows.append(
                        {
                            **common,
                            "reason_codes": "|".join(item.reason_codes),
                            "error_type": "SEGMENT_RECAPTURE_UNMATCHED",
                            "item_id": item.item_id,
                            "item_status": item.status.value,
                        }
                    )
                    continue
                gt_index, overlap = match
                expected = f"bread_{int(record['annotations'][gt_index]['category_id']):02d}"
                for bucket in selected_counts:
                    bucket["segment_recapture_matched_boxes"] += 1
                detail_rows.append(
                    {
                        **common,
                        "reason_codes": "|".join(item.reason_codes),
                        "error_type": "SEGMENT_RECAPTURE",
                        "item_id": item.item_id,
                        "item_status": item.status.value,
                        "expected_class_id": expected,
                        "iou": round(overlap, 6),
                    }
                )
            elif item.status is ItemStatus.UNKNOWN:
                for bucket in selected_counts:
                    bucket["unknown_boxes"] += 1
                if match is None:
                    for bucket in selected_counts:
                        bucket["unknown_unmatched"] += 1
                    detail_rows.append(
                        {
                            **common,
                            "error_type": "UNKNOWN_UNMATCHED",
                            "item_id": item.item_id,
                            "item_status": item.status.value,
                            "top3_class_ids": "|".join(
                                candidate.class_id for candidate in item.top3
                            ),
                        }
                    )
                    continue
                gt_index, overlap = match
                expected = f"bread_{int(record['annotations'][gt_index]['category_id']):02d}"
                top3 = [candidate.class_id for candidate in item.top3]
                top3_correct = expected in top3
                top1_correct = bool(top3) and top3[0] == expected
                for bucket in selected_counts:
                    bucket["classified_matched_boxes"] += 1
                    bucket["top1_correct"] += int(top1_correct)
                    bucket["unknown_matched_boxes"] += 1
                    bucket["unknown_top3_correct"] += int(top3_correct)
                    bucket["unknown_top3_missing"] += int(not top3_correct)
                if not top3_correct:
                    detail_rows.append(
                        {
                            **common,
                            "error_type": "UNKNOWN_TOP3_MISSING",
                            "item_id": item.item_id,
                            "item_status": item.status.value,
                            "expected_class_id": expected,
                            "top3_class_ids": "|".join(top3),
                            "iou": round(overlap, 6),
                        }
                    )
            else:
                raise RuntimeError(f"unsupported segmentation status: {item.status}")

    report = {
        "evaluation": "worker_difficulty_diagnostic",
        "provider": provider,
        "package_version": package.metadata.package_version,
        "dataset_root": str(args.dataset_root.resolve()),
        "match_iou_threshold": args.match_iou_threshold,
        "border_policy": package.metadata.quality.border_policy,
        "detector_uncertainty_policy": {
            "score_threshold": package.metadata.detector.uncertainty_score_threshold,
            "min_area_ratio": package.metadata.detector.uncertainty_min_area_ratio,
            "match_iou_threshold": package.metadata.detector.uncertainty_match_iou_threshold,
        },
        "classification_policy": "exclude IMAGE_RECAPTURE and SEGMENT_RECAPTURE from classifier accuracy; APPROVED wrong includes unmatched; UNKNOWN Top-3 rate uses matched UNKNOWN boxes only",
        "difficulty_source": "COCO file_name E/M/H directory or filename _e_/_m_/_h_ token",
        "evaluation_passes_per_image": 1,
        "latency_scope": "single pass per image; decode, preprocess, detector, classifier when executed, and postprocess; file read excluded",
        "diagnostic_set_note": args.dataset_note
        or "All requested images are included; this set may overlap model development/evaluation data and is not automatically treated as an independent generalization test.",
        "by_difficulty": {
            difficulty: _finalize_counts(counts[difficulty]) for difficulty in DIFFICULTIES
        },
        "overall": _finalize_counts(counts["ALL"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.details_output:
        _write_details(args.details_output, detail_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate packaged Worker outcomes by E/M/H filename difficulty"
    )
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--dataset-note")
    parser.add_argument(
        "--border-policy",
        choices=("always_recapture", "classifier_confidence"),
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
