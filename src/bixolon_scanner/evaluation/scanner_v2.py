from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..contracts import ItemStatus, Status, load_runtime_package_v2, load_store_catalog_package
from ..contracts.catalog import sha256_file
from ..pipeline import DecisionPipeline
from ..pipeline.ports import ClassificationResult, Detection, DetectionResult
from ..runtime.catalog import OnnxCatalogClassifier, OnnxEmbedder
from ..runtime.detector_v2 import build_detector_v2
from ..runtime.imaging import decode_image
from ..runtime.onnx import box_iou


def _resolve_image_path(dataset_root: Path, value: str) -> Path:
    root = dataset_root.resolve()
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("evaluation image path escaped the dataset root") from exc
    return resolved


def _records(path: Path, dataset_root: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            raise ValueError("evaluation JSON must be a COCO object")
        annotations_by_image: dict[int, list[dict]] = {}
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            annotations_by_image.setdefault(image_id, []).append(
                {
                    "bbox_xywh": [float(value) for value in annotation["bbox"]],
                    "category_id": int(annotation["category_id"]),
                }
            )
        rows = [
            {
                **image,
                "image_id": int(image["id"]),
                "image_path": str(image["file_name"]),
                "annotations": annotations_by_image.get(int(image["id"]), []),
            }
            for image in payload["images"]
        ]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line]
    for row in rows:
        row["resolved_path"] = _resolve_image_path(dataset_root, str(row["image_path"]))
    return rows


def _match(
    detections: list[Detection], annotations: list[dict], threshold: float
) -> tuple[dict[int, int], set[int]]:
    ground_truth = [
        Detection(x, y, x + width, y + height, 1.0)
        for x, y, width, height in (row["bbox_xywh"] for row in annotations)
    ]
    candidates = sorted(
        (
            (box_iou(detection, target), detection_index, target_index)
            for detection_index, detection in enumerate(detections)
            for target_index, target in enumerate(ground_truth)
        ),
        reverse=True,
    )
    matches = {}
    used_targets: set[int] = set()
    for iou, detection_index, target_index in candidates:
        if iou < threshold:
            break
        if detection_index not in matches and target_index not in used_targets:
            matches[detection_index] = target_index
            used_targets.add(target_index)
    return matches, set(range(len(ground_truth))) - used_targets


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _latency(values: list[float]) -> dict:
    if not values:
        return {
            "sample_count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


class RecordingDetector:
    def __init__(self, detector):
        self.detector = detector
        self.version = detector.version
        self.last_result: DetectionResult | None = None
        self.last_latency_ms: float | None = None

    def detect(self, image) -> DetectionResult:
        started = time.perf_counter()
        self.last_result = self.detector.detect(image)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return self.last_result


class RecordingClassifier:
    def __init__(self, classifier: OnnxCatalogClassifier):
        self.classifier = classifier
        self.version = classifier.version
        self.metadata = classifier.metadata
        self.last_result: ClassificationResult | None = None
        self.last_latency_ms: float | None = None

    def classify(self, image, detections: list[Detection]) -> ClassificationResult:
        started = time.perf_counter()
        self.last_result = self.classifier.classify(image, detections)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return self.last_result


@dataclass
class Counts:
    image_count: int = 0
    segmentation_image_count: int = 0
    image_recapture_count: int = 0
    ground_truth_count: int = 0
    prediction_count: int = 0
    matched_count: int = 0
    false_negative_count: int = 0
    false_positive_count: int = 0
    false_negative_image_count: int = 0
    false_positive_image_count: int = 0
    approved_count: int = 0
    approved_misrecognition_count: int = 0
    unknown_count: int = 0
    unknown_candidate_out_count: int = 0
    segment_recapture_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    full_path_latencies_ms: list[float] = field(default_factory=list)
    image_recapture_latencies_ms: list[float] = field(default_factory=list)
    refinement_latencies_ms: list[float] = field(default_factory=list)
    detector_latencies_ms: list[float] = field(default_factory=list)
    classifier_latencies_ms: list[float] = field(default_factory=list)
    refinement_count: int = 0


def _reported_metrics(counts: Counts) -> tuple[dict, dict]:
    gt = counts.ground_truth_count
    segmentation_images = counts.segmentation_image_count
    segmentation_objects = counts.prediction_count
    requested = {
        "segmentation_rate": _rate(segmentation_images, counts.image_count),
        "image_recapture_rate": _rate(counts.image_recapture_count, counts.image_count),
        "segmentation_object_count": segmentation_objects,
        "approved_rate": _rate(counts.approved_count, segmentation_objects),
        "unknown_top3_rate": _rate(counts.unknown_count, segmentation_objects),
        "segment_recapture_rate": _rate(counts.segment_recapture_count, segmentation_objects),
        "segmentation_image_false_negative_rate": _rate(
            counts.false_negative_image_count, segmentation_images
        ),
        "segmentation_image_false_positive_rate": _rate(
            counts.false_positive_image_count, segmentation_images
        ),
        "approved_object_misrecognition_rate": _rate(counts.approved_misrecognition_count, gt),
        "approved_output_misrecognition_rate": _rate(
            counts.approved_misrecognition_count, counts.approved_count
        ),
        "correct_approved_rate": _rate(
            counts.approved_count - counts.approved_misrecognition_count, gt
        ),
        "unknown_top3_candidate_out_rate": _rate(counts.unknown_candidate_out_count, gt),
        "mean_speed_ms": float(np.mean(counts.latencies_ms)),
    }
    promotion = {
        "approved_all_gt_rate": _rate(counts.approved_count, gt),
        "correct_approved_all_gt_rate": _rate(
            counts.approved_count - counts.approved_misrecognition_count, gt
        ),
        "approved_misrecognition_all_gt_rate": _rate(counts.approved_misrecognition_count, gt),
        "unknown_candidate_out_all_gt_rate": _rate(counts.unknown_candidate_out_count, gt),
    }
    return requested, promotion


def evaluate(args: argparse.Namespace) -> dict:
    signing_key = os.environ.get(args.signing_key_env, "").encode()
    if len(signing_key) < 16:
        raise ValueError("Catalog signing key must contain at least 16 bytes")
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(
        args.catalog,
        signing_key=signing_key,
        expected_store_id=args.store_id,
        expected_key_id=args.key_id,
    )
    detector = RecordingDetector(build_detector_v2(runtime, args.provider, args.cuda_dll_dir))
    embedder = OnnxEmbedder(runtime, args.provider, args.cuda_dll_dir)
    classifier = RecordingClassifier(OnnxCatalogClassifier(runtime, catalog, embedder))
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier.metadata,
        runtime.metadata.quality,
        worker_version=runtime.metadata.worker_version,
        embedder_version=runtime.metadata.embedder.version,
        detector_policy_version=runtime.metadata.detector_policy_version,
        classifier_policy_version=runtime.metadata.classifier_policy.version,
        catalog_version=catalog.metadata.catalog_version,
    )
    records = _records(args.manifest, args.dataset_root)
    expected_image_count = int(getattr(args, "expected_image_count", 300))
    evidence_role = str(getattr(args, "evidence_role", "development_regression"))
    if expected_image_count <= 0 or len(records) != expected_image_count:
        raise ValueError(
            f"evaluation requires exactly {expected_image_count} images; observed {len(records)}"
        )
    warmup_image = decode_image(
        records[0]["resolved_path"].read_bytes(),
        max_bytes=50_000_000,
        max_pixels=50_000_000,
        jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
    )
    try:
        for index in range(args.warmup_count):
            pipeline.scan(warmup_image, request_id=f"scanner2-warmup-{index:06d}")
    finally:
        warmup_image.close()
    counts = Counts()
    trace = []
    classifier_diagnostics: list[dict] = []
    request_interval_seconds = args.request_interval_ms / 1000.0
    next_request_started = time.perf_counter()
    for ordinal, record in enumerate(records, start=1):
        # The public performance contract starts at API-internal decode. File I/O belongs to
        # the benchmark harness, not to the Worker request path, where multipart bytes are
        # already available before decode begins.
        image_bytes = record["resolved_path"].read_bytes()
        remaining_interval = next_request_started - time.perf_counter()
        if remaining_interval > 0.0:
            time.sleep(remaining_interval)
        started = time.perf_counter()
        next_request_started = started + request_interval_seconds
        image = decode_image(
            image_bytes,
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
        )
        try:
            response = pipeline.scan(
                image,
                request_id=(
                    f"scanner2-"
                    f"{'development' if evidence_role == 'development_regression' else 'stress'}-"
                    f"{ordinal:06d}"
                ),
            )
        finally:
            image.close()
        elapsed = (time.perf_counter() - started) * 1000.0
        counts.image_count += 1
        counts.ground_truth_count += len(record["annotations"])
        counts.latencies_ms.append(elapsed)
        detector_result = detector.last_result
        if detector_result is None or detector.last_latency_ms is None:
            raise RuntimeError("detector result was not recorded")
        counts.detector_latencies_ms.append(detector.last_latency_ms)
        if detector_result.refinement_executed:
            counts.refinement_count += 1
            counts.refinement_latencies_ms.append(elapsed)
        detail = {
            "image_id": record["image_id"],
            "status": response.status.value,
            "reason_codes": response.reason_codes,
            "latency_ms": elapsed,
            "ground_truth_count": len(record["annotations"]),
            "decision": {
                key: value
                for key, value in response.model_dump(mode="json").items()
                if key not in {"request_id", "processing_time_ms"}
            },
        }
        if response.status is Status.IMAGE_RECAPTURE:
            counts.image_recapture_count += 1
            counts.image_recapture_latencies_ms.append(elapsed)
            trace.append(detail)
            continue
        if response.status is not Status.SEGMENTATION:
            raise RuntimeError("development evaluation received an ERROR response")
        counts.full_path_latencies_ms.append(elapsed)
        result = detector_result
        classification = classifier.last_result
        if (
            classification is None
            or classification.approval_scores is None
            or classifier.last_latency_ms is None
        ):
            raise RuntimeError("classifier result was not recorded")
        counts.classifier_latencies_ms.append(classifier.last_latency_ms)
        detections = sorted(result.detections, key=lambda value: (value.y1, value.x1))
        matches, missed = _match(detections, record["annotations"], args.match_iou_threshold)
        false_positive_count = len(detections) - len(matches)
        counts.segmentation_image_count += 1
        counts.prediction_count += len(detections)
        counts.matched_count += len(matches)
        counts.false_negative_count += len(missed)
        counts.false_positive_count += false_positive_count
        counts.false_negative_image_count += bool(missed)
        counts.false_positive_image_count += false_positive_count > 0
        status_counts = {status.value: 0 for status in ItemStatus}
        item_diagnostics = []
        ranking = (
            classification.ranking_scores
            if classification.ranking_scores is not None
            else classification.ranking_logits
        )
        ranking_order = np.argsort(-ranking, axis=1, kind="stable")
        for detection_index, segmentation in enumerate(response.segmentations):
            target_index = matches.get(detection_index)
            if target_index is None:
                continue
            target = f"bread_{int(record['annotations'][target_index]['category_id']):02d}"
            predicted = classifier.metadata.labels[int(ranking_order[detection_index, 0])].class_id
            classifier_correct = predicted == target
            top3_ids = {
                classifier.metadata.labels[int(index)].class_id
                for index in ranking_order[detection_index, :3]
            }
            top3_hit = target in top3_ids
            approval_score = float(classification.approval_scores[detection_index])
            top3_safety_score = (
                None
                if classification.top3_safety_scores is None
                else float(classification.top3_safety_scores[detection_index])
            )
            classifier_diagnostics.append(
                {
                    "approval_score": approval_score,
                    "top1_correct": classifier_correct,
                    "top3_hit": top3_hit,
                    "top3_safety_score": top3_safety_score,
                }
            )
            item_diagnostics.append(
                {
                    "detection_index": detection_index,
                    "target_class_id": target,
                    "classifier_top1_class_id": predicted,
                    "classifier_top1_correct": classifier_correct,
                    "classifier_top3_hit": top3_hit,
                    "approval_score": approval_score,
                    "top3_safety_score": top3_safety_score,
                    "final_status": segmentation.status.value,
                }
            )
            status_counts[segmentation.status.value] += 1
            if segmentation.status is ItemStatus.APPROVED:
                counts.approved_count += 1
                counts.approved_misrecognition_count += (
                    segmentation.prediction is None or segmentation.prediction.class_id != target
                )
            elif segmentation.status is ItemStatus.UNKNOWN:
                counts.unknown_count += 1
                counts.unknown_candidate_out_count += target not in {
                    candidate.class_id for candidate in segmentation.top3
                }
            else:
                counts.segment_recapture_count += 1
        trace.append(
            {
                **detail,
                "prediction_count": len(detections),
                "matched_count": len(matches),
                "false_negative_count": len(missed),
                "false_positive_count": false_positive_count,
                "matched_status_counts": status_counts,
                "matched_classifier_diagnostics": item_diagnostics,
            }
        )
    gt = counts.ground_truth_count
    requested, promotion_metrics = _reported_metrics(counts)
    limits = {
        "minimum_segmentation_rate": 0.90,
        "minimum_approved_rate": 0.90,
        "maximum_fn_image_rate": 0.001,
        "maximum_fp_image_rate": 0.001,
        "maximum_approved_misrecognition_rate": 0.001,
        "maximum_candidate_out_rate": 0.001,
        "maximum_mean_ms": 100.0,
        "maximum_p95_ms": 100.0,
        "maximum_p99_ms": 150.0,
    }
    performance = _latency(counts.latencies_ms)
    full_path_performance = _latency(counts.full_path_latencies_ms)
    recapture_performance = _latency(counts.image_recapture_latencies_ms)
    refinement_performance = _latency(counts.refinement_latencies_ms)
    allowed_classifier_errors = int(np.floor(limits["maximum_approved_misrecognition_rate"] * gt))
    threshold_prefix_count = 0
    threshold_prefix_errors = 0
    threshold_lower_bound = None
    grouped_scores: dict[float, list[bool]] = {}
    for row in classifier_diagnostics:
        grouped_scores.setdefault(row["approval_score"], []).append(row["top1_correct"])
    for score in sorted(grouped_scores, reverse=True):
        correctness = grouped_scores[score]
        next_errors = threshold_prefix_errors + sum(not correct for correct in correctness)
        if next_errors > allowed_classifier_errors:
            break
        threshold_prefix_count += len(correctness)
        threshold_prefix_errors = next_errors
        threshold_lower_bound = score
    top1_correct_count = sum(row["top1_correct"] for row in classifier_diagnostics)
    nonapproved = [
        row
        for row in classifier_diagnostics
        if threshold_lower_bound is None or row["approval_score"] < threshold_lower_bound
    ]
    top3_miss_safety = [
        row["top3_safety_score"]
        for row in nonapproved
        if not row["top3_hit"] and row["top3_safety_score"] is not None
    ]
    top3_safety_lower_bound = (
        None
        if not top3_miss_safety
        else float(
            np.nextafter(
                np.float32(max(top3_miss_safety)),
                np.float32(np.inf),
                dtype=np.float32,
            )
        )
    )
    safe_unknown = [
        row
        for row in nonapproved
        if row["top3_safety_score"] is not None
        and (top3_safety_lower_bound is None or row["top3_safety_score"] >= top3_safety_lower_bound)
    ]
    threshold_diagnostic = {
        "role": "development_diagnostic_not_a_selected_policy",
        "matched_classifier_sample_count": len(classifier_diagnostics),
        "classifier_top1_correct_count": top1_correct_count,
        "classifier_top1_accuracy": _rate(top1_correct_count, len(classifier_diagnostics)),
        "allowed_classifier_error_count": allowed_classifier_errors,
        "maximum_threshold_prefix_approved_count": threshold_prefix_count,
        "maximum_threshold_prefix_approved_rate_over_all_gt": _rate(threshold_prefix_count, gt),
        "classifier_error_count_at_prefix": threshold_prefix_errors,
        "approval_score_lower_bound": threshold_lower_bound,
        "top3_safety_lower_bound_for_zero_candidate_out": top3_safety_lower_bound,
        "unknown_count_at_diagnostic_thresholds": len(safe_unknown),
        "unknown_candidate_out_count_at_diagnostic_thresholds": sum(
            not row["top3_hit"] for row in safe_unknown
        ),
        "segment_recapture_count_at_diagnostic_thresholds": len(nonapproved) - len(safe_unknown),
    }
    gates = {
        "segmentation_rate": requested["segmentation_rate"] >= limits["minimum_segmentation_rate"],
        "approved_rate": promotion_metrics["approved_all_gt_rate"]
        >= limits["minimum_approved_rate"],
        "fn_image_rate": requested["segmentation_image_false_negative_rate"]
        <= limits["maximum_fn_image_rate"],
        "fp_image_rate": requested["segmentation_image_false_positive_rate"]
        <= limits["maximum_fp_image_rate"],
        "approved_misrecognition_rate": requested["approved_object_misrecognition_rate"]
        <= limits["maximum_approved_misrecognition_rate"],
        "candidate_out_rate": requested["unknown_top3_candidate_out_rate"]
        <= limits["maximum_candidate_out_rate"],
        "performance": full_path_performance["mean_ms"] <= limits["maximum_mean_ms"]
        and full_path_performance["p95_ms"] <= limits["maximum_p95_ms"]
        and full_path_performance["p99_ms"] <= limits["maximum_p99_ms"],
    }
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace),
        encoding="utf-8",
    )
    all_regression_gates_met = all(gates.values())
    development_evaluation = evidence_role == "development_regression"
    gate_key = "development_gates" if development_evaluation else "stress_regression_gates"
    diagnostic_key = (
        "development_threshold_diagnostic"
        if development_evaluation
        else "stress_threshold_diagnostic"
    )
    report = {
        "schema_version": "2.0",
        "evaluation": (
            "scanner_2_0_development_300"
            if development_evaluation
            else "scanner_2_0_stress_regression"
        ),
        "promotion_evidence": False,
        "evidence_role": evidence_role,
        "dataset": {
            "manifest_sha256": sha256_file(args.manifest),
            "image_count": counts.image_count,
            "ground_truth_object_count": gt,
            "overlaps_runtime_development": bool(
                getattr(args, "overlaps_runtime_development", True)
            ),
        },
        "versions": {
            "worker": runtime.metadata.worker_version,
            "detector": runtime.metadata.detector.version,
            "embedder": runtime.metadata.embedder.version,
            "classifier_policy": runtime.metadata.classifier_policy.version,
            "catalog": catalog.metadata.catalog_version,
        },
        "counts": {
            key: value for key, value in vars(counts).items() if not key.endswith("latencies_ms")
        },
        "requested_metrics": requested,
        "promotion_metrics": promotion_metrics,
        "performance": {
            **performance,
            "scope": ("decode+preprocess+detector-ensemble+selective-refinement+embedder+decision"),
            "warmup_count": args.warmup_count,
            "request_interval_ms": args.request_interval_ms,
            "gate_path": "full_path_only",
            "full_path": full_path_performance,
            "image_recapture_early_exit": recapture_performance,
            "selective_refinement": refinement_performance,
            "detector_stage": _latency(counts.detector_latencies_ms),
            "classifier_stage": _latency(counts.classifier_latencies_ms),
        },
        "limits": limits,
        gate_key: {**gates, "all_met": all_regression_gates_met},
        diagnostic_key: threshold_diagnostic,
        "production_status": (
            (
                "development_gates_passed_rc_prerequisites_pending"
                if all_regression_gates_met
                else "development_gates_failed"
            )
            if development_evaluation
            else (
                "non_promotion_stress_regression_passed"
                if all_regression_gates_met
                else "non_promotion_stress_regression_failed"
            )
        ),
        "trace": {
            "path": args.trace_output.resolve().as_posix(),
            "sha256": sha256_file(args.trace_output),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "provider": args.provider,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Scanner 2.0 on a locked development or stress-regression set"
    )
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument(
        "--request-interval-ms",
        type=float,
        default=0.0,
        help="Minimum interval between measured request starts; use 1000 for the 1 image/s gate.",
    )
    parser.add_argument("--expected-image-count", type=int, default=300)
    parser.add_argument(
        "--evidence-role",
        choices=("development_regression", "stress_regression"),
        default="development_regression",
    )
    parser.add_argument(
        "--overlaps-runtime-development",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.request_interval_ms < 0.0:
        parser.error("--request-interval-ms must be non-negative")
    evaluate(args)


if __name__ == "__main__":
    main()
