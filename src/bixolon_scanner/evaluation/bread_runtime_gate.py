from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import ItemStatus, Status
from ..contracts.model_package import load_model_package, sha256_file
from ..pipeline import DecisionPipeline
from ..runtime.imaging import decode_image
from ..runtime.onnx import build_onnx_adapters
from .onnx_detector import load_records
from .release import RecordingClassifier, RecordingDetector, _match


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _latency(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _decision_trace(image_id: int | str, response: Any) -> dict[str, Any]:
    """Return the provider-comparable public decision, excluding request timing."""
    payload = response.model_dump(mode="json")
    payload.pop("request_id", None)
    payload.pop("processing_time_ms", None)
    return {"image_id": image_id, **payload}


@dataclass
class RuntimeGateCounts:
    image_count: int = 0
    segmentation_image_count: int = 0
    image_recapture_count: int = 0
    judgeable_ground_truth_object_count: int = 0
    predicted_segmentation_count: int = 0
    matched_segmentation_count: int = 0
    false_negative_count: int = 0
    false_positive_count: int = 0
    false_negative_image_count: int = 0
    false_positive_image_count: int = 0
    approved_count: int = 0
    approved_misrecognition_count: int = 0
    unknown_count: int = 0
    unknown_top3_candidate_out_count: int = 0
    segment_recapture_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)


def build_runtime_gate_metrics(counts: RuntimeGateCounts) -> dict[str, Any]:
    total_gt = counts.judgeable_ground_truth_object_count
    segmentation_images = counts.segmentation_image_count
    rates = {
        "segmentation_image_rate": _rate(segmentation_images, counts.image_count),
        "end_to_end_approved_object_rate": _rate(counts.approved_count, total_gt),
        "segmentation_image_false_negative_rate": _rate(
            counts.false_negative_image_count, segmentation_images
        ),
        "segmentation_image_false_positive_rate": _rate(
            counts.false_positive_image_count, segmentation_images
        ),
        "approved_object_misrecognition_rate": _rate(
            counts.approved_misrecognition_count, total_gt
        ),
        "unknown_top3_candidate_out_rate": _rate(counts.unknown_top3_candidate_out_count, total_gt),
        "unknown_rate_diagnostic_only": _rate(counts.unknown_count, total_gt),
        "segment_recapture_rate_diagnostic_only": _rate(counts.segment_recapture_count, total_gt),
    }
    limits = {
        "minimum_segmentation_image_rate": 0.90,
        "minimum_end_to_end_approved_object_rate": 0.90,
        "maximum_segmentation_image_false_negative_rate": 0.001,
        "maximum_segmentation_image_false_positive_rate": 0.001,
        "maximum_approved_object_misrecognition_rate": 0.001,
        "maximum_unknown_top3_candidate_out_rate": 0.001,
        "final_end_to_end_approved_object_goal": 0.99,
    }
    gates = {
        "segmentation_image_rate": rates["segmentation_image_rate"]
        >= limits["minimum_segmentation_image_rate"],
        "end_to_end_approved_object_rate": rates["end_to_end_approved_object_rate"]
        >= limits["minimum_end_to_end_approved_object_rate"],
        "segmentation_image_false_negative_rate": rates["segmentation_image_false_negative_rate"]
        <= limits["maximum_segmentation_image_false_negative_rate"],
        "segmentation_image_false_positive_rate": rates["segmentation_image_false_positive_rate"]
        <= limits["maximum_segmentation_image_false_positive_rate"],
        "approved_object_misrecognition_rate": rates["approved_object_misrecognition_rate"]
        <= limits["maximum_approved_object_misrecognition_rate"],
        "unknown_top3_candidate_out_rate": rates["unknown_top3_candidate_out_rate"]
        <= limits["maximum_unknown_top3_candidate_out_rate"],
    }
    return {
        "counts": {key: value for key, value in vars(counts).items() if key != "latencies_ms"},
        "rates": rates,
        "limits": limits,
        "operational_gates": {**gates, "all_met": all(gates.values())},
        "final_end_to_end_approved_goal_met": (
            rates["end_to_end_approved_object_rate"]
            >= limits["final_end_to_end_approved_object_goal"]
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    annotation_path = dataset_root / "annotations" / args.annotation_name
    records = load_records(dataset_root, args.annotation_name)
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
        worker_version=package.metadata.worker_version,
    )
    if records and args.warmup_count:
        warmup_image = decode_image(
            records[0]["image_path"].read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        try:
            for index in range(args.warmup_count):
                pipeline.scan(warmup_image, request_id=f"bread11-warmup-{index:06d}")
        finally:
            warmup_image.close()
    counts = RuntimeGateCounts()
    details: list[dict[str, Any]] = []
    decision_trace: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records, start=1):
        started = time.perf_counter()
        image = decode_image(
            record["image_path"].read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=package.metadata.input.jpeg_draft_size,
        )
        try:
            response = pipeline.scan(image, request_id=f"bread11-runtime-{ordinal:06d}")
        finally:
            image.close()
        counts.image_count += 1
        decision_trace.append(_decision_trace(record["image_id"], response))
        counts.judgeable_ground_truth_object_count += len(record["annotations"])
        counts.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        if response.status is Status.IMAGE_RECAPTURE:
            counts.image_recapture_count += 1
            details.append(
                {
                    "image_id": record["image_id"],
                    "status": response.status.value,
                    "reason_codes": response.reason_codes,
                    "ground_truth_count": len(record["annotations"]),
                }
            )
            continue
        if response.status is not Status.SEGMENTATION:
            raise RuntimeError(f"runtime gate received unsupported status: {response.status}")
        result = recording_detector.last_result
        if result is None:
            raise RuntimeError("runtime gate detector result was not recorded")
        detections = sorted(result.detections, key=lambda value: (value.y1, value.x1))
        if len(detections) != len(response.segmentations):
            raise RuntimeError("runtime segmentation count does not match detector output")
        annotations = [
            {"bbox": row["bbox_xywh"], "category_id": row["category_id"]}
            for row in record["annotations"]
        ]
        matches, missed = _match(detections, annotations, args.match_iou_threshold)
        false_positive_count = len(detections) - len(matches)
        counts.segmentation_image_count += 1
        counts.predicted_segmentation_count += len(detections)
        counts.matched_segmentation_count += len(matches)
        counts.false_negative_count += len(missed)
        counts.false_positive_count += false_positive_count
        counts.false_negative_image_count += bool(missed)
        counts.false_positive_image_count += false_positive_count > 0
        for index, segmentation in enumerate(response.segmentations):
            gt_index = matches.get(index)
            if gt_index is None:
                continue
            target = f"bread_{int(annotations[gt_index]['category_id']):02d}"
            if segmentation.status is ItemStatus.APPROVED:
                counts.approved_count += 1
                counts.approved_misrecognition_count += (
                    segmentation.prediction is None or segmentation.prediction.class_id != target
                )
            elif segmentation.status is ItemStatus.UNKNOWN:
                counts.unknown_count += 1
                counts.unknown_top3_candidate_out_count += target not in {
                    candidate.class_id for candidate in segmentation.top3
                }
            else:
                counts.segment_recapture_count += 1
        if missed or false_positive_count:
            details.append(
                {
                    "image_id": record["image_id"],
                    "status": response.status.value,
                    "ground_truth_count": len(annotations),
                    "prediction_count": len(detections),
                    "matched_count": len(matches),
                    "false_negative_count": len(missed),
                    "false_positive_count": false_positive_count,
                }
            )

    metrics = build_runtime_gate_metrics(counts)
    performance = _latency(counts.latencies_ms)
    latency_gate = performance["mean_ms"] <= 100.0 and performance["p95_ms"] <= 100.0
    latency_required = provider == "cuda"
    performance_requirement_met = not latency_required or latency_gate
    development_requirements_met = (
        metrics["operational_gates"]["all_met"]
        and metrics["final_end_to_end_approved_goal_met"]
        and performance_requirement_met
    )
    trace_path = getattr(args, "decision_trace_output", None)
    trace_evidence = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in decision_trace
            ),
            encoding="utf-8",
        )
        trace_evidence = {
            "path": trace_path.resolve().as_posix(),
            "sha256": sha256_file(trace_path),
            "image_count": len(decision_trace),
            "excludes": ["request_id", "processing_time_ms"],
        }
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_1_1_runtime_six_gate",
        "evidence_role": args.evidence_role,
        "dataset_version": args.dataset_version,
        "dataset": {
            "annotation_path": annotation_path.as_posix(),
            "annotation_sha256": sha256_file(annotation_path),
            "image_count": len(records),
        },
        "provider": provider,
        "versions": {
            "worker": package.metadata.worker_version,
            "detector": package.metadata.detector.version,
            "classifier": package.metadata.classifier.version,
        },
        "denominator_contract": {
            "approved_and_object_error_rates": "all_judgeable_ground_truth_objects",
            "recaptured_or_missed_ground_truth_objects_remain_in_denominator": True,
            "unmatched_detector_false_positives_excluded_from_object_denominator": True,
            "unmatched_detector_false_positives_checked_by_image_gate": True,
        },
        "metrics": metrics,
        "performance": {
            **performance,
            "warmup_count": args.warmup_count,
            "scope": "decode+preprocess+detector+classifier+postprocess+decision",
            "mean_and_p95_at_most_100ms": latency_gate,
            "latency_gate_required_for_provider": latency_required,
            "provider_performance_requirement_met": performance_requirement_met,
        },
        "development_requirements_met": development_requirements_met,
        "promotion_eligible": args.evidence_role == "independent" and development_requirements_met,
        "limitations": {
            "development_evidence_is_not_independent_promotion_evidence": (
                args.evidence_role == "development"
            )
        },
        "decision_trace": trace_evidence,
        "error_images": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Bread 1.1 runtime six-gate contract")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation-name", default="instances.json")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-role", choices=("development", "independent"), required=True)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument("--decision-trace-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
