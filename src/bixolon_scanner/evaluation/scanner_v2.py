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
from ..runtime.catalog import build_catalog_classifier
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


def _resolve_coco_image_path(annotation_path: Path, dataset_root: Path, value: str) -> Path:
    root = dataset_root.resolve()
    candidates = [root / value, annotation_path.resolve().parent / value]
    valid: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        valid.append(resolved)
        if resolved.is_file():
            return resolved
    if not valid:
        raise ValueError("evaluation image path escaped the dataset root")
    return valid[0]


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
        rows = []
        for image in payload["images"]:
            resolved = _resolve_coco_image_path(path, dataset_root, str(image["file_name"]))
            rows.append(
                {
                    **image,
                    "image_id": int(image["id"]),
                    "image_path": resolved.relative_to(dataset_root.resolve()).as_posix(),
                    "annotations": annotations_by_image.get(int(image["id"]), []),
                    "resolved_path": resolved,
                }
            )
    else:
        rows = [json.loads(line) for line in text.splitlines() if line]
    for row in rows:
        if "resolved_path" not in row:
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

    def detect(self, image) -> DetectionResult:
        self.last_result = self.detector.detect(image)
        return self.last_result


class RecordingClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        self.version = classifier.version
        self.metadata = classifier.metadata
        self.last_result: ClassificationResult | None = None

    def classify(self, image, detections: list[Detection]) -> ClassificationResult:
        self.last_result = self.classifier.classify(image, detections)
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
    refinement_count: int = 0


def evaluate(args: argparse.Namespace) -> dict:
    signing_key_value = os.environ.get(args.signing_key_env, "")
    signing_key = signing_key_value.encode() if signing_key_value else None
    runtime = load_runtime_package_v2(args.runtime)
    catalog = load_store_catalog_package(
        args.catalog,
        signing_key=signing_key,
        expected_store_id=args.store_id,
        expected_key_id=args.key_id,
    )
    detector = RecordingDetector(
        build_detector_v2(
            runtime,
            args.provider,
            args.cuda_dll_dir,
            cpu_detector_workers=args.cpu_detector_workers,
            cpu_intra_op_threads=args.cpu_detector_threads,
        )
    )
    catalog_classifier, embedder = build_catalog_classifier(
        runtime,
        catalog,
        args.provider,
        args.cuda_dll_dir,
        cpu_intra_op_threads=args.cpu_embedder_threads,
    )
    classifier = RecordingClassifier(catalog_classifier)
    pipeline = DecisionPipeline(
        detector,
        classifier,
        classifier.metadata,
        runtime.metadata.quality,
        runtime.metadata.count_verifier,
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
    embedder.warmup()
    classifier_warmup = getattr(catalog_classifier, "warmup", None)
    if callable(classifier_warmup):
        classifier_warmup()
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
    for ordinal, record in enumerate(records, start=1):
        # The public performance contract starts at API-internal decode. File I/O belongs to
        # the benchmark harness, not to the Worker request path, where multipart bytes are
        # already available before decode begins.
        image_bytes = record["resolved_path"].read_bytes()
        started = time.perf_counter()
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
        if detector_result is None:
            raise RuntimeError("detector result was not recorded")
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
        if classification is None or classification.approval_scores is None:
            raise RuntimeError("classifier result was not recorded")
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
        classifier_order = np.argsort(-classification.logits, axis=1, kind="stable")
        retrieval_order = (
            None
            if classification.retrieval_logits is None
            else np.argsort(-classification.retrieval_logits, axis=1, kind="stable")
        )
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
            classifier_top1_index = int(classifier_order[detection_index, 0])
            classifier_top2_index = int(classifier_order[detection_index, 1])
            classifier_top1_logit = float(
                classification.logits[detection_index, classifier_top1_index]
            )
            classifier_top2_logit = float(
                classification.logits[detection_index, classifier_top2_index]
            )
            retrieval_top1_class_id = None
            retrieval_top1_similarity = None
            classifier_retrieval_top1_agreement = None
            if retrieval_order is not None and classification.retrieval_logits is not None:
                retrieval_top1_index = int(retrieval_order[detection_index, 0])
                retrieval_top1_class_id = classifier.metadata.labels[retrieval_top1_index].class_id
                retrieval_top1_similarity = float(
                    classification.retrieval_logits[detection_index, retrieval_top1_index]
                )
                classifier_retrieval_top1_agreement = classifier_top1_index == retrieval_top1_index
            top3_safety_score = (
                None
                if classification.top3_safety_scores is None
                else float(classification.top3_safety_scores[detection_index])
            )
            item_diagnostics.append(
                {
                    "detection_index": detection_index,
                    "detector_class_index": detections[detection_index].class_id,
                    "detector_class_id": (
                        None
                        if detections[detection_index].class_id is None
                        else f"bread_{detections[detection_index].class_id + 1:02d}"
                    ),
                    "detector_class_correct": (
                        None
                        if detections[detection_index].class_id is None
                        else detections[detection_index].class_id + 1
                        == int(record["annotations"][target_index]["category_id"])
                    ),
                    "detector_score": detections[detection_index].score,
                    "target_class_id": target,
                    "classifier_top1_class_id": predicted,
                    "classifier_top2_class_id": classifier.metadata.labels[
                        classifier_top2_index
                    ].class_id,
                    "classifier_top1_correct": classifier_correct,
                    "classifier_top3_hit": top3_hit,
                    "classifier_top1_logit": classifier_top1_logit,
                    "classifier_top2_logit": classifier_top2_logit,
                    "classifier_top2_logit_gap": classifier_top1_logit - classifier_top2_logit,
                    "retrieval_top1_class_id": retrieval_top1_class_id,
                    "retrieval_top1_similarity": retrieval_top1_similarity,
                    "classifier_retrieval_top1_agreement": (classifier_retrieval_top1_agreement),
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
    segmentation_images = counts.segmentation_image_count
    requested = {
        "segmentation_rate": _rate(segmentation_images, counts.image_count),
        "image_recapture_rate": _rate(counts.image_recapture_count, counts.image_count),
        "approved_rate": _rate(counts.approved_count, gt),
        "unknown_top3_rate": _rate(counts.unknown_count, gt),
        "segment_recapture_rate": _rate(counts.segment_recapture_count, gt),
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
    limits = {
        "minimum_correct_approved_rate": 0.99,
        "maximum_false_negative_count": 0,
        "maximum_false_positive_count": 0,
        "maximum_approved_misrecognition_count": 0,
        "maximum_unknown_candidate_out_count": 0,
        "maximum_mean_ms": args.maximum_mean_ms,
        "maximum_p95_ms": args.maximum_p95_ms,
    }
    performance = _latency(counts.latencies_ms)
    full_path_performance = _latency(counts.full_path_latencies_ms)
    recapture_performance = _latency(counts.image_recapture_latencies_ms)
    refinement_performance = _latency(counts.refinement_latencies_ms)
    target_results = {
        "error_count": True,
        "correct_approved_rate": (
            requested["correct_approved_rate"] >= limits["minimum_correct_approved_rate"]
        ),
        "false_negative_count": (
            counts.false_negative_count <= limits["maximum_false_negative_count"]
        ),
        "false_positive_count": (
            counts.false_positive_count <= limits["maximum_false_positive_count"]
        ),
        "approved_misrecognition_count": (
            counts.approved_misrecognition_count <= limits["maximum_approved_misrecognition_count"]
        ),
        "unknown_candidate_out_count": (
            counts.unknown_candidate_out_count <= limits["maximum_unknown_candidate_out_count"]
        ),
        "full_path_performance": (
            full_path_performance["mean_ms"] <= limits["maximum_mean_ms"]
            and full_path_performance["p95_ms"] <= limits["maximum_p95_ms"]
        ),
    }
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace),
        encoding="utf-8",
    )
    report = {
        "schema_version": "2.0",
        "evaluation": f"scanner_0_1_3_full_validation_{counts.image_count}",
        "evaluation_scope": (
            "full_valid_data" if evidence_role == "development_regression" else "stress_diagnostic"
        ),
        "dataset": {
            "manifest_sha256": sha256_file(args.manifest),
            "image_count": counts.image_count,
            "ground_truth_object_count": gt,
            "held_out_test_set": False,
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
        "metrics": requested,
        "performance": {
            **performance,
            "scope": ("decode+preprocess+detector+selective-refinement+embedder+decision"),
            "warmup_count": args.warmup_count,
            "measurement_path": "full_path_only",
            "full_path": full_path_performance,
            "image_recapture_early_exit": recapture_performance,
            "selective_refinement": refinement_performance,
        },
        "targets": limits,
        "target_results": {**target_results, "all_met": all(target_results.values())},
        "trace": {
            "path": args.trace_output.name,
            "sha256": sha256_file(args.trace_output),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "provider": args.provider,
            "cpu_detector_workers": args.cpu_detector_workers,
            "cpu_detector_threads": args.cpu_detector_threads,
            "cpu_embedder_threads": args.cpu_embedder_threads,
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
    parser.add_argument("--key-id")
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--provider", choices=("cuda", "cpu", "openvino"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--cpu-detector-workers", type=int, default=1)
    parser.add_argument("--cpu-detector-threads", type=int, default=0)
    parser.add_argument("--cpu-embedder-threads", type=int, default=0)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument("--expected-image-count", type=int, default=300)
    parser.add_argument("--maximum-mean-ms", type=float, default=100.0)
    parser.add_argument("--maximum-p95-ms", type=float, default=100.0)
    parser.add_argument("--maximum-p99-ms", type=float, default=150.0)
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
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
