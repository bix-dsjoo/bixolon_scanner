from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from ..contracts import ItemStatus, ScanItem, Status, load_runtime_package_v2
from ..contracts.catalog import load_store_catalog_package, sha256_file
from ..pipeline import DecisionPipeline
from ..pipeline.ports import ClassificationResult, Detection, DetectionResult
from ..runtime.catalog import OnnxCatalogClassifier, OnnxEmbedder
from ..runtime.detector_v2 import build_detector_v2
from ..runtime.imaging import decode_image
from ..runtime.onnx import box_iou
from .scanner_v2_private_preflight import (
    PrivateAnnotation,
    PrivateImageRecord,
    PrivateTestPlan,
    load_private_manifest,
    release_lock_self_sha256,
    validate_locked_gate_tool,
)

ALPHA = 0.05


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def one_sided_error_upper_95(failure_count: int, trial_count: int) -> float | None:
    """One-sided 95% Clopper-Pearson upper bound for an error probability."""
    if trial_count <= 0:
        return None
    if not 0 <= failure_count <= trial_count:
        raise ValueError("failure count must be between zero and the trial count")
    if failure_count == trial_count:
        return 1.0
    return float(beta.ppf(1.0 - ALPHA, failure_count + 1, trial_count - failure_count))


def one_sided_success_lower_95(success_count: int, trial_count: int) -> float | None:
    """One-sided 95% Clopper-Pearson lower bound for a success probability."""
    if trial_count <= 0:
        return None
    if not 0 <= success_count <= trial_count:
        raise ValueError("success count must be between zero and the trial count")
    if success_count == 0:
        return 0.0
    return float(beta.ppf(ALPHA, success_count, trial_count - success_count + 1))


def error_gate(failure_count: int, trial_count: int, maximum: float) -> dict[str, Any]:
    point = rate(failure_count, trial_count)
    upper = one_sided_error_upper_95(failure_count, trial_count)
    return {
        "failure_count": failure_count,
        "trial_count": trial_count,
        "point": point,
        "upper_95": upper,
        "maximum": maximum,
        "passes": point is not None and upper is not None and point <= maximum and upper <= maximum,
    }


def success_gate(success_count: int, trial_count: int, minimum: float) -> dict[str, Any]:
    point = rate(success_count, trial_count)
    lower = one_sided_success_lower_95(success_count, trial_count)
    return {
        "success_count": success_count,
        "trial_count": trial_count,
        "point": point,
        "lower_95": lower,
        "minimum": minimum,
        "passes": point is not None and lower is not None and point >= minimum and lower >= minimum,
    }


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
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
    def __init__(self, detector: Any):
        self.detector = detector
        self.version = detector.version
        self.last_result: DetectionResult | None = None

    def detect(self, image: Any) -> DetectionResult:
        self.last_result = self.detector.detect(image)
        return self.last_result


class RecordingClassifier:
    def __init__(self, classifier: OnnxCatalogClassifier):
        self.classifier = classifier
        self.version = classifier.version
        self.metadata = classifier.metadata
        self.last_result: ClassificationResult | None = None

    def classify(self, image: Any, detections: list[Detection]) -> ClassificationResult:
        self.last_result = self.classifier.classify(image, detections)
        return self.last_result


@dataclass
class AnnotationOutcome:
    item: ScanItem | None = None
    forced_top3: set[str] = field(default_factory=set)


@dataclass
class ImageOutcome:
    status: Status
    false_negative_count: int
    false_positive_count: int
    annotations: dict[str, AnnotationOutcome]
    unmatched_items: list[ScanItem]
    latency_ms: float


def match_items(
    items: list[ScanItem], annotations: list[PrivateAnnotation], threshold: float
) -> tuple[dict[int, int], set[int], set[int]]:
    predictions = [
        Detection(
            item.bbox.x,
            item.bbox.y,
            item.bbox.x + item.bbox.width,
            item.bbox.y + item.bbox.height,
            1.0,
        )
        for item in items
    ]
    targets = [
        Detection(x, y, x + width, y + height, 1.0)
        for x, y, width, height in (annotation.bbox_xywh for annotation in annotations)
    ]
    candidates = sorted(
        (
            (box_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
        ),
        reverse=True,
    )
    matches: dict[int, int] = {}
    used_targets: set[int] = set()
    for iou, prediction_index, target_index in candidates:
        if iou < threshold:
            break
        if prediction_index not in matches and target_index not in used_targets:
            matches[prediction_index] = target_index
            used_targets.add(target_index)
    return (
        matches,
        set(range(len(targets))) - used_targets,
        set(range(len(predictions))) - set(matches),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def claim_private_run(
    path: Path,
    *,
    release_lock_sha256: str,
    preflight_sha256: str,
    manifest_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0",
        "state": "started_private_inference_do_not_rerun",
        "release_lock_sha256": release_lock_sha256,
        "preflight_sha256": preflight_sha256,
        "manifest_sha256": manifest_sha256,
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("owner-private test already has a single-run claim") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def finish_private_run(path: Path, *, state: str, report_sha256: str | None = None) -> None:
    payload = _read_json(path)
    if payload.get("state") != "started_private_inference_do_not_rerun":
        raise RuntimeError("owner-private single-run claim is not in the started state")
    payload["state"] = state
    if report_sha256 is not None:
        payload["report_sha256"] = report_sha256
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_locked_directory(directory: Path, lock: dict[str, Any]) -> None:
    locked_root = Path(str(lock["path"]))
    expected: dict[str, tuple[int, str]] = {}
    for row in lock.get("files", []):
        locked_path = Path(str(row["path"]))
        try:
            relative = locked_path.relative_to(locked_root).as_posix()
        except ValueError as exc:
            raise ValueError("release lock contains an artifact path outside its root") from exc
        expected[relative] = (int(row["size_bytes"]), str(row["sha256"]))
    actual = {
        path.relative_to(directory).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in directory.rglob("*")
        if path.is_file()
    }
    if not expected or actual != expected:
        raise ValueError("runtime or Catalog bytes differ from the pre-private release lock")


def _validate_preflight(
    *,
    release_lock_path: Path,
    preflight_path: Path,
    plan_path: Path,
    manifest_path: Path,
    runtime_dir: Path,
    catalog_dir: Path,
) -> tuple[dict[str, Any], PrivateTestPlan, list[PrivateImageRecord]]:
    release_lock = _read_json(release_lock_path)
    lock_sha256 = release_lock_self_sha256(release_lock)
    validate_locked_gate_tool(release_lock, Path(__file__))
    if release_lock.get("status") != "owner_private_test_pending" or release_lock.get(
        "remaining_gates"
    ) != ["owner_private_locked_production_test"]:
        raise ValueError("candidate is not locked immediately before the private production gate")
    preflight = _read_json(preflight_path)
    if (
        preflight.get("evaluation") != "scanner_2_0_owner_private_preflight"
        or preflight.get("eligible_for_single_private_run") is not True
        or preflight.get("production_inference_executed") is not False
        or preflight.get("release_lock_sha256") != lock_sha256
        or preflight.get("gate_tool_sha256")
        != next(
            (
                row.get("sha256")
                for row in release_lock.get("supply_chain", {}).get("private_gate_tools", [])
                if Path(str(row.get("path", ""))).name == "scanner_v2_private_preflight.py"
            ),
            None,
        )
        or preflight.get("plan_sha256") != sha256_file(plan_path)
        or preflight.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("private preflight is missing, stale, or ineligible")
    plan = PrivateTestPlan.model_validate(_read_json(plan_path))
    records = load_private_manifest(manifest_path)
    if plan.release_lock_sha256 != lock_sha256 or plan.manifest_sha256 != sha256_file(
        manifest_path
    ):
        raise ValueError("private plan no longer targets the locked candidate and manifest")
    if len(records) != preflight.get("image_count") or len(records) != plan.image_count:
        raise ValueError("private image count differs from the model-free preflight")
    _verify_locked_directory(runtime_dir.resolve(), release_lock["artifacts"]["runtime"])
    _verify_locked_directory(catalog_dir.resolve(), release_lock["artifacts"]["catalog"])
    return release_lock, plan, records


def _forced_top3(
    classification: ClassificationResult,
    classifier: RecordingClassifier,
    prediction_index: int,
) -> set[str]:
    ranking = (
        classification.ranking_scores
        if classification.ranking_scores is not None
        else classification.ranking_logits
    )
    order = np.argsort(-ranking[prediction_index], kind="stable")
    return {classifier.metadata.labels[int(index)].class_id for index in order[:3]}


def _scan_private_image(
    *,
    record: PrivateImageRecord,
    path: Path,
    pipeline: DecisionPipeline,
    detector: RecordingDetector,
    classifier: RecordingClassifier,
    runtime: Any,
    ordinal: int,
    match_iou_threshold: float,
) -> ImageOutcome:
    image_bytes = path.read_bytes()
    detector.last_result = None
    classifier.last_result = None
    started = time.perf_counter()
    image = decode_image(
        image_bytes,
        max_bytes=50_000_000,
        max_pixels=50_000_000,
        jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
    )
    try:
        response = pipeline.scan(image, request_id=f"scanner2-private-{ordinal:06d}")
    finally:
        image.close()
    latency_ms = (time.perf_counter() - started) * 1000.0
    outcomes = {annotation.annotation_id: AnnotationOutcome() for annotation in record.annotations}
    if response.status is Status.IMAGE_RECAPTURE:
        return ImageOutcome(
            status=response.status,
            false_negative_count=len(record.annotations),
            false_positive_count=0,
            annotations=outcomes,
            unmatched_items=[],
            latency_ms=latency_ms,
        )
    if response.status is not Status.SEGMENTATION:
        raise RuntimeError("private production evaluation received an ERROR response")
    if detector.last_result is None or classifier.last_result is None:
        raise RuntimeError("private production full path did not record both model results")
    if len(response.segmentations) != len(classifier.last_result.logits):
        raise RuntimeError("private production output count differs from classifier output")
    matches, missed, unmatched = match_items(
        response.segmentations, record.annotations, match_iou_threshold
    )
    for prediction_index, annotation_index in matches.items():
        annotation = record.annotations[annotation_index]
        outcomes[annotation.annotation_id] = AnnotationOutcome(
            item=response.segmentations[prediction_index],
            forced_top3=_forced_top3(classifier.last_result, classifier, prediction_index),
        )
    return ImageOutcome(
        status=response.status,
        false_negative_count=len(missed),
        false_positive_count=len(unmatched),
        annotations=outcomes,
        unmatched_items=[response.segmentations[index] for index in sorted(unmatched)],
        latency_ms=latency_ms,
    )


def certification_report(
    plan: PrivateTestPlan,
    records: list[PrivateImageRecord],
    outcomes: dict[str, ImageOutcome],
) -> dict[str, dict[str, Any]]:
    records_by_id = {record.image_id: record for record in records}
    errors: dict[str, list[bool]] = defaultdict(list)
    successes: dict[str, list[bool]] = defaultdict(list)
    approval_observations: list[bool] = []
    for trial in plan.trials:
        record = records_by_id[trial.image_id]
        outcome = outcomes[trial.image_id]
        annotation = (
            None
            if trial.annotation_id is None
            else next(
                value for value in record.annotations if value.annotation_id == trial.annotation_id
            )
        )
        annotation_outcome = (
            None if annotation is None else outcome.annotations[annotation.annotation_id]
        )
        if trial.endpoint == "approval_safety":
            item = None if annotation_outcome is None else annotation_outcome.item
            if item is not None and item.status is ItemStatus.APPROVED:
                approval_observations.append(
                    item.prediction is not None
                    and annotation is not None
                    and item.prediction.class_id == annotation.target_class_id
                )
        elif trial.endpoint == "detector_fn":
            errors[trial.endpoint].append(
                outcome.status is not Status.SEGMENTATION or outcome.false_negative_count > 0
            )
        elif trial.endpoint == "detector_fp":
            errors[trial.endpoint].append(outcome.false_positive_count > 0)
        elif trial.endpoint == "top3_safety":
            errors[trial.endpoint].append(
                annotation is None
                or annotation_outcome is None
                or annotation.target_class_id not in annotation_outcome.forced_top3
            )
        elif trial.endpoint == "ood_false_approval":
            item = None if annotation_outcome is None else annotation_outcome.item
            errors[trial.endpoint].append(item is not None and item.status is ItemStatus.APPROVED)
        elif trial.endpoint == "image_recapture_recall":
            successes[trial.endpoint].append(outcome.status is Status.IMAGE_RECAPTURE)
        elif trial.endpoint == "unnecessary_image_recapture":
            errors[trial.endpoint].append(outcome.status is Status.IMAGE_RECAPTURE)
        elif trial.endpoint == "invalid_roi_action":
            item = None if annotation_outcome is None else annotation_outcome.item
            successes[trial.endpoint].append(
                item is not None and item.status is ItemStatus.SEGMENT_RECAPTURE
            )

    report = {
        "approval_safety": error_gate(
            sum(not correct for correct in approval_observations),
            len(approval_observations),
            0.001,
        ),
        "detector_fn": error_gate(sum(errors["detector_fn"]), len(errors["detector_fn"]), 0.001),
        "detector_fp": error_gate(sum(errors["detector_fp"]), len(errors["detector_fp"]), 0.001),
        "top3_safety": error_gate(sum(errors["top3_safety"]), len(errors["top3_safety"]), 0.001),
        "ood_false_approval": error_gate(
            sum(errors["ood_false_approval"]), len(errors["ood_false_approval"]), 0.001
        ),
        "image_recapture_recall": success_gate(
            sum(successes["image_recapture_recall"]),
            len(successes["image_recapture_recall"]),
            0.99,
        ),
        "unnecessary_image_recapture": error_gate(
            sum(errors["unnecessary_image_recapture"]),
            len(errors["unnecessary_image_recapture"]),
            0.01,
        ),
        "invalid_roi_action": success_gate(
            sum(successes["invalid_roi_action"]),
            len(successes["invalid_roi_action"]),
            0.99,
        ),
    }
    report["approval_safety"]["denominator"] = (
        "predeclared approval_safety groups that produced an APPROVED output"
    )
    return report


def aggregate_metrics(
    records: list[PrivateImageRecord], outcomes: dict[str, ImageOutcome]
) -> tuple[dict[str, int], dict[str, float | None]]:
    counts: Counter[str] = Counter()
    eligible = [record for record in records if record.expected_image_status == "SEGMENTATION"]
    recapture = [record for record in records if record.expected_image_status == "IMAGE_RECAPTURE"]
    counts["image_count"] = len(records)
    counts["eligible_image_count"] = len(eligible)
    counts["recapture_gt_image_count"] = len(recapture)
    counts["segmentation_image_count"] = sum(
        outcomes[record.image_id].status is Status.SEGMENTATION for record in records
    )
    counts["image_recapture_count"] = sum(
        outcomes[record.image_id].status is Status.IMAGE_RECAPTURE for record in records
    )
    counts["eligible_segmentation_count"] = sum(
        outcomes[record.image_id].status is Status.SEGMENTATION for record in eligible
    )
    counts["judgeable_gt_object_count"] = sum(len(record.annotations) for record in eligible)
    counts["segmentation_fn_image_count"] = sum(
        outcomes[record.image_id].status is Status.SEGMENTATION
        and outcomes[record.image_id].false_negative_count > 0
        for record in eligible
    )
    counts["segmentation_fp_image_count"] = sum(
        outcomes[record.image_id].status is Status.SEGMENTATION
        and outcomes[record.image_id].false_positive_count > 0
        for record in eligible
    )
    counts["eligible_image_recapture_count"] = sum(
        outcomes[record.image_id].status is Status.IMAGE_RECAPTURE for record in eligible
    )
    counts["correct_image_recapture_count"] = sum(
        outcomes[record.image_id].status is Status.IMAGE_RECAPTURE for record in recapture
    )

    for record in eligible:
        outcome = outcomes[record.image_id]
        for annotation in record.annotations:
            annotation_outcome = outcome.annotations[annotation.annotation_id]
            item = annotation_outcome.item
            if (
                annotation.catalog_membership == "in_catalog"
                and annotation.expected_item_status == "APPROVED"
            ):
                counts["forced_top3_trial_count"] += 1
                counts["forced_top3_candidate_out_count"] += (
                    annotation.target_class_id not in annotation_outcome.forced_top3
                )
            if annotation.catalog_membership == "ood":
                counts["ood_object_count"] += 1
                counts["ood_false_approved_count"] += (
                    item is not None and item.status is ItemStatus.APPROVED
                )
            if annotation.expected_item_status == "SEGMENT_RECAPTURE":
                counts["invalid_roi_object_count"] += 1
                counts["invalid_roi_correct_action_count"] += (
                    item is not None and item.status is ItemStatus.SEGMENT_RECAPTURE
                )
            if item is None:
                continue
            if item.status is ItemStatus.APPROVED:
                counts["approved_object_count"] += 1
                correct = (
                    annotation.catalog_membership == "in_catalog"
                    and annotation.expected_item_status == "APPROVED"
                    and item.prediction is not None
                    and item.prediction.class_id == annotation.target_class_id
                )
                counts["correct_approved_count"] += correct
                counts["wrong_approved_count"] += not correct
            elif item.status is ItemStatus.UNKNOWN:
                counts["unknown_top3_object_count"] += 1
                counts["unknown_top3_candidate_out_count"] += annotation.target_class_id not in {
                    candidate.class_id for candidate in item.top3
                }
            else:
                counts["segment_recapture_object_count"] += 1
        for item in outcome.unmatched_items:
            counts["unmatched_output_count"] += 1
            if item.status is ItemStatus.APPROVED:
                counts["approved_output_false_positive_count"] += 1
                counts["wrong_approved_count"] += 1
    for record in recapture:
        outcome = outcomes[record.image_id]
        if outcome.status is not Status.SEGMENTATION:
            continue
        for item in outcome.unmatched_items:
            counts["recapture_gt_unexpected_output_count"] += 1
            if item.status is ItemStatus.APPROVED:
                counts["unsafe_approved_on_recapture_gt_count"] += 1
                counts["approved_output_false_positive_count"] += 1
                counts["wrong_approved_count"] += 1
    counts["approved_output_count"] = (
        counts["approved_object_count"] + counts["approved_output_false_positive_count"]
    )

    gt = counts["judgeable_gt_object_count"]
    predicted_segmentation_eligible = counts["eligible_segmentation_count"]
    metrics = {
        "segmentation_rate_all_images": rate(
            counts["segmentation_image_count"], counts["image_count"]
        ),
        "image_recapture_rate_all_images": rate(
            counts["image_recapture_count"], counts["image_count"]
        ),
        "segmentation_rate_eligible_images": rate(
            counts["eligible_segmentation_count"], counts["eligible_image_count"]
        ),
        "approved_rate_judgeable_gt": rate(counts["approved_object_count"], gt),
        "correct_approved_rate_judgeable_gt": rate(counts["correct_approved_count"], gt),
        "unknown_top3_rate_judgeable_gt": rate(counts["unknown_top3_object_count"], gt),
        "segment_recapture_rate_judgeable_gt": rate(counts["segment_recapture_object_count"], gt),
        "segmentation_image_false_negative_rate": rate(
            counts["segmentation_fn_image_count"], predicted_segmentation_eligible
        ),
        "segmentation_image_false_positive_rate": rate(
            counts["segmentation_fp_image_count"], predicted_segmentation_eligible
        ),
        "approved_object_misrecognition_rate_judgeable_gt": rate(
            counts["wrong_approved_count"], gt
        ),
        "approved_output_misrecognition_rate": rate(
            counts["wrong_approved_count"], counts["approved_output_count"]
        ),
        "unknown_top3_candidate_out_rate_judgeable_gt": rate(
            counts["unknown_top3_candidate_out_count"], gt
        ),
        "forced_top3_candidate_out_rate": rate(
            counts["forced_top3_candidate_out_count"], counts["forced_top3_trial_count"]
        ),
        "ood_false_approval_rate": rate(
            counts["ood_false_approved_count"], counts["ood_object_count"]
        ),
        "image_recapture_recall": rate(
            counts["correct_image_recapture_count"], counts["recapture_gt_image_count"]
        ),
        "unnecessary_image_recapture_rate": rate(
            counts["eligible_image_recapture_count"], counts["eligible_image_count"]
        ),
        "invalid_roi_correct_segment_recapture_recall": rate(
            counts["invalid_roi_correct_action_count"], counts["invalid_roi_object_count"]
        ),
    }
    return dict(sorted(counts.items())), metrics


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.resolve() == args.run_state.resolve():
        raise ValueError("private aggregate report and single-run state must use different files")
    if args.output.exists():
        raise FileExistsError("owner-private aggregate report already exists")
    release_lock, plan, records = _validate_preflight(
        release_lock_path=args.release_lock,
        preflight_path=args.preflight_report,
        plan_path=args.plan,
        manifest_path=args.manifest,
        runtime_dir=args.runtime,
        catalog_dir=args.catalog,
    )
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
    if plan.store_count != 1 or any(record.store_id != args.store_id for record in records):
        raise ValueError("one private evaluation run must target exactly one locked Store Catalog")
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

    private_hashes = {record.image_sha256 for record in records}
    if sha256_file(args.warmup_image) in private_hashes:
        raise ValueError("warm-up image must not be part of the owner-private test")
    warmup_bytes = args.warmup_image.read_bytes()
    warmup = decode_image(
        warmup_bytes,
        max_bytes=50_000_000,
        max_pixels=50_000_000,
        jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
    )
    try:
        for index in range(args.warmup_count):
            pipeline.scan(warmup, request_id=f"scanner2-private-warmup-{index:06d}")
    finally:
        warmup.close()

    dataset_root = args.dataset_root.resolve()
    outcomes: dict[str, ImageOutcome] = {}
    claim_private_run(
        args.run_state,
        release_lock_sha256=release_lock["lock_sha256"],
        preflight_sha256=sha256_file(args.preflight_report),
        manifest_sha256=sha256_file(args.manifest),
    )
    try:
        for ordinal, record in enumerate(records, start=1):
            path = (dataset_root / record.image_path).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError("private image path escaped its mounted root") from exc
            if sha256_file(path) != record.image_sha256:
                raise ValueError("private image changed after model-free preflight")
            outcomes[record.image_id] = _scan_private_image(
                record=record,
                path=path,
                pipeline=pipeline,
                detector=detector,
                classifier=classifier,
                runtime=runtime,
                ordinal=ordinal,
                match_iou_threshold=args.match_iou_threshold,
            )
    except Exception:
        finish_private_run(args.run_state, state="failed_after_private_run_started_do_not_rerun")
        raise

    counts, metrics = aggregate_metrics(records, outcomes)
    certification = certification_report(plan, records, outcomes)
    point_gates = {
        "segmentation_rate": metrics["segmentation_rate_eligible_images"] is not None
        and metrics["segmentation_rate_eligible_images"] >= 0.90,
        "correct_approved_rate": metrics["correct_approved_rate_judgeable_gt"] is not None
        and metrics["correct_approved_rate_judgeable_gt"] >= 0.95,
        "wrong_approved_over_all_gt": metrics["approved_object_misrecognition_rate_judgeable_gt"]
        is not None
        and metrics["approved_object_misrecognition_rate_judgeable_gt"] <= 0.001,
        "wrong_approved_over_approved_output": metrics["approved_output_misrecognition_rate"]
        is not None
        and metrics["approved_output_misrecognition_rate"] <= 0.001,
        "segmentation_image_false_negative_rate": metrics["segmentation_image_false_negative_rate"]
        is not None
        and metrics["segmentation_image_false_negative_rate"] <= 0.001,
        "segmentation_image_false_positive_rate": metrics["segmentation_image_false_positive_rate"]
        is not None
        and metrics["segmentation_image_false_positive_rate"] <= 0.001,
        "forced_top3_candidate_out_rate": metrics["forced_top3_candidate_out_rate"] is not None
        and metrics["forced_top3_candidate_out_rate"] <= 0.001,
        "ood_false_approval_rate": metrics["ood_false_approval_rate"] is not None
        and metrics["ood_false_approval_rate"] <= 0.001,
        "image_recapture_recall": metrics["image_recapture_recall"] is not None
        and metrics["image_recapture_recall"] >= 0.99,
        "unnecessary_image_recapture_rate": metrics["unnecessary_image_recapture_rate"] is not None
        and metrics["unnecessary_image_recapture_rate"] <= 0.01,
        "invalid_roi_correct_action_recall": metrics["invalid_roi_correct_segment_recapture_recall"]
        is not None
        and metrics["invalid_roi_correct_segment_recapture_recall"] >= 0.99,
    }
    statistical_gates = {name: value["passes"] for name, value in certification.items()}
    production_eligible = all(point_gates.values()) and all(statistical_gates.values())
    latencies = [outcome.latency_ms for outcome in outcomes.values()]
    report = {
        "schema_version": "2.0",
        "evaluation": "scanner_2_0_owner_private_production_gate",
        "release_candidate": release_lock["release_candidate"],
        "release_lock_sha256": release_lock["lock_sha256"],
        "dataset": {
            "dataset_id": plan.dataset_id,
            "immutable_revision": plan.immutable_revision,
            "manifest_sha256": sha256_file(args.manifest),
            "image_count": len(records),
            "store_count": plan.store_count,
            "single_run_count_per_private_image": 1,
        },
        "counts": counts,
        "requested_metrics": {
            **metrics,
            "mean_speed_ms": float(np.mean(latencies)),
        },
        "performance": {
            **latency_summary(latencies),
            "scope": "decode+preprocess+detector-ensemble+selective-refinement+embedder+decision",
            "warmup_count_on_non_private_image": args.warmup_count,
        },
        "point_gates": point_gates,
        "statistical_certification": certification,
        "long_term_99_percent_correct_approved_goal_met": metrics[
            "correct_approved_rate_judgeable_gt"
        ]
        is not None
        and metrics["correct_approved_rate_judgeable_gt"] >= 0.99,
        "production_eligible": production_eligible,
        "decision": "promote_exact_locked_candidate" if production_eligible else "reject_candidate",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "provider": args.provider,
        },
        "privacy": {
            "per_image_records_written": False,
            "image_paths_written": False,
            "image_bytes_written": False,
            "embeddings_or_logits_written": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    finish_private_run(
        args.run_state,
        state="completed_private_run_do_not_rerun",
        report_sha256=sha256_file(args.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the owner-private Scanner 2.0 production gate exactly once"
    )
    parser.add_argument("--release-lock", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--warmup-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--signing-key-env", default="BIXOLON_CATALOG_SIGNING_KEY")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--warmup-count", type=int, default=100)
    evaluate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
