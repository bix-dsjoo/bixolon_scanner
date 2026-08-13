from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..imaging import decode_image
from ..inference import build_onnx_adapters
from ..package import load_model_package, sha256_file
from ..pipeline import DecisionPipeline, quality_reasons

SCHEMA_VERSION = "1.0"
DEFAULT_SELECTION_SALT = "bixolon-rpc-full-path-v1"
RPC_OPERATION_DAG_VERSION = "rpc-operational-dag-v1"


def _worker_policy_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the package metadata contract implied by rpc_data_scale.json."""
    detector = config.get("detector")
    training = config.get("training")
    if not isinstance(detector, dict) or not isinstance(training, dict):
        raise ValueError("RPC config must define detector and training sections")
    required_detector = (
        "image_size",
        "max_queries",
        "uncertainty_score_threshold",
        "uncertainty_min_area_ratio",
        "uncertainty_match_iou_threshold",
        "min_object_area_ratio",
        "border_margin_ratio",
        "border_policy",
    )
    missing = [field for field in required_detector if field not in detector]
    if missing:
        raise ValueError("RPC config lacks Worker policy fields: " + ", ".join(missing))
    if "image_size" not in training or "eval_margin_ratio" not in training:
        raise ValueError("RPC config lacks classifier crop/preprocess policy")
    worker = config.get("worker", {})
    if not isinstance(worker, dict):
        raise ValueError("RPC worker configuration must be an object")
    detector_size = int(detector["image_size"])
    classifier_size = int(training["image_size"])
    resize_reducing_gap = float(worker.get("resize_reducing_gap", 1.0))
    return {
        "input": {"jpeg_draft_size": int(worker.get("jpeg_draft_size", 1500))},
        "detector": {
            "input_size": [detector_size, detector_size],
            "color_order": "RGB",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "max_queries": int(detector["max_queries"]),
            "uncertainty_score_threshold": float(detector["uncertainty_score_threshold"]),
            "uncertainty_min_area_ratio": float(detector["uncertainty_min_area_ratio"]),
            "uncertainty_match_iou_threshold": float(detector["uncertainty_match_iou_threshold"]),
            "resize_reducing_gap": resize_reducing_gap,
        },
        "classifier": {
            "input_size": [classifier_size, classifier_size],
            "color_order": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_margin_ratio": float(training["eval_margin_ratio"]),
            "resize_reducing_gap": resize_reducing_gap,
        },
        "quality": {
            "min_object_area_ratio": float(detector["min_object_area_ratio"]),
            "border_margin_ratio": float(detector["border_margin_ratio"]),
            "border_policy": str(detector["border_policy"]),
            "min_sharpness": detector.get("min_sharpness"),
            "min_mean_luminance": detector.get("min_mean_luminance"),
            "max_mean_luminance": detector.get("max_mean_luminance"),
        },
    }


def _export_bridge(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    policy = _worker_policy_from_config(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "rpc-export-worker-policy-v1",
        "source_config_sha256": sha256_file(config_path),
        # This section is consumed directly by training.export --config.
        "export": {
            "detector_size": policy["detector"]["input_size"][0],
            "uncertainty_score_threshold": policy["detector"]["uncertainty_score_threshold"],
            "uncertainty_min_area_ratio": policy["detector"]["uncertainty_min_area_ratio"],
            "uncertainty_match_iou_threshold": policy["detector"][
                "uncertainty_match_iou_threshold"
            ],
            "crop_margin": policy["classifier"]["crop_margin_ratio"],
            "resize_reducing_gap": policy["classifier"]["resize_reducing_gap"],
            "jpeg_draft_size": policy["input"]["jpeg_draft_size"],
            "min_object_area_ratio": policy["quality"]["min_object_area_ratio"],
            "border_margin_ratio": policy["quality"]["border_margin_ratio"],
            "border_policy": policy["quality"]["border_policy"],
            "min_sharpness": policy["quality"]["min_sharpness"],
            "min_mean_luminance": policy["quality"]["min_mean_luminance"],
            "max_mean_luminance": policy["quality"]["max_mean_luminance"],
        },
        # max_queries and preprocessing fields are derived/validated rather than CLI args.
        "expected_package_metadata": policy,
    }


def _metadata_value(metadata: Any, section: str, field: str) -> Any:
    section_value = getattr(metadata, section)
    return getattr(section_value, field)


def _validate_export_worker_policy(package: Any, bridge: dict[str, Any]) -> None:
    expected = bridge.get("expected_package_metadata")
    if not isinstance(expected, dict):
        raise ValueError("export bridge has no expected package metadata")
    for section, fields in expected.items():
        if not isinstance(fields, dict):
            raise ValueError(f"export bridge section is invalid: {section}")
        for field, expected_value in fields.items():
            actual = _metadata_value(package.metadata, section, field)
            if isinstance(actual, tuple):
                actual = list(actual)
            if isinstance(expected_value, float) and expected_value is not None:
                matches = actual is not None and math.isclose(
                    float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                matches = actual == expected_value
            if not matches:
                raise ValueError(f"package Worker policy mismatch: {section}.{field}")


class _CapturingDetector:
    """Keep the detector result while preserving the DecisionPipeline call path."""

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.version = adapter.version
        self.last_result: Any | None = None

    def detect(self, image: Any) -> Any:
        self.last_result = self.adapter.detect(image)
        return self.last_result


class _CapturingClassifier:
    """Count classifier calls without changing inputs, outputs, or versions."""

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.version = adapter.version
        self.call_count = 0

    def classify(self, image: Any, detections: Any) -> Any:
        self.call_count += 1
        return self.adapter.classify(image, detections)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _next_rows_chain(previous: str, row_line: bytes) -> str:
    return _sha256_bytes(bytes.fromhex(previous) + row_line)


def _rows_chain(path: Path) -> tuple[str, int]:
    chain = _sha256_bytes(b"")
    count = 0
    for row_line in path.read_bytes().splitlines(keepends=True):
        if not row_line.strip():
            continue
        chain = _next_rows_chain(chain, row_line)
        count += 1
    return chain, count


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical_bytes(value) + b"\n" for value in values))


def _logical_hashes(paths: dict[str, Path]) -> dict[str, str]:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required artifacts are missing: " + ", ".join(missing))
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _package_artifact_hashes(package_dir: Path, package: Any) -> dict[str, str]:
    paths = {
        "metadata.json": package_dir / "metadata.json",
        package.metadata.detector.filename: package.detector_path,
        package.metadata.classifier.filename: package.classifier_path,
    }
    return _logical_hashes(paths)


def _validate_package_evidence(
    report: dict[str, Any],
    evidence_name: str,
    package: Any,
    expected_hashes: dict[str, str],
) -> None:
    expected_identity = {
        "package_version": package.metadata.package_version,
        "dataset_version": package.metadata.dataset_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
    }
    for field, expected in expected_identity.items():
        if report.get(field) != expected:
            raise ValueError(f"{evidence_name} {field} does not match the package")
    reported_hashes = report.get("package_artifact_sha256")
    if not isinstance(reported_hashes, dict):
        raise ValueError(f"{evidence_name} has no package artifact checksums")
    for filename, expected in expected_hashes.items():
        if reported_hashes.get(filename) != expected:
            raise ValueError(f"{evidence_name} package artifact checksum mismatch: {filename}")


def _validate_package_bridge_metadata(
    package: Any,
    detector_evaluation: dict[str, Any],
    classifier_calibration: dict[str, Any],
) -> None:
    if (
        detector_evaluation.get("detector_role") != "checkout_baseline_operational"
        or detector_evaluation.get("threshold_policy") != "calibration_oof_only"
        or detector_evaluation.get("selection_threshold_policy") != "frozen_calibration_threshold"
        or detector_evaluation.get("frozen_threshold_selection_gate") is not True
        or detector_evaluation.get("selection_target_recall_satisfied") is not True
        or not isinstance(detector_evaluation.get("selection_metrics"), dict)
        or not isinstance(detector_evaluation.get("train_gate_complete_sha256"), str)
    ):
        raise ValueError("package operational baseline detector evidence is invalid")
    comparisons = (
        (
            package.metadata.detector.score_threshold,
            detector_evaluation.get("selected_score_threshold"),
            "detector score threshold",
        ),
        (
            package.metadata.detector.nms_iou_threshold,
            detector_evaluation.get("nms_iou_threshold"),
            "detector NMS threshold",
        ),
        (
            package.metadata.classifier.approval_threshold,
            classifier_calibration.get("approval_threshold"),
            "classifier approval threshold",
        ),
        (
            package.metadata.classifier.temperature,
            classifier_calibration.get("temperature"),
            "classifier temperature",
        ),
    )
    for actual, expected, name in comparisons:
        if expected is None or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"package {name} does not match bridged training metadata")
    if package.metadata.calibration is None or package.metadata.detector_evaluation is None:
        raise ValueError("operational package lacks calibration/evaluation metadata")
    calibration = package.metadata.calibration
    for field in (
        "sample_count",
        "approved_precision",
        "approval_coverage",
        "false_approval_rate_upper_95",
        "risk_control_satisfied",
    ):
        bridge_field = (
            "approved_false_rate_upper_95" if field == "false_approval_rate_upper_95" else field
        )
        if getattr(calibration, field) != classifier_calibration.get(bridge_field):
            raise ValueError(f"package calibration metadata mismatch: {field}")
    metrics = detector_evaluation.get("metrics", {})
    evaluation = package.metadata.detector_evaluation
    for field in ("recall", "precision", "count_accuracy"):
        if not math.isclose(
            float(getattr(evaluation, field)),
            float(metrics.get(field)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"package detector evaluation mismatch: {field}")
    if evaluation.target_recall_satisfied != detector_evaluation.get("target_recall_satisfied"):
        raise ValueError("package detector target-recall disposition mismatch")


def _validate_benchmark_manifest_evidence(
    benchmark_dir: Path,
    benchmark_report: dict[str, Any],
    detector_report_sha256: str,
    test_annotation_sha256: str,
) -> None:
    manifest_path = benchmark_dir / "manifest.json"
    jsonl_path = benchmark_dir / "manifest.jsonl"
    ledger_path = benchmark_dir / "checksums.json"
    if benchmark_report.get("benchmark_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("benchmark report manifest checksum mismatch")
    if benchmark_report.get("benchmark_manifest_checksums_sha256") != sha256_file(ledger_path):
        raise ValueError("benchmark report manifest ledger checksum mismatch")
    manifest = _read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("benchmark manifest has no records")
    jsonl_records = _read_manifest(jsonl_path)
    if jsonl_records != records:
        raise ValueError("benchmark JSON and JSONL records differ")
    image_hashes: dict[str, str] = {}
    for record in records:
        relative = str(record["image_path"])
        if relative in image_hashes:
            raise ValueError("benchmark manifest has duplicate image paths")
        image_path = _safe_manifest_image(benchmark_dir, relative)
        digest = sha256_file(image_path)
        if digest != record.get("source_image_sha256"):
            raise ValueError(f"benchmark image checksum mismatch: {relative}")
        image_hashes[relative] = digest
    actual_image_files = {
        path.relative_to(benchmark_dir).as_posix()
        for path in (benchmark_dir / "images").rglob("*")
        if path.is_file()
    }
    if actual_image_files != set(image_hashes):
        raise ValueError("benchmark image file set does not match manifest records")
    if benchmark_report.get("image_artifact_sha256") != image_hashes:
        raise ValueError("benchmark report image artifact checksums mismatch")

    ledger = _read_json(ledger_path)
    inputs = ledger.get("inputs")
    outputs = ledger.get("outputs")
    if (
        ledger.get("phase") != "benchmark-manifest"
        or not isinstance(inputs, dict)
        or not isinstance(outputs, dict)
    ):
        raise ValueError("benchmark manifest checksum ledger is invalid")
    policy = manifest.get("selection_policy", {})
    if policy.get("instances_test2019_sha256") != test_annotation_sha256:
        raise ValueError(
            "benchmark manifest test annotation checksum is not the final detector evidence"
        )
    expected_inputs = {
        "detector_report": detector_report_sha256,
        "instances_test2019": test_annotation_sha256,
        **{
            f"source_image/{int(record['image_id'])}": image_hashes[str(record["image_path"])]
            for record in records
        },
    }
    if inputs != expected_inputs or policy.get("detector_report_sha256") != detector_report_sha256:
        raise ValueError("benchmark manifest input checksums are not bound to evidence")
    expected_output_paths = {
        "manifest.json": manifest_path,
        "manifest.jsonl": jsonl_path,
        **{relative: _safe_manifest_image(benchmark_dir, relative) for relative in image_hashes},
    }
    if outputs != _logical_hashes(expected_output_paths):
        raise ValueError("benchmark manifest output checksums no longer match artifacts")


def _validate_resume(ledger_path: Path, inputs: dict[str, str], output_dir: Path) -> bool:
    if not ledger_path.is_file():
        return False
    ledger = _read_json(ledger_path)
    if ledger.get("inputs") != inputs:
        raise ValueError("resume input checksums do not match the recorded artifacts")
    outputs = ledger.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("resume checksum ledger has no output checksums")
    for relative, expected in outputs.items():
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"resume output checksum mismatch: {relative}")
    return True


def _validate_full_dataset_inputs(
    output_root: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Path]]:
    detector_dir = output_root / "detector"
    detector_complete_path = detector_dir / "baseline" / "complete.json"
    detector_complete = _read_json(detector_complete_path)
    if (
        detector_complete.get("checkpoint_set") != "baseline"
        or detector_complete.get("role") == "train_gate_only"
    ):
        raise ValueError("operational detector must be the immutable checkout baseline")
    artifact_root = str(detector_complete.get("artifact_root", "."))
    threshold_path = (detector_dir / artifact_root / "threshold.json").resolve()
    try:
        threshold_path.relative_to(detector_dir.resolve())
    except ValueError as exc:
        raise ValueError("active detector threshold escapes detector directory") from exc
    train_gate_complete_path = detector_dir / "train-gate" / "complete.json"
    train_gate_complete = _read_json(train_gate_complete_path)
    if train_gate_complete.get("role") != "train_gate_only":
        raise ValueError("detector train-gate marker has an invalid role")
    selection_gate = train_gate_complete.get("baseline_frozen_selection_gate")
    if (
        not isinstance(selection_gate, dict)
        or selection_gate.get("selection_threshold_policy") != "frozen_calibration_threshold"
        or selection_gate.get("frozen_threshold_selection_gate") is not True
        or selection_gate.get("selection_target_recall_satisfied") is not True
    ):
        raise ValueError("baseline frozen-threshold selection gate is invalid")
    final_complete_path = detector_dir / "final" / "complete.json"
    final_complete = _read_json(final_complete_path)
    if (
        final_complete.get("contract") != "rpc-final-detector-baseline-val-all-v1"
        or final_complete.get("target_adaptation_stage") != "disabled_train_gate_only"
        or final_complete.get("operational_detector_role")
        != "checkout_baseline_val_all_operational"
        or final_complete.get("train_gate_role") != "offline_roi_train_gate_only"
    ):
        raise ValueError("operational final detector role is invalid")
    checkpoint_root = detector_dir / "final" / "stage-a-base" / "best"
    checkpoint_candidates = [
        path
        for path in (
            checkpoint_root / "model.safetensors",
            checkpoint_root / "pytorch_model.bin",
        )
        if path.is_file()
    ]
    if len(checkpoint_candidates) != 1:
        raise ValueError("final detector checkpoint weights are missing or ambiguous")
    final_checkpoint_path = checkpoint_candidates[0]
    experiment_path = output_root / "prepared" / "experiment.json"
    lock_path = output_root / "model_lock.json"
    config = _read_json(config_path)
    experiment = _read_json(experiment_path)
    lock = _read_json(lock_path)
    if config.get("experiment", {}).get("mode") != "full_dataset":
        raise ValueError("RPC operational packaging requires experiment.mode=full_dataset")
    expected_count = int(config["experiment"]["expected_num_classes"])
    if expected_count != 200:
        raise ValueError("RPC operational packaging requires exactly 200 classes")
    if (
        experiment.get("mode") != "full_dataset"
        or int(experiment.get("category_count", -1)) != expected_count
    ):
        raise ValueError("prepared experiment is not the locked 200-class full dataset")
    if lock.get("mode") != "full_dataset" or not lock.get("model_run"):
        raise ValueError("model_lock.json does not identify a full-dataset model run")
    if (
        lock.get("operational_detector_role") != "checkout_baseline_val_all_operational"
        or lock.get("train_gate_role") != "offline_roi_train_gate_only"
    ):
        raise ValueError("model lock detector role separation is invalid")
    configured_seeds = config["experiment"].get("seeds")
    if configured_seeds:
        if len(configured_seeds) != 1:
            raise ValueError("full-dataset operational packaging requires one locked seed")
        expected_run = f"runs/full/seed{int(configured_seeds[0])}"
        if str(lock["model_run"]).replace("\\", "/") != expected_run:
            raise ValueError("model lock run does not match the configured full-dataset seed")
    run_dir = output_root / str(lock["model_run"])
    calibration_path = run_dir / "calibration.json"
    paths = {
        "config": config_path,
        "detector_complete": detector_complete_path,
        "detector_threshold": threshold_path,
        "detector_train_gate_complete": train_gate_complete_path,
        "final_detector_complete": final_complete_path,
        "final_detector_checkpoint": final_checkpoint_path,
        "prepared_experiment": experiment_path,
        "classifier_calibration": calibration_path,
        "model_lock": lock_path,
    }
    _logical_hashes(paths)
    locked_artifacts = {
        "checkpoint_sha256": run_dir / "best.pt",
        "calibration_sha256": calibration_path,
        "selection_report_sha256": run_dir / "selection_report.json",
    }
    for field, path in locked_artifacts.items():
        if field in lock:
            if not path.is_file() or sha256_file(path) != lock[field]:
                raise ValueError(f"model lock checksum mismatch: {path.name}")
    detector_lock_artifacts = {
        "rpc_config_sha256": config_path,
        "active_detector_complete_sha256": detector_complete_path,
        "active_detector_threshold_sha256": threshold_path,
        "detector_train_gate_complete_sha256": train_gate_complete_path,
        "final_detector_complete_sha256": final_complete_path,
        "final_detector_checkpoint_sha256": final_checkpoint_path,
    }
    for field, path in detector_lock_artifacts.items():
        if not path.is_file() or lock.get(field) != sha256_file(path):
            raise ValueError(f"model lock checksum mismatch: {field}")
    if (
        final_complete.get("active_threshold_sha256") != lock["active_detector_threshold_sha256"]
        or final_complete.get("stage_a_checkpoint_sha256")
        != lock["final_detector_checkpoint_sha256"]
        or final_complete.get("train_gate_complete_sha256")
        != lock["detector_train_gate_complete_sha256"]
    ):
        raise ValueError("final detector complete marker is not bound to the model lock")
    return config, experiment, lock, run_dir, paths


def _rpc_labels(categories: Any, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(categories, list):
        raise ValueError("prepared experiment categories must be an array")
    ordered = sorted(categories, key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in ordered]
    if ids != list(range(1, expected_count + 1)):
        raise ValueError("RPC category IDs must be contiguous 1..200 in logit order")
    names = [str(row["name"]).strip() for row in ordered]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("RPC category names must be non-empty and unique")
    return [
        {
            "logit_index": category_id - 1,
            "category_id": category_id,
            "class_id": str(category_id),
            "class_name": class_name,
        }
        for category_id, class_name in zip(ids, names)
    ]


def package_inputs(
    output_root: Path,
    config_path: Path,
    *,
    destination: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Bridge locked RPC training artifacts into the generic export contracts."""
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    destination = (destination or output_root / "package-inputs").resolve()
    config, experiment, _lock, run_dir, input_paths = _validate_full_dataset_inputs(
        output_root, config_path
    )
    input_hashes = _logical_hashes(input_paths)
    ledger_path = destination / "checksums.json"
    if resume and _validate_resume(ledger_path, input_hashes, destination):
        return _read_json(ledger_path)

    threshold = _read_json(input_paths["detector_threshold"])
    metrics = threshold.get("calibration_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("detector threshold has no calibration_metrics")
    if threshold.get("threshold_policy") != "calibration_oof_only":
        raise ValueError("operational detector requires immutable baseline calibration")
    selection_gate = _read_json(input_paths["detector_train_gate_complete"])[
        "baseline_frozen_selection_gate"
    ]
    if float(selection_gate["selection_score_threshold"]) != float(
        threshold["selected_score_threshold"]
    ):
        raise ValueError("baseline frozen-threshold selection score is invalid")
    detector_options = config.get("detector", {})
    detector_evaluation = {
        "schema_version": SCHEMA_VERSION,
        "threshold_policy": threshold.get("threshold_policy", "calibration_oof_only"),
        "selected_score_threshold": float(threshold["selected_score_threshold"]),
        "nms_iou_threshold": float(detector_options["nms_iou_threshold"]),
        "target_recall": float(threshold["target_recall"]),
        "target_recall_satisfied": bool(threshold["target_recall_satisfied"]),
        "metrics": metrics,
        "selection_threshold_policy": selection_gate["selection_threshold_policy"],
        "selection_score_threshold": float(selection_gate["selection_score_threshold"]),
        "selection_metrics": selection_gate["selection_metrics"],
        "selection_target_recall_satisfied": True,
        "frozen_threshold_selection_gate": True,
        "detector_role": "checkout_baseline_operational",
        "train_gate_complete_sha256": sha256_file(input_paths["detector_train_gate_complete"]),
    }

    calibration = _read_json(run_dir / "calibration.json")
    matched = int(calibration["matched_count"])
    unmatched = int(calibration["unmatched_detector_count"])
    if matched < 0 or unmatched < 0:
        raise ValueError("classifier calibration counts must be non-negative")
    classifier_calibration = dict(calibration)
    classifier_calibration["sample_count"] = matched + unmatched

    expected_count = int(config["experiment"]["expected_num_classes"])
    labels = _rpc_labels(experiment.get("categories"), expected_count)
    dataset_identity = {
        "contract": "rpc2019-coco-category-logit-order-v1",
        "labels": labels,
        "source_hashes": dict(sorted(experiment.get("source_hashes", {}).items())),
    }
    dataset_digest = _sha256_bytes(_canonical_bytes(dataset_identity))
    manifest_metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": f"rpc2019-{dataset_digest}",
        "dataset_identity_sha256": dataset_digest,
        "category_count": expected_count,
        "labels": labels,
    }
    export_bridge = _export_bridge(config, config_path)

    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "detector-evaluation.json": detector_evaluation,
        "classifier-calibration.json": classifier_calibration,
        "manifest-metadata.json": manifest_metadata,
        "export-config.json": export_bridge,
    }
    for name, value in outputs.items():
        _write_json(destination / name, value)
    output_hashes = {name: sha256_file(destination / name) for name in sorted(outputs)}
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "phase": "package-inputs",
        "inputs": input_hashes,
        "outputs": output_hashes,
    }
    _write_json(ledger_path, ledger)
    return ledger


def _annotations_by_image(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        grouped[int(annotation["image_id"])].append(
            {
                "annotation_id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "bbox_xywh": [float(value) for value in annotation["bbox"]],
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )
    for image_annotations in grouped.values():
        image_annotations.sort(key=lambda row: row["annotation_id"])
    return grouped


def _safe_source_image(dataset_root: Path, filename: str) -> Path:
    image_root = (dataset_root / "test2019").resolve()
    source = (image_root / filename).resolve()
    try:
        source.relative_to(image_root)
    except ValueError as exc:
        raise ValueError("COCO image filename escapes test2019") from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _safe_manifest_image(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    source = (root / relative_path).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("Worker evaluation image path escapes dataset root") from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def build_benchmark_manifest(
    output_root: Path,
    dataset_root: Path,
    *,
    detector_report_path: Path | None = None,
    annotation_path: Path | None = None,
    destination: Path | None = None,
    max_images: int = 1000,
    salt: str = DEFAULT_SELECTION_SALT,
    resume: bool = False,
) -> dict[str, Any]:
    """Select deterministic expected-full-path RPC frames and materialize them."""
    if max_images < 1 or max_images > 1000:
        raise ValueError("max_images must be in 1..1000")
    output_root = output_root.resolve()
    dataset_root = dataset_root.resolve()
    detector_report_path = (
        detector_report_path or output_root / "test" / "detector_report.json"
    ).resolve()
    annotation_path = (annotation_path or dataset_root / "instances_test2019.json").resolve()
    destination = (destination or output_root / "benchmark").resolve()
    report = _read_json(detector_report_path)
    coco = _read_json(annotation_path)
    images = {int(row["id"]): row for row in coco.get("images", [])}
    annotations = _annotations_by_image(coco)
    eligible = [row for row in report.get("outcomes", []) if row.get("recapture_reasons") == []]
    if not eligible:
        raise ValueError("detector report has no expected full-path test frames")
    missing_ids = sorted(
        int(row["image_id"]) for row in eligible if int(row["image_id"]) not in images
    )
    if missing_ids:
        raise ValueError(f"detector outcomes are absent from COCO images: {missing_ids[:5]}")
    annotation_sha = sha256_file(annotation_path)
    report_sha = sha256_file(detector_report_path)
    selected = sorted(
        eligible,
        key=lambda row: (
            hashlib.sha256(f"{salt}:{int(row['image_id'])}".encode("utf-8")).hexdigest(),
            int(row["image_id"]),
        ),
    )[:max_images]

    source_paths: dict[int, Path] = {}
    source_hashes: dict[int, str] = {}
    for outcome in selected:
        image_id = int(outcome["image_id"])
        source = _safe_source_image(dataset_root, str(images[image_id]["file_name"]))
        source_paths[image_id] = source
        source_hashes[image_id] = sha256_file(source)
    input_hashes = {
        "detector_report": report_sha,
        "instances_test2019": annotation_sha,
        **{
            f"source_image/{image_id}": digest for image_id, digest in sorted(source_hashes.items())
        },
    }
    ledger_path = destination / "checksums.json"
    if resume and _validate_resume(ledger_path, input_hashes, destination):
        return _read_json(destination / "manifest.json")

    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for stale_image in image_dir.iterdir():
        if stale_image.is_file():
            stale_image.unlink()
    records: list[dict[str, Any]] = []
    for selection_index, outcome in enumerate(selected):
        image_id = int(outcome["image_id"])
        image_row = images[image_id]
        suffix = Path(str(image_row["file_name"])).suffix.lower()
        destination_name = f"{image_id}{suffix}"
        target = image_dir / destination_name
        if target.exists():
            target.unlink()
        try:
            os.link(source_paths[image_id], target)
            materialization = "hardlink"
        except OSError:
            shutil.copy2(source_paths[image_id], target)
            materialization = "copy"
        image_annotations = annotations.get(image_id, [])
        if int(outcome.get("ground_truth_count", len(image_annotations))) != len(image_annotations):
            raise ValueError(f"ground-truth count mismatch for image {image_id}")
        records.append(
            {
                "record_type": "detection",
                "split": "test",
                "source": "rpc_test2019_benchmark",
                "selection_index": selection_index,
                "selection_policy": "salted_sha256_expected_full_path",
                "image_id": image_id,
                "image_path": f"images/{destination_name}",
                "source_image_sha256": source_hashes[image_id],
                "width": int(image_row["width"]),
                "height": int(image_row["height"]),
                "level": str(image_row.get("level", outcome.get("level", "unknown"))),
                "ground_truth_count": len(image_annotations),
                "expected_detection_count": int(
                    outcome.get("detection_count", len(image_annotations))
                ),
                "expected_recapture_reasons": [],
                "materialization": materialization,
                "annotations": image_annotations,
            }
        )
    manifest_jsonl = destination / "manifest.jsonl"
    _write_jsonl(manifest_jsonl, records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": {
            "name": "salted_sha256_expected_full_path",
            "salt": salt,
            "maximum_images": max_images,
            "eligible_image_count": len(eligible),
            "selected_image_count": len(records),
            "detector_report_sha256": report_sha,
            "instances_test2019_sha256": annotation_sha,
        },
        "records": records,
    }
    manifest_json = destination / "manifest.json"
    _write_json(manifest_json, manifest)
    output_paths = {
        "manifest.json": manifest_json,
        "manifest.jsonl": manifest_jsonl,
        **{record["image_path"]: destination / record["image_path"] for record in records},
    }
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "phase": "benchmark-manifest",
        "inputs": input_hashes,
        "outputs": _logical_hashes(output_paths),
    }
    _write_json(ledger_path, ledger)
    return manifest


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = _read_json(path)
    records = payload.get("records")
    if isinstance(records, list):
        return records
    if isinstance(payload.get("images"), list) and isinstance(payload.get("annotations"), list):
        annotations = _annotations_by_image(payload)
        return [
            {
                "record_type": "detection",
                "split": "test",
                "source": "rpc_test2019",
                "image_id": int(image["id"]),
                "image_path": f"test2019/{image['file_name']}",
                "width": int(image["width"]),
                "height": int(image["height"]),
                "level": str(image.get("level", "unknown")),
                "ground_truth_count": len(annotations.get(int(image["id"]), [])),
                "annotations": annotations.get(int(image["id"]), []),
            }
            for image in sorted(payload["images"], key=lambda row: int(row["id"]))
        ]
    raise ValueError("JSON manifest is neither a benchmark manifest nor RPC COCO")


def _manifest_kind(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    payload = _read_json(path)
    if isinstance(payload.get("images"), list) and isinstance(payload.get("annotations"), list):
        return "rpc_coco_test"
    if isinstance(payload.get("records"), list):
        return "benchmark_records"
    return "unknown"


def _image_ids_sha256(image_ids: Iterable[int]) -> str:
    return _sha256_bytes(_canonical_bytes(sorted(int(image_id) for image_id in image_ids)))


def _source_set_sha256(rows: Iterable[dict[str, Any]]) -> str:
    canonical = [
        {
            "image_id": int(row["image_id"]),
            "source_image_sha256": str(row["source_image_sha256"]),
        }
        for row in rows
    ]
    return _sha256_bytes(_canonical_bytes(canonical))


def _worker_manifest_provenance(
    manifest_path: Path,
    records: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    rows_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    state = _read_json(state_path)
    kind = _manifest_kind(manifest_path)
    manifest_ids = [int(record["image_id"]) for record in records]
    completed_ids = [int(row["image_id"]) for row in completed]
    return {
        "contract": "sealed-rpc-test-manifest-v1",
        "manifest_format": kind,
        "sealed_full_test": kind == "rpc_coco_test",
        "manifest_sha256": sha256_file(manifest_path),
        "test_annotation_sha256": (sha256_file(manifest_path) if kind == "rpc_coco_test" else None),
        "manifest_row_count": len(records),
        "evaluated_row_count": len(completed),
        "manifest_image_ids_sha256": _image_ids_sha256(manifest_ids),
        "evaluated_image_ids_sha256": _image_ids_sha256(completed_ids),
        "source_image_set_sha256": _source_set_sha256(completed),
        "rows_filename": rows_path.name,
        "rows_file_sha256": sha256_file(rows_path),
        "rows_chain_sha256": state.get("rows_sha256"),
        "state_filename": state_path.name,
        "state_file_sha256": sha256_file(state_path),
        "state_completed_count": int(state.get("completed_count", -1)),
        "state_input_fingerprint": state.get("input_fingerprint"),
        "state_rows_checksum_scheme": state.get("rows_checksum_scheme"),
    }


def _evidence_sibling(report_path: Path, filename: Any, label: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(f"Worker accuracy provenance has invalid {label} filename")
    return report_path.parent / filename


def _validate_worker_test_manifest_evidence(
    worker_report_path: Path,
    worker_report: dict[str, Any],
    detector_report: dict[str, Any],
    test_annotation_sha256: str,
) -> dict[str, Any]:
    provenance = worker_report.get("test_manifest")
    if not isinstance(provenance, dict):
        raise ValueError("Worker accuracy report has no sealed test manifest provenance")
    if provenance.get("contract") != "sealed-rpc-test-manifest-v1" or not provenance.get(
        "sealed_full_test"
    ):
        raise ValueError("Worker accuracy was not evaluated on a sealed full test manifest")
    if provenance.get("manifest_format") != "rpc_coco_test":
        raise ValueError("Worker accuracy used an arbitrary or partial manifest")
    for field in ("manifest_sha256", "test_annotation_sha256"):
        if provenance.get(field) != test_annotation_sha256:
            raise ValueError("Worker accuracy manifest is not the final test annotation")

    outcomes = detector_report.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("final detector report has no test outcomes")
    detector_ids = [int(row["image_id"]) for row in outcomes]
    if len(detector_ids) != len(set(detector_ids)):
        raise ValueError("final detector report contains duplicate test image IDs")
    expected_count = len(detector_ids)
    counts = (
        provenance.get("manifest_row_count"),
        provenance.get("evaluated_row_count"),
        provenance.get("state_completed_count"),
        worker_report.get("image_count"),
    )
    if any(int(count) != expected_count for count in counts):
        raise ValueError("Worker accuracy did not evaluate the complete sealed test manifest")
    expected_ids_sha256 = _image_ids_sha256(detector_ids)
    if provenance.get("manifest_image_ids_sha256") != expected_ids_sha256:
        raise ValueError("Worker accuracy manifest image IDs differ from detector final test")
    if provenance.get("evaluated_image_ids_sha256") != expected_ids_sha256:
        raise ValueError("Worker accuracy rows are not the complete test manifest")

    rows_path = _evidence_sibling(worker_report_path, provenance.get("rows_filename"), "rows")
    state_path = _evidence_sibling(worker_report_path, provenance.get("state_filename"), "state")
    if not rows_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("Worker accuracy rows/state evidence is missing")
    if sha256_file(rows_path) != provenance.get("rows_file_sha256"):
        raise ValueError("Worker accuracy rows artifact checksum mismatch")
    if sha256_file(state_path) != provenance.get("state_file_sha256"):
        raise ValueError("Worker accuracy state artifact checksum mismatch")
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_chain_sha256, scanned_count = _rows_chain(rows_path)
    state = _read_json(state_path)
    if scanned_count != expected_count or len(rows) != expected_count:
        raise ValueError("Worker accuracy row artifact is incomplete")
    if (
        rows_chain_sha256 != provenance.get("rows_chain_sha256")
        or state.get("rows_sha256") != rows_chain_sha256
    ):
        raise ValueError("Worker accuracy rows chain checksum mismatch")
    if state.get("input_fingerprint") != worker_report.get("input_fingerprint"):
        raise ValueError("Worker accuracy state input fingerprint mismatch")
    if state.get("rows_checksum_scheme") != "sha256-chain-v1":
        raise ValueError("Worker accuracy state checksum scheme is unsupported")
    if _image_ids_sha256(int(row["image_id"]) for row in rows) != expected_ids_sha256:
        raise ValueError("Worker accuracy row IDs differ from detector final test")
    if _source_set_sha256(rows) != provenance.get("source_image_set_sha256"):
        raise ValueError("Worker accuracy test source image hash mismatch")
    return provenance


def _xywh_to_xyxy(box: Iterable[float]) -> np.ndarray:
    x, y, width, height = [float(value) for value in box]
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _match_items(
    item_boxes: list[np.ndarray], annotations: list[dict[str, Any]], threshold: float
) -> dict[int, int]:
    """Return a deterministic maximum-cardinality thresholded bipartite match."""
    gt_boxes = [_xywh_to_xyxy(row["bbox_xywh"]) for row in annotations]
    adjacency: dict[int, list[int]] = {}
    for item_index, item_box in enumerate(item_boxes):
        eligible = [(gt_index, _iou(item_box, gt_box)) for gt_index, gt_box in enumerate(gt_boxes)]
        adjacency[item_index] = [
            gt_index
            for gt_index, _overlap in sorted(
                (candidate for candidate in eligible if candidate[1] >= threshold),
                key=lambda value: (-value[1], value[0]),
            )
        ]

    gt_owner: dict[int, int] = {}

    def augment(item_index: int, visited_gt: set[int]) -> bool:
        for gt_index in adjacency[item_index]:
            if gt_index in visited_gt:
                continue
            visited_gt.add(gt_index)
            owner = gt_owner.get(gt_index)
            if owner is None or augment(owner, visited_gt):
                gt_owner[gt_index] = item_index
                return True
        return False

    for item_index in sorted(adjacency, key=lambda index: (len(adjacency[index]), index)):
        augment(item_index, set())
    return {item_index: gt_index for gt_index, item_index in sorted(gt_owner.items())}


def _expected_hard_gate_reasons(
    image: Any, detection_result: Any, package_metadata: Any
) -> list[str]:
    reasons = quality_reasons(image, detection_result, package_metadata.quality)
    if detection_result.uncertain_candidate_count:
        reasons.append("DETECTOR_UNCERTAIN_OBJECT")
    count_metadata = package_metadata.count_verifier
    if count_metadata is not None:
        if detection_result.verified_count is None or detection_result.count_confidence is None:
            raise ValueError("count verifier result is missing")
        if detection_result.count_confidence < count_metadata.confidence_threshold:
            reasons.append("DETECTOR_COUNT_UNCERTAIN")
        elif detection_result.verified_count != len(detection_result.detections):
            reasons.append("DETECTOR_COUNT_MISMATCH")
    return list(dict.fromkeys(reasons))


def aggregate_worker_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate resumable, response-surface Worker evaluation rows."""
    image_count = len(rows)
    ground_truth = sum(int(row["ground_truth_count"]) for row in rows)
    predictions = sum(int(row["detector_prediction_count"]) for row in rows)
    matched = sum(int(row["detector_matched_count"]) for row in rows)
    count_correct = sum(bool(row["detector_count_correct"]) for row in rows)
    classifier_items = sum(int(row["classifier_item_count"]) for row in rows)
    classifier_matched = sum(int(row["classifier_item_matched_count"]) for row in rows)
    approved = sum(int(row["approved_count"]) for row in rows)
    approved_correct = sum(int(row["approved_correct_count"]) for row in rows)
    unknown = sum(int(row["unknown_count"]) for row in rows)
    unknown_matched = sum(int(row["unknown_matched_count"]) for row in rows)
    unknown_unmatched = sum(int(row["unknown_unmatched_count"]) for row in rows)
    unknown_top1_correct = sum(int(row.get("unknown_top1_correct_count", 0)) for row in rows)
    unknown_top3_correct = sum(int(row["unknown_top3_correct_count"]) for row in rows)
    full_path_ms = [
        float(row["processing_time_ms"]) for row in rows if bool(row["classifier_executed"])
    ]

    def ratio(numerator: int, denominator: int, *, empty: float | None = 0.0):
        return numerator / denominator if denominator else empty

    return {
        "image_count": image_count,
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
        "detector": {
            "ground_truth_count": ground_truth,
            "prediction_count": predictions,
            "matched_count": matched,
            "recall": ratio(matched, ground_truth),
            "precision": ratio(matched, predictions),
            "count_accuracy": ratio(count_correct, image_count),
        },
        "classifier": {
            "item_count": classifier_items,
            "item_matched_count": classifier_matched,
            "approved_count": approved,
            "approved_correct_count": approved_correct,
            "approved_precision": ratio(approved_correct, approved, empty=None),
            "approval_coverage": ratio(approved, classifier_items),
            "normal_matched_top1_accuracy": ratio(
                approved_correct + unknown_top1_correct,
                classifier_matched,
                empty=None,
            ),
            "unknown_count": unknown,
            "unknown_matched_count": unknown_matched,
            "unknown_unmatched_count": unknown_unmatched,
            "unknown_top1_correct_count": unknown_top1_correct,
            "unknown_top3_correct_count": unknown_top3_correct,
            "unknown_top3_accuracy": ratio(unknown_top3_correct, unknown_matched, empty=None),
        },
        "pipeline_contract": {
            "expected_hard_gate_image_count": sum(bool(row["expected_hard_gate"]) for row in rows),
            "hard_gate_classifier_called_violations": sum(
                bool(row["hard_gate_classifier_called_violation"]) for row in rows
            ),
            "hard_gate_response_status_violations": sum(
                bool(row["hard_gate_response_status_violation"]) for row in rows
            ),
            "hard_gate_classifier_version_violations": sum(
                bool(row["hard_gate_classifier_version_violation"]) for row in rows
            ),
            "hard_gate_reason_mismatch_violations": sum(
                bool(row["hard_gate_reason_mismatch_violation"]) for row in rows
            ),
            "hard_gate_contract_violations": sum(
                bool(row["hard_gate_contract_violation"]) for row in rows
            ),
            "classifier_execution_version_mismatch_violations": sum(
                bool(row["classifier_execution_version_mismatch"]) for row in rows
            ),
            "classifier_call_count_violations": sum(
                bool(row["classifier_call_count_violation"]) for row in rows
            ),
            "pipeline_contract_violations": sum(
                bool(row["pipeline_contract_violation"]) for row in rows
            ),
        },
        "full_path_latency_ms": {
            "count": len(full_path_ms),
            "p50": float(np.percentile(full_path_ms, 50)) if full_path_ms else None,
            "p95": float(np.percentile(full_path_ms, 95)) if full_path_ms else None,
            "p99": float(np.percentile(full_path_ms, 99)) if full_path_ms else None,
        },
    }


def worker_eval(
    package_dir: Path,
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path,
    *,
    rows_path: Path | None = None,
    provider: str = "cuda",
    cuda_dll_dir: Path | None = None,
    match_iou_threshold: float = 0.5,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate actual packaged ONNX behavior exclusively through DecisionPipeline."""
    if not math.isfinite(match_iou_threshold) or not 0.0 < match_iou_threshold <= 1.0:
        raise ValueError("match_iou_threshold must be finite and in (0, 1]")
    package_dir = package_dir.resolve()
    manifest_path = manifest_path.resolve()
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    rows_path = (rows_path or output_path.with_suffix(".jsonl")).resolve()
    package = load_model_package(package_dir)
    records = _read_manifest(manifest_path)
    if not records:
        raise ValueError("Worker evaluation manifest is empty")
    record_ids = [int(record["image_id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Worker evaluation manifest contains duplicate image IDs")
    detector_adapter, classifier, selected_provider = build_onnx_adapters(
        package, provider, cuda_dll_dir=cuda_dll_dir
    )
    package_inputs = _package_artifact_hashes(package_dir, package)
    fingerprint = _sha256_bytes(
        _canonical_bytes(
            {
                "manifest_sha256": sha256_file(manifest_path),
                "package": package_inputs,
                "match_iou_threshold": match_iou_threshold,
                "requested_provider": provider,
                "selected_provider": selected_provider,
            }
        )
    )
    state_path = rows_path.with_suffix(rows_path.suffix + ".state.json")
    completed: list[dict[str, Any]] = []
    rows_chain_sha256 = _sha256_bytes(b"")
    if resume and rows_path.is_file() != state_path.is_file():
        raise ValueError("Worker evaluation resume rows/state are incomplete")
    if resume and rows_path.is_file():
        state = _read_json(state_path)
        if state.get("input_fingerprint") != fingerprint:
            raise ValueError("Worker evaluation resume inputs changed")
        rows_chain_sha256, scanned_count = _rows_chain(rows_path)
        if state.get("rows_sha256") != rows_chain_sha256:
            raise ValueError("Worker evaluation resume rows checksum mismatch")
        completed = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if int(state.get("completed_count", -1)) != len(completed) or scanned_count != len(
            completed
        ):
            raise ValueError("Worker evaluation resume completed count mismatch")
    else:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        rows_path.write_text("", encoding="utf-8")
        _write_json(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "input_fingerprint": fingerprint,
                "requested_provider": provider,
                "selected_provider": selected_provider,
                "completed_count": 0,
                "rows_sha256": rows_chain_sha256,
                "rows_checksum_scheme": "sha256-chain-v1",
            },
        )
    completed_ids = {int(row["image_id"]) for row in completed}
    if len(completed_ids) != len(completed):
        raise ValueError("Worker evaluation resume rows contain duplicate image IDs")
    manifest_ids = [int(record["image_id"]) for record in records]
    completed_order = [int(row["image_id"]) for row in completed]
    if completed_order != manifest_ids[: len(completed_order)]:
        raise ValueError("Worker evaluation resume rows are not the manifest prefix")
    for row, record in zip(completed, records):
        current_source_sha256 = sha256_file(
            _safe_manifest_image(dataset_root, str(record["image_path"]))
        )
        if row.get("source_image_sha256") != current_source_sha256:
            raise ValueError("Worker evaluation resume source image checksum changed")
        expected_source_sha256 = record.get("source_image_sha256")
        if expected_source_sha256 is not None and current_source_sha256 != expected_source_sha256:
            raise ValueError("Worker evaluation resume row checksum does not match manifest")

    detector = _CapturingDetector(detector_adapter)
    classifier = _CapturingClassifier(classifier)
    pipeline = DecisionPipeline(
        detector,
        classifier,
        package.metadata.classifier,
        package.metadata.quality,
        package.metadata.count_verifier,
    )
    with rows_path.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            image_id = int(record["image_id"])
            if image_id in completed_ids:
                continue
            encoded = _safe_manifest_image(dataset_root, str(record["image_path"])).read_bytes()
            source_image_sha256 = _sha256_bytes(encoded)
            expected_image_sha256 = record.get("source_image_sha256")
            if expected_image_sha256 is not None and source_image_sha256 != expected_image_sha256:
                raise ValueError(f"benchmark source checksum mismatch: image {image_id}")
            image = decode_image(
                encoded,
                max_bytes=50_000_000,
                max_pixels=50_000_000,
                jpeg_draft_size=package.metadata.input.jpeg_draft_size,
            )
            classifier_calls_before = classifier.call_count
            response = pipeline.scan(image, request_id=f"rpc-worker-{image_id:08d}")
            classifier_call_delta = classifier.call_count - classifier_calls_before
            if detector.last_result is None:
                raise RuntimeError("DecisionPipeline did not execute the detector")
            detector_boxes = [
                np.asarray(
                    [detection.x1, detection.y1, detection.x2, detection.y2],
                    dtype=np.float64,
                )
                for detection in detector.last_result.detections
            ]
            items = response.items
            item_boxes = [
                _xywh_to_xyxy([item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height])
                for item in items
            ]
            annotations = list(record["annotations"])
            detector_matched = _match_items(detector_boxes, annotations, match_iou_threshold)
            item_matched = _match_items(item_boxes, annotations, match_iou_threshold)
            approved_count = 0
            approved_correct = 0
            unknown_count = 0
            unknown_matched = 0
            unknown_unmatched = 0
            unknown_top1_correct = 0
            unknown_top3_correct = 0
            for item_index, item in enumerate(items):
                target_index = item_matched.get(item_index)
                target_class_id = (
                    str(annotations[target_index]["category_id"])
                    if target_index is not None
                    else None
                )
                if item.status.value == "APPROVED":
                    approved_count += 1
                    approved_correct += int(
                        target_class_id is not None
                        and item.prediction is not None
                        and item.prediction.class_id == target_class_id
                    )
                else:
                    unknown_count += 1
                    if target_class_id is None:
                        unknown_unmatched += 1
                    else:
                        unknown_matched += 1
                        unknown_top1_correct += int(
                            bool(item.top3) and item.top3[0].class_id == target_class_id
                        )
                        unknown_top3_correct += int(
                            target_class_id in {candidate.class_id for candidate in item.top3}
                        )
            expected_hard_gate_reasons = _expected_hard_gate_reasons(
                image, detector.last_result, package.metadata
            )
            expected_hard_gate = bool(expected_hard_gate_reasons)
            classifier_executed = classifier_call_delta > 0
            classifier_version_reported = response.model_versions.classifier
            execution_version_mismatch = classifier_executed != (
                classifier_version_reported is not None
            )
            classifier_called_violation = expected_hard_gate and classifier_call_delta != 0
            classifier_call_count_violation = classifier_call_delta != (
                0 if expected_hard_gate else 1
            )
            response_status_violation = expected_hard_gate and response.status.value != "RECAPTURE"
            classifier_version_violation = (
                expected_hard_gate and classifier_version_reported is not None
            )
            reason_mismatch_violation = (
                expected_hard_gate and response.reason_codes != expected_hard_gate_reasons
            )
            row = {
                "schema_version": SCHEMA_VERSION,
                "image_id": image_id,
                "level": str(record.get("level", "unknown")),
                "source_image_sha256": source_image_sha256,
                "status": response.status.value,
                "reason_codes": response.reason_codes,
                "classifier_executed": classifier_executed,
                "classifier_version_reported": classifier_version_reported,
                "classifier_execution_version_mismatch": execution_version_mismatch,
                "processing_time_ms": float(response.processing_time_ms),
                "ground_truth_count": len(annotations),
                "detector_prediction_count": len(detector_boxes),
                "detector_matched_count": len(detector_matched),
                "detector_count_correct": len(detector_boxes) == len(annotations),
                "classifier_item_count": len(items),
                "classifier_item_matched_count": len(item_matched),
                "approved_count": approved_count,
                "approved_correct_count": approved_correct,
                "unknown_count": unknown_count,
                "unknown_matched_count": unknown_matched,
                "unknown_unmatched_count": unknown_unmatched,
                "unknown_top1_correct_count": unknown_top1_correct,
                "unknown_top3_correct_count": unknown_top3_correct,
                "expected_hard_gate": expected_hard_gate,
                "expected_hard_gate_reasons": expected_hard_gate_reasons,
                "classifier_call_count_delta": classifier_call_delta,
                "classifier_call_count_violation": classifier_call_count_violation,
                "hard_gate_classifier_called_violation": classifier_called_violation,
                "hard_gate_response_status_violation": response_status_violation,
                "hard_gate_classifier_version_violation": classifier_version_violation,
                "hard_gate_reason_mismatch_violation": reason_mismatch_violation,
                "hard_gate_contract_violation": any(
                    (
                        classifier_called_violation,
                        response_status_violation,
                        classifier_version_violation,
                        reason_mismatch_violation,
                    )
                ),
                "pipeline_contract_violation": execution_version_mismatch
                or classifier_call_count_violation
                or any(
                    (
                        classifier_called_violation,
                        response_status_violation,
                        classifier_version_violation,
                        reason_mismatch_violation,
                    )
                ),
            }
            row_line = _canonical_bytes(row) + b"\n"
            stream.write(row_line.decode("utf-8"))
            stream.flush()
            completed.append(row)
            rows_chain_sha256 = _next_rows_chain(rows_chain_sha256, row_line)
            _write_json(
                state_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "input_fingerprint": fingerprint,
                    "requested_provider": provider,
                    "selected_provider": selected_provider,
                    "completed_count": len(completed),
                    "rows_sha256": rows_chain_sha256,
                    "rows_checksum_scheme": "sha256-chain-v1",
                },
            )

    aggregation = aggregate_worker_rows(completed)
    manifest_provenance = _worker_manifest_provenance(
        manifest_path, records, completed, rows_path, state_path
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "worker-eval",
        "package_version": package.metadata.package_version,
        "dataset_version": package.metadata.dataset_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
        "package_artifact_sha256": package_inputs,
        "requested_provider": provider,
        "provider": selected_provider,
        "match_iou_threshold": match_iou_threshold,
        "input_fingerprint": fingerprint,
        "rows_sha256": sha256_file(rows_path),
        "test_manifest": manifest_provenance,
        **aggregation,
    }
    _write_json(output_path, report)
    return report


def _artifact_path(path: Path | None, default: Path) -> Path:
    return (path or default).resolve()


def _full_path_latency(benchmark: dict[str, Any]) -> dict[str, Any]:
    full_path = benchmark.get("by_path", {}).get("full_path")
    if not isinstance(full_path, dict):
        raise ValueError("benchmark report has no full_path latency summary")
    return {
        "count": int(full_path.get("sample_count", full_path.get("count", 0))),
        "p50_ms": float(full_path.get("p50_ms", full_path.get("p50"))),
        "p95_ms": float(full_path.get("p95_ms", full_path.get("p95"))),
        "p99_ms": float(full_path.get("p99_ms", full_path.get("p99"))),
    }


def integrate(
    output_root: Path,
    package_dir: Path,
    *,
    config_path: Path,
    pytorch_report_path: Path | None = None,
    model_lock_path: Path | None = None,
    package_inputs_dir: Path | None = None,
    parity_report_paths: list[Path] | None = None,
    worker_report_path: Path | None = None,
    benchmark_report_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Bind training, package, parity, ORT accuracy, and latency evidence."""
    output_root = output_root.resolve()
    package_dir = package_dir.resolve()
    config_path = config_path.resolve()
    pytorch_report_path = _artifact_path(
        pytorch_report_path, output_root / "reports" / "final_test.json"
    )
    model_lock_path = _artifact_path(model_lock_path, output_root / "model_lock.json")
    package_inputs_dir = _artifact_path(package_inputs_dir, output_root / "package-inputs")
    worker_report_path = _artifact_path(
        worker_report_path, output_root / "reports" / "worker-ort-accuracy.json"
    )
    benchmark_report_path = _artifact_path(
        benchmark_report_path, output_root / "reports" / "benchmark.json"
    )
    output_path = _artifact_path(
        output_path, output_root / "reports" / "integrated-worker-kpi.json"
    )
    if parity_report_paths is None:
        parity_report_paths = sorted((output_root / "reports").glob("parity*.json"))
    parity_report_paths = [path.resolve() for path in parity_report_paths]
    if not parity_report_paths:
        raise FileNotFoundError("at least one PyTorch/ONNX parity report is required")

    evidence_paths = {
        "rpc_config": config_path,
        "pytorch_final_test": pytorch_report_path,
        "detector_final_test": output_root / "test" / "detector_report.json",
        "model_lock": model_lock_path,
        "package_metadata": package_dir / "metadata.json",
        "package_input_checksums": package_inputs_dir / "checksums.json",
        "package_detector_evaluation": package_inputs_dir / "detector-evaluation.json",
        "package_classifier_calibration": package_inputs_dir / "classifier-calibration.json",
        "package_manifest_metadata": package_inputs_dir / "manifest-metadata.json",
        "package_export_config": package_inputs_dir / "export-config.json",
        "worker_ort_accuracy": worker_report_path,
        "benchmark": benchmark_report_path,
        "benchmark_manifest": output_root / "benchmark" / "manifest.json",
        "benchmark_manifest_checksums": output_root / "benchmark" / "checksums.json",
        **{f"parity_{index + 1}": path for index, path in enumerate(parity_report_paths)},
    }
    artifact_hashes = _logical_hashes(evidence_paths)
    package = load_model_package(package_dir)
    expected_package_hashes = _package_artifact_hashes(package_dir, package)
    pytorch_report = _read_json(pytorch_report_path)
    detector_test_report = _read_json(output_root / "test" / "detector_report.json")
    test_annotation_sha256 = detector_test_report.get("test_annotation_sha256")
    if (
        not isinstance(test_annotation_sha256, str)
        or len(test_annotation_sha256) != 64
        or any(character not in "0123456789abcdef" for character in test_annotation_sha256)
    ):
        raise ValueError("final detector report has no valid test annotation checksum")
    model_lock = _read_json(model_lock_path)
    model_run = model_lock.get("model_run")
    if not isinstance(model_run, str) or not model_run:
        raise ValueError("model lock has no model_run")
    locked_run_dir = (output_root / model_run).resolve()
    try:
        locked_run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("model lock run escapes output root") from exc
    for field, filename in (
        ("checkpoint_sha256", "best.pt"),
        ("calibration_sha256", "calibration.json"),
        ("selection_report_sha256", "selection_report.json"),
    ):
        locked_path = locked_run_dir / filename
        if not locked_path.is_file() or sha256_file(locked_path) != model_lock.get(field):
            raise ValueError(f"model lock checksum mismatch: {filename}")
    _config, _experiment, validated_lock, validated_run_dir, bridge_input_paths = (
        _validate_full_dataset_inputs(output_root, config_path)
    )
    if validated_lock != model_lock or validated_run_dir != locked_run_dir:
        raise ValueError("operational inputs differ from the selected model lock")
    detector_bindings = {
        "detector_checkpoint_sha256": model_lock.get("final_detector_checkpoint_sha256"),
        "final_detector_complete_sha256": model_lock.get("final_detector_complete_sha256"),
        "model_lock_sha256": sha256_file(model_lock_path),
        "active_detector_threshold_sha256": model_lock.get("active_detector_threshold_sha256"),
        "train_gate_complete_sha256": model_lock.get("detector_train_gate_complete_sha256"),
        "operational_detector_role": "checkout_baseline_val_all_operational",
        "train_gate_role": "offline_roi_train_gate_only",
    }
    for field, expected in detector_bindings.items():
        if detector_test_report.get(field) != expected:
            raise ValueError(f"final detector report evidence mismatch: {field}")
    package_bridge = _read_json(package_inputs_dir / "checksums.json")
    if package_bridge.get("phase") != "package-inputs":
        raise ValueError("package input checksum ledger is invalid")
    current_bridge_inputs = _logical_hashes(bridge_input_paths)
    if package_bridge.get("inputs") != current_bridge_inputs:
        raise ValueError("package input ledger no longer matches its source artifacts")
    artifact_hashes.update(
        {f"package_source_{name}": digest for name, digest in current_bridge_inputs.items()}
    )
    pytorch_bindings = {
        "classifier_checkpoint_sha256": model_lock.get("checkpoint_sha256"),
        "model_lock_sha256": sha256_file(model_lock_path),
        "detector_report_sha256": sha256_file(output_root / "test" / "detector_report.json"),
        "detector_checkpoint_sha256": detector_test_report.get("detector_checkpoint_sha256"),
    }
    for field, expected in pytorch_bindings.items():
        if pytorch_report.get(field) != expected:
            raise ValueError(f"PyTorch final test evidence mismatch: {field}")
    bridge_outputs = package_bridge.get("outputs")
    if not isinstance(bridge_outputs, dict):
        raise ValueError("package input checksum ledger has no outputs")
    for relative, expected in bridge_outputs.items():
        bridged_path = package_inputs_dir / str(relative)
        if not bridged_path.is_file() or sha256_file(bridged_path) != expected:
            raise ValueError(f"package input checksum mismatch: {relative}")
    manifest_metadata = _read_json(package_inputs_dir / "manifest-metadata.json")
    detector_evaluation = _read_json(package_inputs_dir / "detector-evaluation.json")
    classifier_calibration = _read_json(package_inputs_dir / "classifier-calibration.json")
    export_bridge = _read_json(package_inputs_dir / "export-config.json")
    _validate_package_bridge_metadata(package, detector_evaluation, classifier_calibration)
    if export_bridge.get("source_config_sha256") != sha256_file(config_path):
        raise ValueError("export bridge is not bound to rpc_data_scale config")
    _validate_export_worker_policy(package, export_bridge)
    if manifest_metadata.get("dataset_version") != package.metadata.dataset_version:
        raise ValueError("package dataset version does not match RPC manifest metadata")
    expected_labels = [
        (str(row["class_id"]), str(row["class_name"])) for row in manifest_metadata["labels"]
    ]
    actual_labels = [
        (label.class_id, label.class_name) for label in package.metadata.classifier.labels
    ]
    if expected_labels != actual_labels:
        raise ValueError("package classifier labels do not preserve RPC logit order")
    if pytorch_report.get("model_run") != model_run:
        raise ValueError("PyTorch final test is not the locked model run")
    parity_reports = [_read_json(path) for path in parity_report_paths]
    worker_report = _read_json(worker_report_path)
    benchmark = _read_json(benchmark_report_path)
    _validate_package_evidence(
        worker_report, "Worker accuracy report", package, expected_package_hashes
    )
    worker_manifest_provenance = _validate_worker_test_manifest_evidence(
        worker_report_path,
        worker_report,
        detector_test_report,
        test_annotation_sha256,
    )
    artifact_hashes["worker_ort_accuracy_rows"] = str(
        worker_manifest_provenance["rows_file_sha256"]
    )
    artifact_hashes["worker_ort_accuracy_state"] = str(
        worker_manifest_provenance["state_file_sha256"]
    )
    _validate_package_evidence(benchmark, "benchmark report", package, expected_package_hashes)
    _validate_benchmark_manifest_evidence(
        output_root / "benchmark",
        benchmark,
        sha256_file(output_root / "test" / "detector_report.json"),
        test_annotation_sha256,
    )
    for index, parity_report in enumerate(parity_reports, start=1):
        _validate_package_evidence(
            parity_report,
            f"parity report {index}",
            package,
            expected_package_hashes,
        )
        if parity_report.get("classifier_checkpoint_sha256") != model_lock.get("checkpoint_sha256"):
            raise ValueError(f"parity report {index} classifier checkpoint is not the locked model")
        if parity_report.get("detector_checkpoint_sha256") != detector_test_report.get(
            "detector_checkpoint_sha256"
        ):
            raise ValueError(
                f"parity report {index} detector checkpoint is not the final test model"
            )
    latency = _full_path_latency(benchmark)
    benchmark_sample_count = int(benchmark.get("sample_count", 0))
    warmup_count = int(benchmark.get("warmup_count", 0))
    benchmark_gpu = benchmark.get("gpu")
    benchmark_system = benchmark.get("system")
    gpu_name = str(benchmark_gpu.get("name", "")) if isinstance(benchmark_gpu, dict) else ""
    detector = worker_report.get("detector", {})
    classifier = worker_report.get("classifier", {})
    unknown_top3 = classifier.get("unknown_top3_accuracy")
    approved_count = int(classifier.get("approved_count", 0))
    approved_precision = classifier.get("approved_precision")
    unknown_matched_count = int(classifier.get("unknown_matched_count", 0))
    parity_passed = all(report.get("passes") is True for report in parity_reports)
    parity_providers = {str(report.get("provider")) for report in parity_reports}

    def parity_policy_exact(report: dict[str, Any]) -> bool:
        detector_policy = report.get("detector", {})
        classifier_policy = report.get("classifier", {})
        return (
            detector_policy.get("minimum_iou_tolerance") == 0.99
            and detector_policy.get("coordinate_tolerance") == 0.01
            and detector_policy.get("score_tolerance") == 0.02
            and classifier_policy.get("tolerance") == 0.01
        )

    parity_tolerance_policy_exact = all(parity_policy_exact(report) for report in parity_reports)
    pipeline_contract_violations = int(
        worker_report.get("pipeline_contract", {}).get("pipeline_contract_violations", 0)
    )
    gpu_memory_mib = (
        float(benchmark_gpu.get("memory_total_mib", 0.0))
        if isinstance(benchmark_gpu, dict)
        else 0.0
    )
    windows_build = (
        float(benchmark_system.get("windows_build", 0.0))
        if isinstance(benchmark_system, dict) and benchmark_system.get("windows_build") is not None
        else 0.0
    )
    cpu_name = str(benchmark_system.get("cpu", "")) if isinstance(benchmark_system, dict) else ""
    memory_total_gib = (
        float(benchmark_system.get("memory_total_gib", 0.0))
        if isinstance(benchmark_system, dict)
        and benchmark_system.get("memory_total_gib") is not None
        else 0.0
    )
    gpu_uuid = str(benchmark_gpu.get("uuid", "")) if isinstance(benchmark_gpu, dict) else ""
    gpu_physical_index = (
        int(benchmark_gpu.get("physical_index", -1)) if isinstance(benchmark_gpu, dict) else -1
    )
    ort_cuda_device_id = (
        int(benchmark_gpu.get("ort_cuda_device_id", -1)) if isinstance(benchmark_gpu, dict) else -1
    )
    gpu_selection_source = (
        str(benchmark_gpu.get("selection_source", "")) if isinstance(benchmark_gpu, dict) else ""
    )
    physical_gpu_count = (
        int(benchmark_gpu.get("physical_gpu_count", 0)) if isinstance(benchmark_gpu, dict) else 0
    )
    cuda_visible_devices = (
        benchmark_gpu.get("cuda_visible_devices") if isinstance(benchmark_gpu, dict) else None
    )
    cuda_version = (
        str(benchmark_gpu.get("cuda_version", "")) if isinstance(benchmark_gpu, dict) else ""
    )
    onnxruntime_build_info = str(benchmark.get("onnxruntime_build_info", ""))
    gates = {
        "pytorch_final_test_certified": pytorch_report.get("status") == "test_certified",
        "parity_passed": parity_passed,
        "parity_cpu_and_cuda_present": {"cpu", "cuda"} <= parity_providers,
        "parity_tolerance_policy_exact": parity_tolerance_policy_exact,
        "detector_bbox_recall_at_least_0_99": float(detector.get("recall", 0.0)) >= 0.99,
        "approved_truth_sample_present": approved_count > 0,
        "approved_precision_at_least_0_995": approved_precision is not None
        and float(approved_precision) >= 0.995,
        "unknown_truth_sample_present": unknown_matched_count > 0,
        "unknown_top3_at_least_0_95": unknown_top3 is not None and float(unknown_top3) >= 0.95,
        "worker_match_iou_threshold_is_0_5": worker_report.get("match_iou_threshold") == 0.5,
        "pipeline_contract_satisfied": pipeline_contract_violations == 0,
        "worker_ort_cuda_provider": worker_report.get("provider") == "cuda",
        "benchmark_cuda_provider": benchmark.get("provider") == "cuda",
        "benchmark_gpu_is_desktop_rtx_5080": gpu_name.casefold() == "nvidia geforce rtx 5080",
        "benchmark_gpu_vram_at_least_15000_mib": gpu_memory_mib >= 15_000,
        "benchmark_gpu_device_binding_evidenced": bool(gpu_uuid)
        and gpu_physical_index >= 0
        and ort_cuda_device_id == 0
        and gpu_selection_source in {"single_physical_gpu", "CUDA_VISIBLE_DEVICES_UUID"},
        "benchmark_gpu_selection_is_unambiguous": (
            gpu_selection_source == "single_physical_gpu" and physical_gpu_count == 1
        )
        or (
            gpu_selection_source == "CUDA_VISIBLE_DEVICES_UUID"
            and physical_gpu_count >= 1
            and cuda_visible_devices is not None
            and bool(str(cuda_visible_devices).strip())
        ),
        "benchmark_cuda_version_present": bool(cuda_version.strip()),
        "benchmark_onnxruntime_build_info_present": bool(onnxruntime_build_info.strip()),
        "benchmark_windows_11_build": windows_build >= 22_000,
        "benchmark_cpu_is_285k": "285k" in cpu_name.casefold(),
        "benchmark_memory_at_least_63_gib": memory_total_gib >= 63.0,
        "benchmark_run_count_at_least_1000": benchmark_sample_count >= 1000,
        "benchmark_warmup_count_at_least_30": warmup_count >= 30,
        "all_benchmark_samples_are_full_path": latency["count"] == benchmark_sample_count
        and benchmark_sample_count > 0,
        "full_path_p95_at_most_100_ms": latency["p95_ms"] <= 100.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "integrate",
        "package_version": package.metadata.package_version,
        "dataset_version": package.metadata.dataset_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
        "artifact_sha256": artifact_hashes,
        "dataset_evidence": {
            "test_annotation_sha256": test_annotation_sha256,
            "worker_manifest_sha256": worker_manifest_provenance["manifest_sha256"],
            "worker_test_source_image_set_sha256": worker_manifest_provenance[
                "source_image_set_sha256"
            ],
            "worker_test_row_count": worker_manifest_provenance["evaluated_row_count"],
        },
        "parity_evidence": {
            "providers": sorted(parity_providers),
            "required_tolerances": {
                "detector_minimum_iou": 0.99,
                "detector_coordinate_max_abs_error": 0.01,
                "detector_score_max_abs_error": 0.02,
                "classifier_logits_max_abs_error": 0.01,
            },
        },
        "environment": {
            "worker_provider": worker_report.get("provider"),
            "benchmark_provider": benchmark.get("provider"),
            "onnxruntime_version": benchmark.get("onnxruntime_version"),
            "onnxruntime_build_info": onnxruntime_build_info or None,
            "platform": benchmark.get("platform"),
            "python_version": benchmark.get("python_version"),
            "gpu_name": gpu_name or None,
            "gpu_memory_total_mib": gpu_memory_mib,
            "cuda_version": cuda_version or None,
            "gpu_uuid": gpu_uuid or None,
            "gpu_physical_index": gpu_physical_index,
            "physical_gpu_count": physical_gpu_count,
            "ort_cuda_device_id": ort_cuda_device_id,
            "gpu_selection_source": gpu_selection_source or None,
            "cuda_visible_devices": cuda_visible_devices,
            "cuda_device_order": (
                benchmark_gpu.get("cuda_device_order") if isinstance(benchmark_gpu, dict) else None
            ),
            "driver_version": (
                benchmark_gpu.get("driver_version") if isinstance(benchmark_gpu, dict) else None
            ),
            "warmup_count": warmup_count,
            "run_count": benchmark_sample_count,
            "request_concurrency": 1,
            "windows_build": windows_build,
            "cpu": cpu_name or None,
            "memory_total_gib": memory_total_gib,
        },
        "metrics": {
            "pytorch_test_top1": pytorch_report.get("top1"),
            "pytorch_test_top3": pytorch_report.get("top3"),
            "detector_bbox_recall": detector.get("recall"),
            "detector_bbox_precision": detector.get("precision"),
            "detector_count_accuracy": detector.get("count_accuracy"),
            "approved_count": approved_count,
            "approved_precision": approved_precision,
            "approval_coverage": classifier.get("approval_coverage"),
            "worker_normal_matched_top1_accuracy": classifier.get("normal_matched_top1_accuracy"),
            "unknown_top3_accuracy": unknown_top3,
            "unknown_matched_count": unknown_matched_count,
            "recapture_recall": "not_evaluable",
            "recapture_recall_reason": "RPC test data has no capture-quality ground truth",
            "full_path_latency": latency,
        },
        "gates": gates,
        "all_evaluable_gates_satisfied": all(gates.values()),
        "certification": {
            "status": "not_certified",
            "production_certified": False,
            "non_certified_requirements": ["recapture_recall"],
            "recapture_recall": {
                "status": "not_evaluable",
                "certified": False,
                "reason": "RPC test data has no capture-quality ground truth",
            },
        },
    }
    _write_json(output_path, report)
    return report


def operation_plan(
    output_root: Path,
    package_dir: Path,
    *,
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Write a machine-readable DAG and identify the currently runnable stages."""
    output_root = output_root.resolve()
    package_dir = package_dir.resolve()
    config_path = config_path.resolve()
    output_path = _artifact_path(output_path, output_root / "reports" / "rpc-operation-plan.json")
    stage_specs = [
        (
            "final_test",
            [],
            [output_root / "model_lock.json"],
            [
                output_root / "test" / "detector_report.json",
                output_root / "reports" / "final_test.json",
            ],
        ),
        (
            "package_inputs",
            ["final_test"],
            [],
            [
                output_root / "package-inputs" / "checksums.json",
                output_root / "package-inputs" / "export-config.json",
            ],
        ),
        (
            "detector_classifier_package",
            ["package_inputs"],
            [],
            [
                package_dir / "detector.onnx",
                package_dir / "classifier.onnx",
                package_dir / "metadata.json",
            ],
        ),
        (
            "parity_cpu",
            ["detector_classifier_package"],
            [],
            [output_root / "reports" / "parity-cpu.json"],
        ),
        (
            "parity_cuda",
            ["detector_classifier_package"],
            [],
            [output_root / "reports" / "parity-cuda.json"],
        ),
        (
            "worker_accuracy_full_test",
            ["detector_classifier_package"],
            [output_root / "test" / "detector_report.json"],
            [
                output_root / "reports" / "worker-ort-accuracy.json",
                output_root / "reports" / "worker-ort-accuracy.jsonl",
                output_root / "reports" / "worker-ort-accuracy.jsonl.state.json",
            ],
        ),
        (
            "benchmark_manifest",
            ["final_test"],
            [],
            [
                output_root / "benchmark" / "manifest.json",
                output_root / "benchmark" / "checksums.json",
            ],
        ),
        (
            "benchmark_cuda",
            ["detector_classifier_package", "benchmark_manifest"],
            [],
            [output_root / "reports" / "benchmark.json"],
        ),
        (
            "integrate",
            [
                "parity_cpu",
                "parity_cuda",
                "worker_accuracy_full_test",
                "benchmark_cuda",
            ],
            [],
            [output_root / "reports" / "integrated-worker-kpi.json"],
        ),
    ]
    model_lock_path = output_root / "model_lock.json"
    test_authorized = False
    test_authorization_failures: list[str] = []
    if model_lock_path.is_file():
        try:
            model_lock = _read_json(model_lock_path)
        except (OSError, ValueError, json.JSONDecodeError):
            test_authorization_failures.append("MODEL_LOCK_INVALID")
        else:
            if model_lock.get("status") != "validation_passed":
                test_authorization_failures.append("VALIDATION_GATE_NOT_PASSED")
            if model_lock.get("operational_gate") is not True:
                test_authorization_failures.append("OPERATIONAL_GATE_NOT_PASSED")
            test_authorized = not test_authorization_failures
    artifacts_complete = {
        stage: all(path.is_file() for path in artifacts)
        for stage, _dependencies, _required_inputs, artifacts in stage_specs
    }
    # A failed validation lock must never become runnable merely because the
    # marker file exists (or because stale test artifacts happen to exist).
    artifacts_complete["final_test"] = artifacts_complete["final_test"] and test_authorized
    completed: dict[str, bool] = {}
    for stage, dependencies, _required_inputs, _artifacts in stage_specs:
        completed[stage] = artifacts_complete[stage] and all(
            completed[dependency] for dependency in dependencies
        )
    stages = []
    for stage, dependencies, required_inputs, artifacts in stage_specs:
        inputs_available = all(path.is_file() for path in required_inputs)
        authorization_failures = test_authorization_failures if stage == "final_test" else []
        if authorization_failures:
            status = "blocked"
        elif completed[stage]:
            status = "completed"
        elif inputs_available and all(completed[dependency] for dependency in dependencies):
            status = "ready"
        else:
            status = "blocked"
        stages.append(
            {
                "id": stage,
                "dependencies": dependencies,
                "status": status,
                "required_inputs": [str(path) for path in required_inputs],
                "missing_required_inputs": [
                    str(path) for path in required_inputs if not path.is_file()
                ],
                "artifacts": [str(path) for path in artifacts],
                "missing_artifacts": [str(path) for path in artifacts if not path.is_file()],
                "authorization_failures": authorization_failures,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "operation-plan",
        "dag_contract": RPC_OPERATION_DAG_VERSION,
        "config_sha256": sha256_file(config_path),
        "output_root": str(output_root),
        "package_dir": str(package_dir),
        "stages": stages,
        "next_steps": [stage["id"] for stage in stages if stage["status"] == "ready"],
        "certification_constraint": {
            "recapture_recall": "not_evaluable",
            "production_certification": "not_certified",
        },
    }
    _write_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and integrate RPC operational Worker evidence"
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    package_parser = subparsers.add_parser("package-inputs")
    package_parser.add_argument("--output-root", type=Path, required=True)
    package_parser.add_argument("--config", type=Path, required=True)
    package_parser.add_argument("--destination", type=Path)
    package_parser.add_argument("--resume", action="store_true")

    benchmark_parser = subparsers.add_parser("benchmark-manifest")
    benchmark_parser.add_argument("--output-root", type=Path, required=True)
    benchmark_parser.add_argument("--dataset-root", type=Path, required=True)
    benchmark_parser.add_argument("--detector-report", type=Path)
    benchmark_parser.add_argument("--annotation", type=Path)
    benchmark_parser.add_argument("--destination", type=Path)
    benchmark_parser.add_argument("--max-images", type=int, default=1000)
    benchmark_parser.add_argument("--salt", default=DEFAULT_SELECTION_SALT)
    benchmark_parser.add_argument("--resume", action="store_true")

    worker_parser = subparsers.add_parser("worker-eval")
    worker_parser.add_argument("--package-dir", type=Path, required=True)
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--dataset-root", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--rows", type=Path)
    worker_parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="cuda")
    worker_parser.add_argument("--cuda-dll-dir", type=Path)
    worker_parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    worker_parser.add_argument("--resume", action="store_true")

    integrate_parser = subparsers.add_parser("integrate")
    integrate_parser.add_argument("--output-root", type=Path, required=True)
    integrate_parser.add_argument("--package-dir", type=Path, required=True)
    integrate_parser.add_argument("--config", type=Path, required=True)
    integrate_parser.add_argument("--pytorch-report", type=Path)
    integrate_parser.add_argument("--model-lock", type=Path)
    integrate_parser.add_argument("--package-inputs-dir", type=Path)
    integrate_parser.add_argument("--parity-report", type=Path, action="append")
    integrate_parser.add_argument("--worker-report", type=Path)
    integrate_parser.add_argument("--benchmark-report", type=Path)
    integrate_parser.add_argument("--output", type=Path)

    plan_parser = subparsers.add_parser("next-steps")
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument("--package-dir", type=Path, required=True)
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.phase == "package-inputs":
        report = package_inputs(
            args.output_root,
            args.config,
            destination=args.destination,
            resume=args.resume,
        )
    elif args.phase == "benchmark-manifest":
        report = build_benchmark_manifest(
            args.output_root,
            args.dataset_root,
            detector_report_path=args.detector_report,
            annotation_path=args.annotation,
            destination=args.destination,
            max_images=args.max_images,
            salt=args.salt,
            resume=args.resume,
        )
    elif args.phase == "worker-eval":
        report = worker_eval(
            args.package_dir,
            args.manifest,
            args.dataset_root,
            args.output,
            rows_path=args.rows,
            provider=args.provider,
            cuda_dll_dir=args.cuda_dll_dir,
            match_iou_threshold=args.match_iou_threshold,
            resume=args.resume,
        )
    elif args.phase == "integrate":
        report = integrate(
            args.output_root,
            args.package_dir,
            config_path=args.config,
            pytorch_report_path=args.pytorch_report,
            model_lock_path=args.model_lock,
            package_inputs_dir=args.package_inputs_dir,
            parity_report_paths=args.parity_report,
            worker_report_path=args.worker_report,
            benchmark_report_path=args.benchmark_report,
            output_path=args.output,
        )
    else:
        report = operation_plan(
            args.output_root,
            args.package_dir,
            config_path=args.config,
            output_path=args.output,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.phase == "integrate" and not report["all_evaluable_gates_satisfied"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
