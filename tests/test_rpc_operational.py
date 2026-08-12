from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bixolon_scanner.inference import Detection, DetectionResult
from bixolon_scanner.package import sha256_file
from bixolon_scanner.package import ClassifierMetadata, QualityMetadata
from bixolon_scanner.training.rpc_operational import (
    _image_ids_sha256,
    _rows_chain,
    _source_set_sha256,
    _validate_benchmark_manifest_evidence,
    _expected_hard_gate_reasons,
    _match_items,
    aggregate_worker_rows,
    build_benchmark_manifest,
    integrate,
    operation_plan,
    package_inputs,
    worker_eval,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _categories() -> list[dict[str, object]]:
    return [{"id": index, "name": f"class-{index:03d}"} for index in range(1, 201)]


def _locked_training_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "artifacts"
    config_path = tmp_path / "rpc.json"
    _write_json(
        config_path,
        {
            "experiment": {"mode": "full_dataset", "expected_num_classes": 200},
            "detector": {
                "image_size": 640,
                "nms_iou_threshold": 0.7,
                "max_queries": 300,
                "uncertainty_score_threshold": 0.2,
                "uncertainty_min_area_ratio": 0.039,
                "uncertainty_match_iou_threshold": 0.5,
                "min_object_area_ratio": 0.005,
                "border_margin_ratio": 0.002,
                "border_policy": "classifier_confidence",
            },
            "training": {"image_size": 224, "eval_margin_ratio": 0.05},
        },
    )
    _write_json(
        root / "detector" / "threshold.json",
        {
            "threshold_policy": "calibration_oof_only",
            "selected_score_threshold": 0.42,
            "target_recall": 0.99,
            "target_recall_satisfied": True,
            "calibration_metrics": {
                "recall": 0.995,
                "precision": 0.98,
                "count_accuracy": 0.97,
            },
        },
    )
    _write_json(
        root / "prepared" / "experiment.json",
        {
            "mode": "full_dataset",
            "category_count": 200,
            "categories": list(reversed(_categories())),
            "source_hashes": {"instances_train2019.json": "a" * 64},
        },
    )
    run_dir = root / "runs" / "full" / "seed20260810"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    _write_json(
        run_dir / "calibration.json",
        {
            "temperature": 1.1,
            "approval_threshold": 0.91,
            "approved_precision": 0.999,
            "approval_coverage": 0.8,
            "approved_false_rate_upper_95": 0.004,
            "risk_control_satisfied": True,
            "matched_count": 90,
            "unmatched_detector_count": 10,
        },
    )
    _write_json(run_dir / "selection_report.json", {"ok": True})
    _write_json(
        root / "model_lock.json",
        {
            "mode": "full_dataset",
            "model_run": "runs/full/seed20260810",
            "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            "calibration_sha256": sha256_file(run_dir / "calibration.json"),
            "selection_report_sha256": sha256_file(run_dir / "selection_report.json"),
        },
    )
    return root, config_path


def test_package_inputs_preserves_rpc_logit_order_and_checksums(tmp_path: Path):
    root, config = _locked_training_root(tmp_path)
    first = package_inputs(root, config)
    destination = root / "package-inputs"
    detector = json.loads((destination / "detector-evaluation.json").read_text())
    calibration = json.loads((destination / "classifier-calibration.json").read_text())
    metadata = json.loads((destination / "manifest-metadata.json").read_text())
    export_config = json.loads((destination / "export-config.json").read_text())

    assert detector["metrics"]["recall"] == 0.995
    assert detector["nms_iou_threshold"] == 0.7
    assert calibration["sample_count"] == 100
    assert [row["category_id"] for row in metadata["labels"]] == list(range(1, 201))
    assert [row["logit_index"] for row in metadata["labels"]] == list(range(200))
    assert metadata["labels"][0]["class_id"] == "1"
    assert len(metadata["dataset_identity_sha256"]) == 64
    assert export_config["export"]["uncertainty_min_area_ratio"] == 0.039
    assert export_config["export"]["crop_margin"] == 0.05
    assert export_config["expected_package_metadata"]["detector"]["max_queries"] == 300
    assert first == package_inputs(root, config, resume=True)

    config.write_text(config.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input checksums"):
        package_inputs(root, config, resume=True)


def test_benchmark_manifest_is_deterministic_and_only_uses_full_path_frames(
    tmp_path: Path,
):
    root = tmp_path / "artifacts"
    dataset = tmp_path / "rpc"
    (dataset / "test2019").mkdir(parents=True)
    images = []
    annotations = []
    outcomes = []
    for image_id in range(1, 6):
        filename = f"frame-{image_id}.jpg"
        (dataset / "test2019" / filename).write_bytes(f"image-{image_id}".encode())
        images.append(
            {"id": image_id, "file_name": filename, "width": 20, "height": 10, "level": "easy"}
        )
        annotations.append(
            {"id": image_id, "image_id": image_id, "category_id": image_id, "bbox": [1, 2, 3, 4]}
        )
        outcomes.append(
            {
                "image_id": image_id,
                "level": "easy",
                "ground_truth_count": 1,
                "recapture_reasons": [] if image_id != 4 else ["DETECTOR_NO_OBJECT"],
            }
        )
    _write_json(
        dataset / "instances_test2019.json",
        {"images": images, "annotations": annotations, "categories": _categories()},
    )
    _write_json(root / "test" / "detector_report.json", {"outcomes": outcomes})

    first = build_benchmark_manifest(root, dataset, max_images=3)
    second = build_benchmark_manifest(
        root, dataset, destination=root / "benchmark-again", max_images=3
    )
    assert [row["image_id"] for row in first["records"]] == [
        row["image_id"] for row in second["records"]
    ]
    assert 4 not in {row["image_id"] for row in first["records"]}
    assert all(row["ground_truth_count"] == 1 for row in first["records"])
    assert all(len(row["source_image_sha256"]) == 64 for row in first["records"])
    resumed = build_benchmark_manifest(root, dataset, max_images=3, resume=True)
    assert resumed["selection_policy"] == first["selection_policy"]


def test_aggregate_worker_rows_counts_unmatched_approvals_and_latency():
    rows = [
        {
            "status": "APPROVED",
            "ground_truth_count": 2,
            "detector_prediction_count": 2,
            "detector_matched_count": 2,
            "detector_count_correct": True,
            "classifier_item_count": 2,
            "classifier_item_matched_count": 2,
            "approved_count": 2,
            "approved_correct_count": 2,
            "unknown_count": 0,
            "unknown_matched_count": 0,
            "unknown_unmatched_count": 0,
            "unknown_top3_correct_count": 0,
            "classifier_executed": True,
            "processing_time_ms": 80,
            "expected_hard_gate": False,
            "hard_gate_classifier_called_violation": False,
            "hard_gate_response_status_violation": False,
            "hard_gate_classifier_version_violation": False,
            "hard_gate_reason_mismatch_violation": False,
            "hard_gate_contract_violation": False,
            "classifier_execution_version_mismatch": False,
            "classifier_call_count_violation": False,
            "pipeline_contract_violation": False,
        },
        {
            "status": "UNKNOWN",
            "ground_truth_count": 1,
            "detector_prediction_count": 2,
            "detector_matched_count": 1,
            "detector_count_correct": False,
            "classifier_item_count": 3,
            "classifier_item_matched_count": 1,
            "approved_count": 1,
            "approved_correct_count": 0,
            "unknown_count": 2,
            "unknown_matched_count": 1,
            "unknown_unmatched_count": 1,
            "unknown_top3_correct_count": 1,
            "classifier_executed": True,
            "processing_time_ms": 100,
            "expected_hard_gate": False,
            "hard_gate_classifier_called_violation": False,
            "hard_gate_response_status_violation": False,
            "hard_gate_classifier_version_violation": False,
            "hard_gate_reason_mismatch_violation": False,
            "hard_gate_contract_violation": False,
            "classifier_execution_version_mismatch": False,
            "classifier_call_count_violation": False,
            "pipeline_contract_violation": False,
        },
    ]
    report = aggregate_worker_rows(rows)
    assert report["detector"]["recall"] == 1.0
    assert report["detector"]["precision"] == pytest.approx(3 / 4)
    assert report["detector"]["count_accuracy"] == 0.5
    assert report["classifier"]["approved_precision"] == pytest.approx(2 / 3)
    assert report["classifier"]["unknown_top3_accuracy"] == 1.0
    assert report["classifier"]["unknown_count"] == 2
    assert report["classifier"]["unknown_matched_count"] == 1
    assert report["classifier"]["unknown_unmatched_count"] == 1
    assert report["full_path_latency_ms"]["p95"] == pytest.approx(99.0)


def test_aggregate_requires_real_approved_and_unknown_truth_samples():
    row = {
        "status": "RECAPTURE",
        "ground_truth_count": 1,
        "detector_prediction_count": 1,
        "detector_matched_count": 1,
        "detector_count_correct": True,
        "classifier_item_count": 0,
        "classifier_item_matched_count": 0,
        "approved_count": 0,
        "approved_correct_count": 0,
        "unknown_count": 0,
        "unknown_matched_count": 0,
        "unknown_unmatched_count": 0,
        "unknown_top3_correct_count": 0,
        "classifier_executed": False,
        "processing_time_ms": 10,
        "expected_hard_gate": True,
        "hard_gate_classifier_called_violation": False,
        "hard_gate_response_status_violation": False,
        "hard_gate_classifier_version_violation": False,
        "hard_gate_reason_mismatch_violation": False,
        "hard_gate_contract_violation": False,
        "classifier_execution_version_mismatch": False,
        "classifier_call_count_violation": False,
        "pipeline_contract_violation": False,
    }
    report = aggregate_worker_rows([row])
    assert report["classifier"]["approved_precision"] is None
    assert report["classifier"]["unknown_top3_accuracy"] is None


def test_match_items_maximizes_cardinality_deterministically(monkeypatch):
    overlaps = {
        (1, 10): 0.9,
        (1, 20): 0.8,
        (2, 10): 0.8,
        (2, 20): 0.1,
    }
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_operational._iou",
        lambda item, gt: overlaps[(int(item[0]), int(gt[0]))],
    )
    items = [
        np.asarray([1, 0, 2, 1], dtype=np.float64),
        np.asarray([2, 0, 3, 1], dtype=np.float64),
    ]
    annotations = [
        {"bbox_xywh": [10, 0, 1, 1]},
        {"bbox_xywh": [20, 0, 1, 1]},
    ]
    assert _match_items(items, annotations, 0.5) == {0: 1, 1: 0}
    assert _match_items(items, annotations, 0.5) == {0: 1, 1: 0}


def test_expected_hard_gate_includes_always_recapture_border_policy():
    metadata = SimpleNamespace(
        quality=QualityMetadata(
            border_policy="always_recapture", border_margin_ratio=0.01
        ),
        count_verifier=None,
    )
    result = DetectionResult(
        detections=[Detection(0.0, 2.0, 8.0, 8.0, 0.99)]
    )
    assert _expected_hard_gate_reasons(
        np.zeros((10, 10, 3), dtype=np.uint8), result, metadata
    ) == ["DETECTOR_BORDER_CLIPPED"]


def test_worker_eval_captures_detector_kpi_on_hard_gate_and_guards_resume(
    tmp_path: Path, monkeypatch
):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    detector_path = package_dir / "detector.onnx"
    classifier_path = package_dir / "classifier.onnx"
    metadata_path = package_dir / "metadata.json"
    detector_path.write_bytes(b"detector")
    classifier_path.write_bytes(b"classifier")
    metadata_path.write_text("{}", encoding="utf-8")
    classifier_metadata = ClassifierMetadata(
        filename="classifier.onnx",
        version="1.0.0",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        approval_threshold=0.9,
        temperature=1.0,
        labels=[{"class_id": "1", "class_name": "one"}],
    )
    package = SimpleNamespace(
        detector_path=detector_path,
        classifier_path=classifier_path,
        metadata=SimpleNamespace(
            package_version="1.0.0",
            dataset_version="rpc-test",
            detector=SimpleNamespace(filename="detector.onnx", version="1.0.0"),
            classifier=classifier_metadata,
            quality=QualityMetadata(border_policy="classifier_confidence"),
            count_verifier=None,
            input=SimpleNamespace(jpeg_draft_size=None),
        ),
    )

    class DetectorAdapter:
        version = "1.0.0"

        def __init__(self):
            self.calls = 0

        def detect(self, _image):
            self.calls += 1
            return DetectionResult(
                detections=[Detection(2.0, 2.0, 12.0, 12.0, 0.99)],
                capacity_saturated=self.calls == 1,
            )

    class ClassifierAdapter:
        version = "1.0.0"

        @staticmethod
        def classify(_image, _detections):
            return np.asarray([[1.0]], dtype=np.float32)

    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_operational.load_model_package", lambda _path: package
    )
    monkeypatch.setattr(
        "bixolon_scanner.training.rpc_operational.build_onnx_adapters",
        lambda _package, requested, **_kwargs: (
            DetectorAdapter(),
            ClassifierAdapter(),
            requested,
        ),
    )
    benchmark = tmp_path / "benchmark"
    (benchmark / "images").mkdir(parents=True)
    image_path = benchmark / "images" / "1.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    second_image_path = benchmark / "images" / "2.png"
    Image.new("RGB", (20, 20), "gray").save(second_image_path)
    manifest_path = benchmark / "manifest.json"
    _write_json(
        manifest_path,
        {
            "records": [
                {
                    "image_id": 1,
                    "image_path": "images/1.png",
                    "level": "easy",
                    "annotations": [
                        {"annotation_id": 1, "category_id": 1, "bbox_xywh": [2, 2, 10, 10]}
                    ],
                },
                {
                    "image_id": 2,
                    "image_path": "images/2.png",
                    "level": "medium",
                    "annotations": [
                        {"annotation_id": 2, "category_id": 1, "bbox_xywh": [2, 2, 10, 10]}
                    ],
                },
            ]
        },
    )
    output_path = tmp_path / "worker.json"
    rows_path = tmp_path / "worker.jsonl"
    report = worker_eval(
        package_dir,
        manifest_path,
        benchmark,
        output_path,
        rows_path=rows_path,
        provider="cuda",
    )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    row = rows[0]
    non_hard_row = rows[1]
    assert report["detector"]["recall"] == 1.0
    assert report["detector"]["count_accuracy"] == 1.0
    assert row["detector_prediction_count"] == 1
    assert row["detector_matched_count"] == 1
    assert row["classifier_item_count"] == 0
    assert row["classifier_item_matched_count"] == 0
    assert row["hard_gate_classifier_called_violation"] is False
    assert row["expected_hard_gate_reasons"] == ["DETECTOR_CAPACITY_EXCEEDED"]
    assert row["classifier_call_count_delta"] == 0
    assert row["classifier_version_reported"] is None
    assert non_hard_row["expected_hard_gate"] is False
    assert non_hard_row["classifier_call_count_delta"] == 1
    assert non_hard_row["classifier_call_count_violation"] is False
    assert non_hard_row["classifier_version_reported"] == "1.0.0"
    assert report["pipeline_contract"]["pipeline_contract_violations"] == 0
    state = json.loads(
        rows_path.with_suffix(".jsonl.state.json").read_text(encoding="utf-8")
    )
    assert state["completed_count"] == 2
    assert state["rows_checksum_scheme"] == "sha256-chain-v1"

    for invalid_threshold in (0.0, 1.01, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            worker_eval(
                package_dir,
                manifest_path,
                benchmark,
                output_path,
                match_iou_threshold=invalid_threshold,
            )

    with pytest.raises(ValueError, match="resume inputs changed"):
        worker_eval(
            package_dir,
            manifest_path,
            benchmark,
            output_path,
            rows_path=rows_path,
            provider="cpu",
            resume=True,
        )
    original_rows = rows_path.read_bytes()
    original_state = rows_path.with_suffix(".jsonl.state.json").read_text(encoding="utf-8")
    row["image_id"] = 999
    rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rows checksum"):
        worker_eval(
            package_dir,
            manifest_path,
            benchmark,
            output_path,
            rows_path=rows_path,
            provider="cuda",
            resume=True,
        )
    rows_path.write_bytes(original_rows)
    rows_path.with_suffix(".jsonl.state.json").write_text(
        original_state, encoding="utf-8"
    )
    Image.new("RGB", (20, 20), "black").save(image_path)
    with pytest.raises(ValueError, match="source image checksum changed"):
        worker_eval(
            package_dir,
            manifest_path,
            benchmark,
            output_path,
            rows_path=rows_path,
            provider="cuda",
            resume=True,
        )


def _package(package_dir: Path, labels: list[dict[str, object]], dataset_version: str):
    package_dir.mkdir(parents=True)
    (package_dir / "detector.onnx").write_bytes(b"detector")
    (package_dir / "classifier.onnx").write_bytes(b"classifier")
    _write_json(
        package_dir / "metadata.json",
        {
            "schema_version": "1.1",
            "package_version": "1.0.0",
            "promotion_status": "development",
            "dataset_version": dataset_version,
            "detector": {
                "filename": "detector.onnx",
                "version": "1.0.0",
                "score_threshold": 0.42,
                "uncertainty_score_threshold": 0.2,
                "uncertainty_min_area_ratio": 0.039,
                "uncertainty_match_iou_threshold": 0.5,
                "nms_iou_threshold": 0.7,
                "max_queries": 300,
                "resize_reducing_gap": 1.0,
            },
            "classifier": {
                "filename": "classifier.onnx",
                "version": "1.0.0",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "approval_threshold": 0.91,
                "temperature": 1.1,
                "crop_margin_ratio": 0.05,
                "resize_reducing_gap": 1.0,
                "labels": [
                    {"class_id": row["class_id"], "class_name": row["class_name"]}
                    for row in labels
                ],
            },
            "quality": {
                "min_object_area_ratio": 0.005,
                "border_margin_ratio": 0.002,
                "border_policy": "classifier_confidence",
            },
            "calibration": {
                "sample_count": 100,
                "approved_precision": 0.999,
                "approval_coverage": 0.8,
                "false_approval_rate_upper_95": 0.004,
                "risk_control_satisfied": True,
            },
            "detector_evaluation": {
                "recall": 0.995,
                "precision": 0.98,
                "count_accuracy": 0.97,
                "target_recall_satisfied": True,
            },
            "checksums": {
                "detector.onnx": sha256_file(package_dir / "detector.onnx"),
                "classifier.onnx": sha256_file(package_dir / "classifier.onnx"),
            },
            "licenses": {"detector": "test", "classifier": "test"},
        },
    )


def _sealed_worker_provenance(
    worker_path: Path, annotation_path: Path, source_path: Path
) -> dict[str, object]:
    rows_path = worker_path.with_suffix(".jsonl")
    state_path = rows_path.with_suffix(".jsonl.state.json")
    rows = [{"image_id": 1, "source_image_sha256": sha256_file(source_path)}]
    rows_path.write_text(json.dumps(rows[0], sort_keys=True) + "\n", encoding="utf-8")
    rows_chain, row_count = _rows_chain(rows_path)
    fingerprint = "e" * 64
    _write_json(
        state_path,
        {
            "completed_count": row_count,
            "input_fingerprint": fingerprint,
            "rows_sha256": rows_chain,
            "rows_checksum_scheme": "sha256-chain-v1",
        },
    )
    annotation_sha256 = sha256_file(annotation_path)
    return {
        "input_fingerprint": fingerprint,
        "test_manifest": {
            "contract": "sealed-rpc-test-manifest-v1",
            "manifest_format": "rpc_coco_test",
            "sealed_full_test": True,
            "manifest_sha256": annotation_sha256,
            "test_annotation_sha256": annotation_sha256,
            "manifest_row_count": 1,
            "evaluated_row_count": 1,
            "manifest_image_ids_sha256": _image_ids_sha256([1]),
            "evaluated_image_ids_sha256": _image_ids_sha256([1]),
            "source_image_set_sha256": _source_set_sha256(rows),
            "rows_filename": rows_path.name,
            "rows_file_sha256": sha256_file(rows_path),
            "rows_chain_sha256": rows_chain,
            "state_filename": state_path.name,
            "state_file_sha256": sha256_file(state_path),
            "state_completed_count": 1,
            "state_input_fingerprint": fingerprint,
            "state_rows_checksum_scheme": "sha256-chain-v1",
        },
    }


def test_integrate_marks_recapture_not_evaluable_and_enforces_latency_gate(tmp_path: Path):
    root, config_path = _locked_training_root(tmp_path)
    package_inputs(root, config_path)
    package_dir = tmp_path / "package"
    manifest_metadata = json.loads(
        (root / "package-inputs" / "manifest-metadata.json").read_text()
    )
    labels = manifest_metadata["labels"]
    dataset_version = manifest_metadata["dataset_version"]
    _package(package_dir, labels, dataset_version)
    detector_checkpoint_sha256 = "d" * 64
    rpc_root = tmp_path / "rpc"
    (rpc_root / "test2019").mkdir(parents=True)
    (rpc_root / "test2019" / "frame-1.jpg").write_bytes(b"benchmark-image")
    _write_json(
        rpc_root / "instances_test2019.json",
        {
            "images": [
                {
                    "id": 1,
                    "file_name": "frame-1.jpg",
                    "width": 20,
                    "height": 20,
                    "level": "easy",
                }
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 10]}
            ],
            "categories": _categories(),
        },
    )
    _write_json(
        root / "test" / "detector_report.json",
        {
            "detector_checkpoint_sha256": detector_checkpoint_sha256,
            "test_annotation_sha256": sha256_file(
                rpc_root / "instances_test2019.json"
            ),
            "outcomes": [
                {
                    "image_id": 1,
                    "level": "easy",
                    "ground_truth_count": 1,
                    "recapture_reasons": [],
                }
            ],
        },
    )
    build_benchmark_manifest(root, rpc_root, max_images=1)
    package_hashes = {
        "metadata.json": sha256_file(package_dir / "metadata.json"),
        "detector.onnx": sha256_file(package_dir / "detector.onnx"),
        "classifier.onnx": sha256_file(package_dir / "classifier.onnx"),
    }
    identity = {
        "package_version": "1.0.0",
        "dataset_version": dataset_version,
        "detector_version": "1.0.0",
        "classifier_version": "1.0.0",
        "package_artifact_sha256": package_hashes,
    }
    model_lock = json.loads((root / "model_lock.json").read_text())
    detector_report_sha256 = sha256_file(root / "test" / "detector_report.json")
    _write_json(
        root / "reports" / "final_test.json",
        {
            "status": "test_certified",
            "model_run": "runs/full/seed20260810",
            "classifier_checkpoint_sha256": model_lock["checkpoint_sha256"],
            "model_lock_sha256": sha256_file(root / "model_lock.json"),
            "detector_report_sha256": detector_report_sha256,
            "detector_checkpoint_sha256": detector_checkpoint_sha256,
        },
    )
    parities = []
    for provider in ("cpu", "cuda"):
        parity = root / "reports" / f"parity-{provider}.json"
        _write_json(
            parity,
            {
            **identity,
            "detector_checkpoint_sha256": detector_checkpoint_sha256,
            "classifier_checkpoint_sha256": model_lock["checkpoint_sha256"],
            "provider": provider,
            "detector": {
                "minimum_iou_tolerance": 0.99,
                "coordinate_tolerance": 0.01,
                "score_tolerance": 0.02,
            },
            "classifier": {"tolerance": 0.01},
            "passes": True,
            },
        )
        parities.append(parity)
    worker_path = root / "reports" / "worker-ort-accuracy.json"
    worker_provenance = _sealed_worker_provenance(
        worker_path,
        rpc_root / "instances_test2019.json",
        rpc_root / "test2019" / "frame-1.jpg",
    )
    _write_json(
        worker_path,
        {
            **identity,
            **worker_provenance,
            "provider": "cuda",
            "image_count": 1,
            "match_iou_threshold": 0.5,
            "detector": {"recall": 0.995, "precision": 0.99, "count_accuracy": 0.98},
            "classifier": {
                "approved_count": 100,
                "approved_precision": 0.999,
                "unknown_matched_count": 10,
                "unknown_top3_accuracy": 0.96,
            },
            "pipeline_contract": {"pipeline_contract_violations": 0},
        },
    )
    benchmark_manifest = root / "benchmark" / "manifest.json"
    benchmark_checksums = root / "benchmark" / "checksums.json"
    benchmark_records = json.loads(benchmark_manifest.read_text())["records"]
    image_hashes = {
        record["image_path"]: sha256_file(root / "benchmark" / record["image_path"])
        for record in benchmark_records
    }
    _write_json(
        root / "reports" / "benchmark.json",
        {
            **identity,
            "provider": "cuda",
            "sample_count": 1000,
            "warmup_count": 30,
            "onnxruntime_version": "1.28.0",
            "onnxruntime_build_info": "CUDA 13 test build",
            "benchmark_manifest_sha256": sha256_file(benchmark_manifest),
            "benchmark_manifest_checksums_sha256": sha256_file(benchmark_checksums),
            "image_artifact_sha256": image_hashes,
            "gpu": {
                "name": "NVIDIA GeForce RTX 5080",
                "driver_version": "591.86",
                "memory_total_mib": 16303,
                "cuda_version": "13.0",
                "physical_index": 0,
                "uuid": "GPU-test-uuid",
                "physical_gpu_count": 1,
                "ort_cuda_device_id": 0,
                "selection_source": "single_physical_gpu",
                "cuda_visible_devices": None,
                "cuda_device_order": None,
            },
            "system": {
                "windows_build": 26200,
                "cpu": "Intel(R) Core(TM) Ultra 9 285K",
                "memory_total_gib": 63.7,
            },
            "by_path": {
                "full_path": {
                    "sample_count": 1000,
                    "p50_ms": 60,
                    "p95_ms": 101,
                    "p99_ms": 110,
                }
            }
        },
    )
    report = integrate(
        root, package_dir, config_path=config_path, parity_report_paths=parities
    )
    assert report["metrics"]["recapture_recall"] == "not_evaluable"
    assert report["certification"]["status"] == "not_certified"
    assert report["certification"]["production_certified"] is False
    assert report["gates"]["full_path_p95_at_most_100_ms"] is False
    assert report["gates"]["worker_match_iou_threshold_is_0_5"] is True
    assert report["gates"]["approved_truth_sample_present"] is True
    assert report["gates"]["unknown_truth_sample_present"] is True
    assert report["gates"]["benchmark_gpu_is_desktop_rtx_5080"] is True
    assert report["gates"]["benchmark_windows_11_build"] is True
    assert report["gates"]["parity_cpu_and_cuda_present"] is True
    assert report["gates"]["parity_tolerance_policy_exact"] is True
    assert report["gates"]["benchmark_gpu_device_binding_evidenced"] is True
    assert report["gates"]["all_benchmark_samples_are_full_path"] is True
    assert report["environment"]["request_concurrency"] == 1
    assert report["environment"]["gpu_name"] == "NVIDIA GeForce RTX 5080"
    assert report["environment"]["windows_build"] == 26200
    assert report["detector_version"] == "1.0.0"
    assert report["classifier_version"] == "1.0.0"
    assert report["dataset_evidence"]["test_annotation_sha256"] == sha256_file(
        rpc_root / "instances_test2019.json"
    )
    assert report["dataset_evidence"]["worker_test_row_count"] == 1
    assert report["all_evaluable_gates_satisfied"] is False
    assert report["artifact_sha256"]["worker_ort_accuracy"] == sha256_file(
        root / "reports" / "worker-ort-accuracy.json"
    )
    with pytest.raises(ValueError, match="final detector evidence"):
        _validate_benchmark_manifest_evidence(
            root / "benchmark",
            json.loads((root / "reports" / "benchmark.json").read_text()),
            sha256_file(root / "test" / "detector_report.json"),
            "f" * 64,
        )
    worker_payload = json.loads(worker_path.read_text())
    no_truth_payload = json.loads(json.dumps(worker_payload))
    no_truth_payload["match_iou_threshold"] = 0.4
    no_truth_payload["classifier"].update(
        {
            "approved_count": 0,
            "approved_precision": None,
            "unknown_matched_count": 0,
            "unknown_top3_accuracy": None,
        }
    )
    _write_json(worker_path, no_truth_payload)
    no_truth_report = integrate(
        root, package_dir, config_path=config_path, parity_report_paths=parities
    )
    assert no_truth_report["gates"]["approved_truth_sample_present"] is False
    assert no_truth_report["gates"]["unknown_truth_sample_present"] is False
    assert no_truth_report["gates"]["worker_match_iou_threshold_is_0_5"] is False
    _write_json(worker_path, worker_payload)
    partial_payload = json.loads(json.dumps(worker_payload))
    partial_payload["test_manifest"]["manifest_row_count"] = 0
    _write_json(worker_path, partial_payload)
    with pytest.raises(ValueError, match="complete sealed test manifest"):
        integrate(
            root, package_dir, config_path=config_path, parity_report_paths=parities
        )
    _write_json(worker_path, worker_payload)
    cpu_parity_payload = json.loads(parities[0].read_text())
    altered_parity = json.loads(json.dumps(cpu_parity_payload))
    altered_parity["classifier"]["tolerance"] = 0.02
    _write_json(parities[0], altered_parity)
    policy_report = integrate(
        root, package_dir, config_path=config_path, parity_report_paths=parities
    )
    assert policy_report["gates"]["parity_tolerance_policy_exact"] is False
    _write_json(parities[0], cpu_parity_payload)
    benchmark_image = root / "benchmark" / benchmark_records[0]["image_path"]
    benchmark_image_bytes = benchmark_image.read_bytes()
    benchmark_image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="benchmark image checksum"):
        integrate(
            root, package_dir, config_path=config_path, parity_report_paths=parities
        )
    benchmark_image.write_bytes(benchmark_image_bytes)
    config_path.write_text(config_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifacts"):
        integrate(
            root, package_dir, config_path=config_path, parity_report_paths=parities
        )


def test_integrate_fails_when_required_evidence_is_absent(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        integrate(
            tmp_path,
            tmp_path / "package",
            config_path=tmp_path / "config.json",
            parity_report_paths=[],
        )


def test_operation_plan_reports_ready_stages_without_running_gpu(tmp_path: Path):
    root, config_path = _locked_training_root(tmp_path)
    _write_json(root / "test" / "detector_report.json", {"outcomes": []})
    _write_json(root / "reports" / "final_test.json", {"status": "test_certified"})
    package_dir = tmp_path / "package"
    report = operation_plan(root, package_dir, config_path=config_path)

    by_id = {stage["id"]: stage for stage in report["stages"]}
    assert report["dag_contract"] == "rpc-operational-dag-v1"
    assert by_id["final_test"]["status"] == "completed"
    assert by_id["package_inputs"]["status"] == "ready"
    assert by_id["detector_classifier_package"]["status"] == "blocked"
    assert report["next_steps"] == ["package_inputs", "benchmark_manifest"]
    assert report["certification_constraint"]["production_certification"] == (
        "not_certified"
    )


def test_operation_plan_blocks_final_test_until_model_lock_exists(tmp_path: Path):
    root = tmp_path / "experiment"
    config_path = tmp_path / "config.json"
    _write_json(config_path, {"experiment": {"mode": "full_dataset"}})

    report = operation_plan(root, tmp_path / "package", config_path=config_path)

    by_id = {stage["id"]: stage for stage in report["stages"]}
    assert by_id["final_test"]["status"] == "blocked"
    assert by_id["final_test"]["missing_required_inputs"] == [
        str((root / "model_lock.json").resolve())
    ]
    assert "final_test" not in report["next_steps"]
