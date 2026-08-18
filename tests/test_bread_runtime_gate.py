import argparse
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from bixolon_scanner.contracts import (
    BoundingBox,
    ItemStatus,
    Prediction,
    ScanItem,
    ScanResponse,
    Status,
)
from bixolon_scanner.evaluation.bread_runtime_gate import (
    RuntimeGateCounts,
    _decision_trace,
    _validate_independent_preflight,
    build_runtime_gate_metrics,
)
from bixolon_scanner.experiments.bread.independent_preflight import audit_independent_dataset
from bixolon_scanner.training.bread_cv import difference_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_gate_keeps_recaptured_and_missed_gt_in_approved_denominator() -> None:
    metrics = build_runtime_gate_metrics(
        RuntimeGateCounts(
            image_count=10,
            segmentation_image_count=9,
            image_recapture_count=1,
            judgeable_ground_truth_object_count=100,
            matched_segmentation_count=98,
            false_negative_count=1,
            false_negative_image_count=1,
            approved_count=97,
            approved_misrecognition_count=1,
            unknown_count=1,
            unknown_top3_candidate_out_count=1,
        )
    )

    assert metrics["rates"]["end_to_end_approved_object_rate"] == 0.97
    assert metrics["rates"]["approved_object_misrecognition_rate"] == 0.01
    assert metrics["rates"]["unknown_top3_candidate_out_rate"] == 0.01
    assert metrics["rates"]["segmentation_image_false_negative_rate"] == 1 / 9
    assert not metrics["final_end_to_end_approved_goal_met"]


def test_runtime_gate_boundaries_are_inclusive() -> None:
    metrics = build_runtime_gate_metrics(
        RuntimeGateCounts(
            image_count=1000,
            segmentation_image_count=900,
            judgeable_ground_truth_object_count=1000,
            approved_count=990,
            approved_misrecognition_count=1,
            unknown_top3_candidate_out_count=1,
        )
    )

    assert metrics["operational_gates"]["segmentation_image_rate"]
    assert metrics["operational_gates"]["approved_object_misrecognition_rate"]
    assert metrics["operational_gates"]["unknown_top3_candidate_out_rate"]
    assert metrics["final_end_to_end_approved_goal_met"]


def test_decision_trace_excludes_request_specific_values() -> None:
    response = ScanResponse(
        request_id="request-123",
        status=Status.SEGMENTATION,
        segmentations=[
            ScanItem(
                segmentation_id="segmentation_001",
                bbox=BoundingBox(x=1, y=2, width=3, height=4),
                status=ItemStatus.APPROVED,
                prediction=Prediction(class_id="bread_01", class_name="bread_01"),
                confidence=0.99,
            )
        ],
        processing_time_ms=12.3,
        worker_version="1.1.0",
        detector_version="1.1.0",
        classifier_version="1.1.0",
    )

    trace = _decision_trace(7, response)

    assert trace["image_id"] == 7
    assert "request_id" not in trace
    assert "processing_time_ms" not in trace
    assert trace["segmentations"][0]["prediction"]["class_id"] == "bread_01"


def test_independent_runtime_requires_model_free_preflight_before_inference(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    annotations = dataset_root / "annotations"
    annotations.mkdir(parents=True)
    annotation_path = annotations / "instances.json"
    annotation_path.write_text(
        json.dumps({"images": [], "annotations": [], "categories": []}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        evidence_role="independent",
        preflight_report=None,
        candidate_manifest=None,
        dataset_root=dataset_root,
        dataset_version="independent-v1",
    )

    with pytest.raises(ValueError, match="requires --preflight-report"):
        _validate_independent_preflight(args, annotation_path)


def test_independent_runtime_accepts_matching_eligible_preflight(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images = dataset_root / "images"
    annotations = dataset_root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    image_path = images / "candidate.png"
    Image.new("RGB", (16, 12), (30, 90, 160)).save(image_path)
    annotation_path = annotations / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "candidate.png", "width": 16, "height": 12}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 8]}],
                "categories": [{"id": 1, "name": "bread"}],
            }
        ),
        encoding="utf-8",
    )
    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(
        json.dumps({"annotation_review_status": "finalized"}), encoding="utf-8"
    )
    record_manifest_path = dataset_root / "manifest.jsonl"
    record_manifest_path.write_text(
        json.dumps(
            {
                "image_id": 1,
                "annotation_review_status": "finalized",
                "capture_session_id": "new-session",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_image = tmp_path / "source.png"
    source = Image.new("RGB", (24, 24))
    source.putdata(
        [(255, 255, 255) if (x // 3) % 2 else (0, 0, 0) for y in range(24) for x in range(24)]
    )
    source.save(source_image)
    source_manifest = tmp_path / "source.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "image_sha256": _sha(source_image),
                "perceptual_hash": difference_hash(source_image),
                "image_path": "development/source.png",
                "evaluation_set": "development",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    package_metadata = package_dir / "metadata.json"
    package_metadata.write_text("{}\n", encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "candidate_id": "candidate-v3",
                "lifecycle": "active_development",
                "package": {"metadata_sha256": _sha(package_metadata)},
                "independent_preflight": {
                    "fixed_git_commit": "adfae95",
                    "required_source_manifests": [
                        {"path": "source.jsonl", "sha256": _sha(source_manifest)}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    report = audit_independent_dataset(
        dataset_root=dataset_root,
        annotation_path=annotation_path,
        metadata_path=metadata_path,
        record_manifest_path=record_manifest_path,
        source_manifest_paths=[source_manifest],
        candidate_manifest_path=candidate_path,
        dataset_version="independent-v1",
        candidate_id="candidate-v3",
        candidate_commit="adfae95",
    )
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(report), encoding="utf-8")
    args = argparse.Namespace(
        evidence_role="independent",
        preflight_report=preflight_path,
        candidate_manifest=candidate_path,
        dataset_root=dataset_root,
        dataset_version="independent-v1",
        package_dir=package_dir,
    )

    evidence = _validate_independent_preflight(args, annotation_path)

    assert evidence is not None
    assert evidence["candidate_id"] == "candidate-v3"
