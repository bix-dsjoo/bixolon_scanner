from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bixolon_scanner.training.promote_package import (
    owner_waiver_metadata,
    production_metadata,
    promote_package,
)


def _candidate() -> dict:
    return {
        "schema_version": "1.1",
        "promotion_status": "development",
        "worker_version": "1.0.0",
        "dataset_version": "bread-v1",
        "detector": {"version": "1.0.0"},
        "classifier": {"version": "1.0.0"},
    }


def _report() -> dict:
    return {
        "dataset_version": "bread-v1",
        "gate_dataset": "multi_object_scenes",
        "versions": {
            "worker_version": "1.0.0",
            "detector_version": "1.0.0",
            "classifier_version": "1.0.0",
        },
        "effective_configuration": {
            "jpeg_draft_size_overridden": False,
            "approval_threshold_overridden": False,
        },
        "targets": {"maximum_misrecognition_rate": 0.001},
        "risk_evidence": {
            "approved_sample_count": 1121,
            "observed_error_count": 0,
            "upper_95": 0.0026688072943848507,
        },
        "checks": {
            "recognition_accuracy": True,
            "approved_misrecognition_rate": True,
            "approved_misrecognition_risk_upper_95": False,
            "segmentation_recall": True,
            "segmentation_precision": True,
            "mean_latency": True,
            "p95_latency": True,
        },
        "failures": ["approved_misrecognition_risk_upper_95"],
    }


def test_owner_approved_statistical_risk_is_recorded_without_hiding_it():
    metadata = production_metadata(_candidate(), _report(), decided_on="2026-08-14")

    assert metadata["promotion_status"] == "production"
    assert metadata["promotion"]["method"] == "manual_waiver"
    waiver = metadata["promotion"]["waivers"][0]
    assert waiver["gate"] == "approved_misrecognition_rate_upper_95"
    assert waiver["observed"] == pytest.approx(0.0026688072943848507)
    assert waiver["target"] == 0.001
    assert waiver["sample_count"] == 1121
    assert metadata["promotion"]["remaining_limitations"]


def test_promotion_rejects_any_additional_failed_gate():
    report = copy.deepcopy(_report())
    report["checks"]["recognition_accuracy"] = False
    report["failures"].append("recognition_accuracy")

    with pytest.raises(ValueError, match="unexpected release gate failures"):
        production_metadata(_candidate(), report, decided_on="2026-08-14")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jpeg_draft_size_overridden", True),
        ("approval_threshold_overridden", True),
    ],
)
def test_promotion_rejects_evaluation_overrides(field: str, value: bool):
    report = copy.deepcopy(_report())
    report["effective_configuration"][field] = value

    with pytest.raises(ValueError, match="override"):
        production_metadata(_candidate(), report, decided_on="2026-08-14")


def test_promotion_never_overwrites_an_existing_version_directory(tmp_path):
    production = tmp_path / "bread-worker-1.0.0"
    production.mkdir()
    marker = production / "metadata.json"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        promote_package(
            tmp_path / "candidate",
            tmp_path / "report.json",
            production,
            decided_on="2026-08-14",
            approve_statistical_risk=True,
        )

    assert marker.read_text(encoding="utf-8") == "existing"


def test_owner_approved_bread_1_1_bridge_records_both_known_limitations():
    candidate = _candidate()
    candidate["worker_version"] = "1.1.0"
    candidate["detector"]["version"] = "1.1.0"
    candidate["classifier"]["version"] = "1.1.0"
    report = {
        "promotion_method": "owner_approved_known_limitations",
        "source_candidate_id": "bread-zero-error-1.1.0-domain-lda-fixed-four-v3",
        "dataset_version": "bread-v1",
        "versions": {
            "worker_version": "1.1.0",
            "detector_version": "1.1.0",
            "classifier_version": "1.1.0",
        },
        "checks": {
            "six_operational_gates": True,
            "latency": True,
            "cpu_cuda_parity": True,
            "classifier_training_source_restriction": False,
            "evaluation_set_independence": False,
        },
        "failures": [
            "classifier_training_source_restriction",
            "evaluation_set_independence",
        ],
        "waivers": [
            {
                "gate": "classifier_training_source_restriction",
                "observed": 1.0,
                "target": 0.0,
                "sample_count": 1410,
                "correct_count": 0,
                "reason": "The final LDA head used same-domain development ROIs.",
            },
            {
                "gate": "evaluation_set_independence",
                "observed": 0.0,
                "target": 1.0,
                "sample_count": 300,
                "correct_count": 0,
                "reason": "No unseen locked test is available for this bridge release.",
            },
        ],
        "remaining_limitations": ["Replace the classifier in Worker 1.1.1."],
    }

    metadata = owner_waiver_metadata(candidate, report, decided_on="2026-08-19")

    assert metadata["promotion_status"] == "production"
    assert {row["gate"] for row in metadata["promotion"]["waivers"]} == {
        "classifier_training_source_restriction",
        "evaluation_set_independence",
    }
    assert metadata["promotion"]["remaining_limitations"] == [
        "Replace the classifier in Worker 1.1.1."
    ]


def test_owner_waiver_promotion_copies_every_detector_ensemble_member(tmp_path: Path, monkeypatch):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    detector_paths = [candidate_dir / f"detector-{index}.onnx" for index in range(4)]
    classifier_path = candidate_dir / "classifier.onnx"
    for index, path in enumerate(detector_paths):
        path.write_bytes(f"detector-{index}".encode())
    classifier_path.write_bytes(b"classifier")
    candidate_metadata = {
        "schema_version": "2.0",
        "promotion_status": "development",
        "worker_version": "1.1.0",
        "dataset_version": "bread-v1",
        "detector": {"version": "1.1.0"},
        "classifier": {"version": "1.1.0"},
    }
    metadata_path = candidate_dir / "metadata.json"
    metadata_path.write_text(json.dumps(candidate_metadata) + "\n", encoding="utf-8")
    metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    report = {
        "promotion_method": "owner_approved_known_limitations",
        "source_candidate_id": "bread-zero-error-1.1.0-domain-lda-fixed-four-v3",
        "source_candidate_metadata_sha256": metadata_sha256,
        "dataset_version": "bread-v1",
        "versions": {
            "worker_version": "1.1.0",
            "detector_version": "1.1.0",
            "classifier_version": "1.1.0",
        },
        "checks": {
            "classifier_training_source_restriction": False,
            "evaluation_set_independence": False,
            "six_operational_gates": True,
        },
        "failures": [
            "classifier_training_source_restriction",
            "evaluation_set_independence",
        ],
        "waivers": [
            {
                "gate": "classifier_training_source_restriction",
                "observed": 1.0,
                "target": 0.0,
                "sample_count": 1410,
                "correct_count": 0,
                "reason": "Bridge release classifier source exception.",
            },
            {
                "gate": "evaluation_set_independence",
                "observed": 0.0,
                "target": 1.0,
                "sample_count": 300,
                "correct_count": 0,
                "reason": "Bridge release independent test exception.",
            },
        ],
        "remaining_limitations": ["Replace the classifier in 1.1.1."],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    package_metadata = SimpleNamespace(
        promotion_status="production",
        worker_version="1.1.0",
        detector=SimpleNamespace(version="1.1.0"),
        classifier=SimpleNamespace(version="1.1.0"),
        dataset_version="bread-v1",
    )
    candidate = SimpleNamespace(
        detector_paths=detector_paths,
        classifier_path=classifier_path,
        count_verifier_path=None,
    )

    def fake_load(path: Path):
        if path.resolve() == candidate_dir.resolve():
            return candidate
        return SimpleNamespace(metadata=package_metadata)

    monkeypatch.setattr("bixolon_scanner.training.promote_package.load_model_package", fake_load)
    output_dir = tmp_path / "bread-worker-1.1.0"

    result = promote_package(
        candidate_dir,
        report_path,
        output_dir,
        decided_on="2026-08-19",
        approve_statistical_risk=False,
        approve_known_limitations=True,
    )

    assert result["promotion_status"] == "production"
    assert {path.name for path in output_dir.glob("*.onnx")} == {
        *(path.name for path in detector_paths),
        classifier_path.name,
    }
