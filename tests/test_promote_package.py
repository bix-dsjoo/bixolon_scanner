from __future__ import annotations

import copy

import pytest

from bixolon_scanner.training.promote_package import production_metadata, promote_package


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
