from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bixolon_scanner.evaluation.yolo_free_n100_result import (
    HYBRID_PROFILE,
    assess_yolo_free_n100_result,
)


def _profile(mean: float = 800.0, p95: float = 950.0) -> dict[str, object]:
    return {
        "completed": True,
        "failure_code": None,
        "detector_provider": "openvino",
        "embedder_provider": "same",
        "client_total_ms": {"full_path": {"sample_count": 30, "mean": mean, "p95": p95}},
        "full_path_count": 30,
        "error_count": 0,
    }


def _passing_report() -> dict[str, object]:
    cpu = _profile()
    hybrid = _profile(700.0, 900.0)
    hybrid["embedder_provider"] = "openvino_gpu"
    return {
        "schema_version": "1.1",
        "evaluation": "bixolon_worker_n100_openvino_cpu_vs_intel_gpu_embedder",
        "product_version": "0.1.7",
        "completed": True,
        "passes": True,
        "hardware": {
            "cpu_name": "Intel(R) Processor N100",
            "target_cpu_detected": True,
            "target_intel_gpu_detected": True,
            "video_controllers": [{"name": "Intel(R) UHD Graphics"}],
        },
        "input": {
            "sample_count": 30,
            "image_sha256": [f"{index:064x}" for index in range(30)],
        },
        "selection_policy": {
            "maximum_full_path_mean_ms": 1000.0,
            "maximum_full_path_p95_ms": 1000.0,
            "maximum_confidence_delta": 0.00001,
            "require_semantic_and_confidence_parity": True,
        },
        "execution_contract": {
            "same_worker_executable": True,
            "same_runtime_catalog_and_policy": True,
            "count_verifier_configured": True,
            "count_verifier_comparison_mode": "exact_count",
            "cpu_detector_gpu_embedder": {"silent_cpu_fallback_allowed": False},
        },
        "integrity": {
            "model_graph_or_weight_changed": False,
            "decision_policy_changed": False,
        },
        "profiles": {
            "openvino_cpu_only": cpu,
            HYBRID_PROFILE: hybrid,
        },
        "parity": {
            "evaluated": True,
            "safe": True,
            "semantic_mismatch_count": 0,
            "confidence_vector_mismatch_count": 0,
            "maximum_confidence_delta": 0.000001,
            "confidence_tolerance": 0.00001,
        },
        "comparison": {
            "provider_initialization_safe": True,
            "cpu_only_target_met": True,
            "gpu_embedder_target_met": True,
            "target_met_by_any_profile": True,
            "recommended_provider": "openvino+openvino_gpu",
        },
        "privacy": {
            "image_paths_recorded": False,
            "image_bytes_recorded": False,
            "image_sha256_recorded": True,
        },
    }


def _assess(tmp_path: Path, report: dict[str, object]) -> dict[str, object]:
    path = tmp_path / "n100-yolo-free-0.1.7-openvino-device-matrix.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return assess_yolo_free_n100_result(path)


def test_accepts_complete_n100_result(tmp_path: Path) -> None:
    assessment = _assess(tmp_path, _passing_report())

    assert assessment["passed"] is True
    assert assessment["failed_criteria"] == []
    assert assessment["selected_provider"] == "openvino+openvino_gpu"
    assert assessment["source_report"]["filename"] == (
        "n100-yolo-free-0.1.7-openvino-device-matrix.json"
    )


@pytest.mark.parametrize(
    ("mutation", "failed_criterion"),
    [
        (
            lambda value: value["hardware"].update(target_cpu_detected=False),
            "target_n100_cpu_detected",
        ),
        (
            lambda value: value["hardware"].update(target_intel_gpu_detected=False),
            "target_intel_gpu_detected",
        ),
        (
            lambda value: value["profiles"][HYBRID_PROFILE].update(
                completed=False, failure_code="OPENVINO_GPU_PROFILE_FAILED"
            ),
            "hybrid_profile_completed",
        ),
        (
            lambda value: value["profiles"][HYBRID_PROFILE].update(error_count=1),
            "no_worker_errors",
        ),
        (
            lambda value: value["parity"].update(semantic_mismatch_count=1, safe=False),
            "semantic_parity",
        ),
        (
            lambda value: value["parity"].update(maximum_confidence_delta=0.020001),
            "confidence_parity",
        ),
        (
            lambda value: value["privacy"].update(image_paths_recorded=True),
            "privacy_safe",
        ),
    ],
)
def test_rejects_invalid_hardware_execution_or_parity(
    tmp_path: Path,
    mutation: object,
    failed_criterion: str,
) -> None:
    report = copy.deepcopy(_passing_report())
    mutation(report)

    assessment = _assess(tmp_path, report)

    assert assessment["passed"] is False
    assert failed_criterion in assessment["failed_criteria"]


def test_rejects_when_no_profile_meets_mean_and_p95_target(tmp_path: Path) -> None:
    report = _passing_report()
    for profile in report["profiles"].values():
        profile["client_total_ms"]["full_path"].update(mean=1001.0, p95=1200.0)
    report["comparison"].update(
        cpu_only_target_met=False,
        gpu_embedder_target_met=False,
        target_met_by_any_profile=False,
    )

    assessment = _assess(tmp_path, report)

    assert assessment["passed"] is False
    assert "at_least_one_profile_within_1000ms" in assessment["failed_criteria"]
    assert "selected_provider_within_1000ms" in assessment["failed_criteria"]


def test_accepts_ssdlite_without_count_verifier_and_with_measured_provider_tolerance(
    tmp_path: Path,
) -> None:
    report = _passing_report()
    report["execution_contract"].update(
        count_verifier_configured=False,
        count_verifier_comparison_mode=None,
    )
    report["selection_policy"]["maximum_confidence_delta"] = 0.02
    report["parity"].update(
        maximum_confidence_delta=0.01372,
        confidence_tolerance=0.02,
    )

    assessment = _assess(tmp_path, report)

    assert assessment["passed"] is True


def test_reports_semantic_parity_independently_from_confidence_parity(tmp_path: Path) -> None:
    report = _passing_report()
    report["passes"] = False
    report["parity"].update(
        safe=False,
        maximum_confidence_delta=0.020001,
    )

    assessment = _assess(tmp_path, report)

    assert assessment["criteria"]["semantic_parity"] is True
    assert assessment["criteria"]["confidence_parity"] is False


def test_reassesses_legacy_report_against_current_confidence_limit(tmp_path: Path) -> None:
    report = _passing_report()
    report["passes"] = False
    report["selection_policy"]["maximum_confidence_delta"] = 0.01
    report["parity"].update(
        safe=False,
        maximum_confidence_delta=0.01372,
        confidence_tolerance=0.01,
    )

    assessment = _assess(tmp_path, report)

    assert assessment["passed"] is True
    assert assessment["source_report"]["self_assessment_passed"] is False
    assert assessment["source_report"]["confidence_tolerance"] == 0.01


def test_rejects_private_image_payload_even_when_privacy_flags_are_false(tmp_path: Path) -> None:
    report = _passing_report()
    report["input"]["image_paths"] = ["C:/private/image.jpg"]

    assessment = _assess(tmp_path, report)

    assert assessment["passed"] is False
    assert "privacy_safe" in assessment["failed_criteria"]
