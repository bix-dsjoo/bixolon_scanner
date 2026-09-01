from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ..contracts.catalog import sha256_file

EXPECTED_EVALUATION = "bixolon_worker_n100_openvino_cpu_vs_intel_gpu_embedder"
EXPECTED_PRODUCT_VERSION = "0.1.7"
CPU_PROFILE = "openvino_cpu_only"
HYBRID_PROFILE = "openvino_cpu_detector_intel_gpu_embedder"
MINIMUM_IMAGE_COUNT = 30
MINIMUM_FULL_PATH_COUNT = 10
MAXIMUM_FULL_PATH_LATENCY_MS = 1000.0
MAXIMUM_CONFIDENCE_DELTA = 0.02
FORBIDDEN_IMAGE_PAYLOAD_KEYS = {
    "image_path",
    "image_paths",
    "image_bytes",
    "raw_image",
    "raw_images",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    rendered = float(value)
    return rendered if math.isfinite(rendered) else None


def _profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    full_path = _mapping(
        _mapping(profile.get("client_total_ms"), "profile.client_total_ms").get("full_path"),
        "profile.client_total_ms.full_path",
    )
    return {
        "completed": profile.get("completed") is True,
        "failure_code": profile.get("failure_code"),
        "full_path_count": profile.get("full_path_count"),
        "error_count": profile.get("error_count"),
        "mean_ms": _finite_number(full_path.get("mean")),
        "p95_ms": _finite_number(full_path.get("p95")),
    }


def _latency_target_met(profile: dict[str, Any]) -> bool:
    summary = _profile_summary(profile)
    mean = summary["mean_ms"]
    p95 = summary["p95_ms"]
    return (
        mean is not None
        and p95 is not None
        and mean <= MAXIMUM_FULL_PATH_LATENCY_MS
        and p95 <= MAXIMUM_FULL_PATH_LATENCY_MS
    )


def _contains_forbidden_image_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_IMAGE_PAYLOAD_KEYS or _contains_forbidden_image_payload(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_image_payload(item) for item in value)
    return False


def _valid_sha256_list(value: Any, expected_count: int) -> bool:
    if not isinstance(value, list) or len(value) != expected_count:
        return False
    hexadecimal = set("0123456789abcdef")
    return all(
        isinstance(item, str) and len(item) == 64 and set(item.lower()) <= hexadecimal
        for item in value
    )


def assess_yolo_free_n100_result(report_path: Path) -> dict[str, Any]:
    payload = _mapping(
        json.loads(report_path.read_text(encoding="utf-8")),
        "N100 benchmark report",
    )
    hardware = _mapping(payload.get("hardware"), "hardware")
    input_evidence = _mapping(payload.get("input"), "input")
    policy = _mapping(payload.get("selection_policy"), "selection_policy")
    execution = _mapping(payload.get("execution_contract"), "execution_contract")
    integrity = _mapping(payload.get("integrity"), "integrity")
    profiles = _mapping(payload.get("profiles"), "profiles")
    cpu = _mapping(profiles.get(CPU_PROFILE), f"profiles.{CPU_PROFILE}")
    hybrid = _mapping(profiles.get(HYBRID_PROFILE), f"profiles.{HYBRID_PROFILE}")
    parity = _mapping(payload.get("parity"), "parity")
    comparison = _mapping(payload.get("comparison"), "comparison")
    privacy = _mapping(payload.get("privacy"), "privacy")

    sample_count = input_evidence.get("sample_count")
    cpu_summary = _profile_summary(cpu)
    hybrid_summary = _profile_summary(hybrid)
    cpu_target_met = _latency_target_met(cpu)
    hybrid_target_met = _latency_target_met(hybrid)
    provider_to_profile = {
        "openvino": CPU_PROFILE,
        "openvino+openvino_gpu": HYBRID_PROFILE,
    }
    selected_provider = comparison.get("recommended_provider")
    selected_profile = provider_to_profile.get(selected_provider)
    selected_target_met = {
        CPU_PROFILE: cpu_target_met,
        HYBRID_PROFILE: hybrid_target_met,
    }.get(selected_profile, False)
    maximum_confidence_delta = _finite_number(parity.get("maximum_confidence_delta"))
    confidence_tolerance = _finite_number(parity.get("confidence_tolerance"))
    count_verifier_configured = execution.get("count_verifier_configured")
    count_verifier_mode = execution.get("count_verifier_comparison_mode")
    count_verifier_contract_safe = (
        count_verifier_mode == "exact_count" and count_verifier_configured in {None, True}
    ) or (count_verifier_mode is None and count_verifier_configured is False)

    criteria = {
        "expected_report_identity": (
            payload.get("schema_version") == "1.1"
            and payload.get("evaluation") == EXPECTED_EVALUATION
        ),
        "expected_product_version": payload.get("product_version") == EXPECTED_PRODUCT_VERSION,
        "benchmark_completed": payload.get("completed") is True,
        "target_n100_cpu_detected": hardware.get("target_cpu_detected") is True,
        "target_intel_gpu_detected": hardware.get("target_intel_gpu_detected") is True,
        "minimum_image_sample": (
            isinstance(sample_count, int)
            and not isinstance(sample_count, bool)
            and sample_count >= MINIMUM_IMAGE_COUNT
        ),
        "fixed_execution_contract": (
            execution.get("same_worker_executable") is True
            and execution.get("same_runtime_catalog_and_policy") is True
            and count_verifier_contract_safe
            and _mapping(
                execution.get("cpu_detector_gpu_embedder"),
                "execution_contract.cpu_detector_gpu_embedder",
            ).get("silent_cpu_fallback_allowed")
            is False
        ),
        "model_and_policy_unchanged": (
            integrity.get("model_graph_or_weight_changed") is False
            and integrity.get("decision_policy_changed") is False
        ),
        "fixed_latency_and_parity_policy": (
            _finite_number(policy.get("maximum_full_path_mean_ms")) == MAXIMUM_FULL_PATH_LATENCY_MS
            and _finite_number(policy.get("maximum_full_path_p95_ms"))
            == MAXIMUM_FULL_PATH_LATENCY_MS
            and _finite_number(policy.get("maximum_confidence_delta"))
            in {None, confidence_tolerance}
            and policy.get("require_semantic_and_confidence_parity") is True
        ),
        "cpu_profile_completed": (
            cpu_summary["completed"]
            and cpu_summary["failure_code"] is None
            and cpu.get("detector_provider") == "openvino"
            and cpu.get("embedder_provider") == "same"
        ),
        "hybrid_profile_completed": (
            hybrid_summary["completed"]
            and hybrid_summary["failure_code"] is None
            and hybrid.get("detector_provider") == "openvino"
            and hybrid.get("embedder_provider") == "openvino_gpu"
        ),
        "minimum_full_path_samples": (
            isinstance(cpu_summary["full_path_count"], int)
            and cpu_summary["full_path_count"] >= MINIMUM_FULL_PATH_COUNT
            and isinstance(hybrid_summary["full_path_count"], int)
            and hybrid_summary["full_path_count"] >= MINIMUM_FULL_PATH_COUNT
        ),
        "no_worker_errors": (
            cpu_summary["error_count"] == 0 and hybrid_summary["error_count"] == 0
        ),
        "provider_initialization_safe": comparison.get("provider_initialization_safe") is True,
        "semantic_parity": (
            parity.get("evaluated") is True and parity.get("semantic_mismatch_count") == 0
        ),
        "confidence_parity": (
            parity.get("confidence_vector_mismatch_count") == 0
            and maximum_confidence_delta is not None
            and maximum_confidence_delta <= MAXIMUM_CONFIDENCE_DELTA
        ),
        "reported_latency_flags_consistent": (
            comparison.get("cpu_only_target_met") is cpu_target_met
            and comparison.get("gpu_embedder_target_met") is hybrid_target_met
            and comparison.get("target_met_by_any_profile") is (cpu_target_met or hybrid_target_met)
        ),
        "at_least_one_profile_within_1000ms": cpu_target_met or hybrid_target_met,
        "selected_provider_within_1000ms": selected_profile is not None and selected_target_met,
        "privacy_safe": (
            privacy.get("image_paths_recorded") is False
            and privacy.get("image_bytes_recorded") is False
            and privacy.get("image_sha256_recorded") is True
            and not _contains_forbidden_image_payload(payload)
        ),
        "image_hash_provenance_complete": (
            isinstance(sample_count, int)
            and _valid_sha256_list(input_evidence.get("image_sha256"), sample_count)
        ),
    }
    failed_criteria = [name for name, passed in criteria.items() if not passed]
    return {
        "schema_version": "1.0",
        "evaluation": "yolo_free_0.1.7_n100_result_assessment",
        "source_report": {
            "filename": report_path.name,
            "sha256": sha256_file(report_path),
            "self_assessment_passed": payload.get("passes") is True,
            "confidence_tolerance": confidence_tolerance,
        },
        "acceptance": {
            "minimum_image_count": MINIMUM_IMAGE_COUNT,
            "minimum_full_path_count_per_profile": MINIMUM_FULL_PATH_COUNT,
            "maximum_full_path_mean_ms": MAXIMUM_FULL_PATH_LATENCY_MS,
            "maximum_full_path_p95_ms": MAXIMUM_FULL_PATH_LATENCY_MS,
            "maximum_confidence_delta": MAXIMUM_CONFIDENCE_DELTA,
        },
        "passed": not failed_criteria,
        "failed_criteria": failed_criteria,
        "criteria": criteria,
        "hardware": {
            "cpu_name": hardware.get("cpu_name"),
            "video_controllers": hardware.get("video_controllers"),
        },
        "profiles": {
            CPU_PROFILE: cpu_summary | {"target_met": cpu_target_met},
            HYBRID_PROFILE: hybrid_summary | {"target_met": hybrid_target_met},
        },
        "selected_provider": selected_provider,
        "parity": {
            "semantic_mismatch_count": parity.get("semantic_mismatch_count"),
            "confidence_vector_mismatch_count": parity.get("confidence_vector_mismatch_count"),
            "maximum_confidence_delta": maximum_confidence_delta,
            "confidence_tolerance": confidence_tolerance,
        },
        "limitation": (
            "Diagnostic assessment only; not an SLA, certification, or deployment approval."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assess a YOLO-free 0.1.7 N100 result")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    assessment = assess_yolo_free_n100_result(args.report)
    rendered = json.dumps(assessment, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not assessment["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
