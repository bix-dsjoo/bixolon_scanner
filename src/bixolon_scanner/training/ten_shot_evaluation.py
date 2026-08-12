from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionGates:
    target_top1_accuracy: float = 0.97
    minimum_top1_accuracy: float = 0.95
    minimum_overall_top3_accuracy: float = 0.95
    minimum_approved_precision: float = 0.995
    maximum_false_approval_rate_upper_95: float = 0.005
    target_maximum_approval_coverage_drop_percentage_points: float = 5.0
    minimum_approval_coverage: float = 0.85
    minimum_unknown_top3_accuracy: float = 0.95
    minimum_recapture_recall: float = 0.99
    maximum_full_path_p95_ms: float = 100.0

    def validate(self) -> None:
        rates = (
            self.target_top1_accuracy,
            self.minimum_top1_accuracy,
            self.minimum_overall_top3_accuracy,
            self.minimum_approved_precision,
            self.maximum_false_approval_rate_upper_95,
            self.minimum_approval_coverage,
            self.minimum_unknown_top3_accuracy,
            self.minimum_recapture_recall,
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("promotion rate gates must be between zero and one")
        if self.target_top1_accuracy < self.minimum_top1_accuracy:
            raise ValueError("target Top-1 cannot be below the waiver floor")
        if self.target_maximum_approval_coverage_drop_percentage_points < 0:
            raise ValueError("coverage drop gate must be non-negative")
        if self.maximum_full_path_p95_ms <= 0:
            raise ValueError("latency gate must be positive")


def _metric(source: dict[str, Any], key: str) -> float:
    if source.get(key) is None:
        raise ValueError(f"promotion input is missing {key}")
    return float(source[key])


def _zero_error_false_approval_upper(count: int, confidence_level: float = 0.95) -> float:
    if count <= 0:
        return 1.0
    return 1.0 - (1.0 - confidence_level) ** (1.0 / count)


def _floor_checks(metrics: dict[str, Any], gates: PromotionGates) -> dict[str, bool]:
    checks = {
        "top1_floor": _metric(metrics, "top1_accuracy") >= gates.minimum_top1_accuracy,
        "overall_top3_floor": (
            _metric(metrics, "overall_top3_accuracy")
            >= gates.minimum_overall_top3_accuracy
        ),
        "approved_samples_present": int(metrics.get("approved_count", 0)) > 0,
        "approved_precision_floor": (
            _metric(metrics, "approved_precision") >= gates.minimum_approved_precision
        ),
        "approval_risk_upper_bound": (
            _metric(metrics, "false_approval_rate_upper_95")
            <= gates.maximum_false_approval_rate_upper_95
        ),
        "approval_coverage_floor": (
            _metric(metrics, "approval_coverage") >= gates.minimum_approval_coverage
        ),
        "recapture_recall": (
            _metric(metrics, "recapture_recall") >= gates.minimum_recapture_recall
        ),
    }
    maximum_count = metrics.get("risk_certification_maximum_count")
    if maximum_count is not None:
        checks["approval_risk_certification_feasible"] = (
            _zero_error_false_approval_upper(int(maximum_count))
            <= gates.maximum_false_approval_rate_upper_95
        )
    return checks


def evaluate_promotion_gates(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    parity: dict[str, Any],
    benchmark: dict[str, Any],
    detector: dict[str, Any],
    gates: PromotionGates,
    regression_sets: dict[str, dict[str, Any]] | None = None,
    independent_pilot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return production, manual_waiver, or experiment_only without tuning on test."""
    gates.validate()
    coverage_drop = (
        _metric(baseline, "approval_coverage") - _metric(candidate, "approval_coverage")
    ) * 100.0
    common = {
        "detector_onnx_frozen": bool(detector.get("onnx_frozen")),
        "detector_metadata_frozen": bool(detector.get("metadata_frozen")),
        "pytorch_cpu_parity": bool(parity.get("pytorch_cpu_pass")),
        "pytorch_cuda_parity": bool(parity.get("pytorch_cuda_pass")),
        "cross_provider_parity": bool(parity.get("cross_provider_pass")),
        "full_path_latency": (
            _metric(benchmark, "full_path_p95_ms") <= gates.maximum_full_path_p95_ms
        ),
    }
    candidate_floor = _floor_checks(candidate, gates)
    regression_checks: dict[str, bool] = {}
    required_regressions = {"test_94", "bread_project_2_300"}
    available_regressions = set((regression_sets or {}))
    regression_checks["required_regression_sets_present"] = (
        required_regressions <= available_regressions
    )
    for name, metrics in sorted((regression_sets or {}).items()):
        for check, passed in _floor_checks(metrics, gates).items():
            regression_checks[f"{name}:{check}"] = passed
    # Compatibility-only provenance: these legacy sets are explicitly not
    # independent. Independence is recorded, not used to pretend they are a pilot.
    if independent_pilot is not None:
        regression_checks["independent_pilot_has_samples"] = (
            int(independent_pilot.get("image_count", 0)) > 0
        )
        regression_checks["independent_pilot_no_train_overlap"] = (
            int(independent_pilot.get("train_overlap_count", -1)) == 0
        )
        regression_checks["independent_pilot_no_policy_fit_overlap"] = (
            int(independent_pilot.get("policy_fit_overlap_count", -1)) == 0
        )
    waiver_checks = {**common, **candidate_floor, **regression_checks}
    production_checks = {
        **waiver_checks,
        "top1_target": _metric(candidate, "top1_accuracy") >= gates.target_top1_accuracy,
        "overall_top3_target": (
            _metric(candidate, "overall_top3_accuracy")
            >= gates.minimum_overall_top3_accuracy
        ),
        "approved_precision_target": (
            _metric(candidate, "approved_precision") >= gates.minimum_approved_precision
        ),
        "coverage_non_regression": (
            coverage_drop
            <= gates.target_maximum_approval_coverage_drop_percentage_points
        ),
        "unknown_top3_target": (
            _metric(candidate, "unknown_top3_accuracy")
            >= gates.minimum_unknown_top3_accuracy
        ),
    }
    if all(production_checks.values()):
        status = "production"
        active = production_checks
    elif all(waiver_checks.values()):
        status = "manual_waiver"
        active = waiver_checks
    else:
        status = "experiment_only"
        active = waiver_checks
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "promotion_status": status,
        "passed": status != "experiment_only",
        "manual_approval_required": status == "manual_waiver",
        "checks": active,
        "production_checks": production_checks,
        "waiver_checks": waiver_checks,
        "failures": [name for name, value in active.items() if not value],
        "observed": {
            "top1_accuracy": _metric(candidate, "top1_accuracy"),
            "overall_top3_accuracy": _metric(candidate, "overall_top3_accuracy"),
            "approved_precision": _metric(candidate, "approved_precision"),
            "approval_coverage": _metric(candidate, "approval_coverage"),
            "approval_coverage_drop_percentage_points": coverage_drop,
            "unknown_top3_accuracy": _metric(candidate, "unknown_top3_accuracy"),
            "recapture_recall": _metric(candidate, "recapture_recall"),
            "full_path_p95_ms": _metric(benchmark, "full_path_p95_ms"),
        },
        "gates": asdict(gates),
        "evaluation_limitations": {
            "regression_sets_are_independent": False,
            "test_and_bread_project_2_are_regression_sets_only": True,
        },
    }


def write_promotion_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
