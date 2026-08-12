from __future__ import annotations

from bixolon_scanner.training.ten_shot_evaluation import (
    PromotionGates,
    evaluate_promotion_gates,
)


def _inputs():
    candidate = {
            "top1_accuracy": 0.98,
            "overall_top3_accuracy": 0.995,
            "approved_count": 500,
            "approved_precision": 0.998,
            "false_approval_rate_upper_95": 0.004,
            "approval_coverage": 0.94,
            "unknown_top3_accuracy": 0.96,
            "recapture_recall": 0.995,
        }
    return {
        "candidate": candidate,
        "baseline": {"top1_accuracy": 0.99, "approval_coverage": 0.97},
        "parity": {
            "pytorch_cpu_pass": True,
            "pytorch_cuda_pass": True,
            "cross_provider_pass": True,
        },
        "benchmark": {"full_path_p95_ms": 90.0},
        "detector": {"onnx_frozen": True, "metadata_frozen": True},
        "regression_sets": {
            "test_94": dict(candidate),
            "bread_project_2_300": dict(candidate),
        },
        "independent_pilot": {
            "image_count": 100,
            "train_overlap_count": 0,
            "policy_fit_overlap_count": 0,
        },
        "gates": PromotionGates(),
    }


def test_promotion_requires_every_accuracy_safety_parity_and_independence_gate():
    report = evaluate_promotion_gates(**_inputs())
    assert report["passed"] is True
    assert report["promotion_status"] == "production"
    assert report["failures"] == []


def test_zero_approval_coverage_cannot_pass_as_safe():
    values = _inputs()
    values["candidate"]["approved_count"] = 0
    values["candidate"]["approval_coverage"] = 0.0
    report = evaluate_promotion_gates(**values)
    assert report["passed"] is False
    assert "approved_samples_present" in report["failures"]
    assert "approval_coverage_floor" in report["failures"]


def test_existing_non_independent_regression_set_cannot_be_final_pilot():
    values = _inputs()
    values["independent_pilot"]["policy_fit_overlap_count"] = 99
    report = evaluate_promotion_gates(**values)
    assert report["promotion_status"] == "experiment_only"
    assert "independent_pilot_no_policy_fit_overlap" in report["failures"]


def test_underpowered_regression_cannot_claim_half_percent_risk_bound():
    values = _inputs()
    values["regression_sets"]["test_94"][
        "risk_certification_maximum_count"
    ] = 478
    report = evaluate_promotion_gates(**values)

    assert report["promotion_status"] == "experiment_only"
    assert (
        "test_94:approval_risk_certification_feasible" in report["failures"]
    )
