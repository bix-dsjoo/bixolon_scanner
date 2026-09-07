from types import SimpleNamespace

import pytest

from bixolon_scanner.evaluation.limited_source_review import record_review, review_measurement
from bixolon_scanner.evaluation.routing_audit import audit_routing
from bixolon_scanner.training.limited_source import write_json


def test_unreachable_verification_and_detail_routes_are_reported():
    metadata = SimpleNamespace(
        classifier_policy=SimpleNamespace(
            ridge_approval_metric="l2_normalized_logit_margin",
            ridge_approval_minimum_margin=0.8,
            ridge_approval_thresholds=None,
        ),
        classifier_verification=SimpleNamespace(
            verify_all_approved_candidates=False, ambiguity_maximum_approval_score=0.55
        ),
        classifier_resolution_fallback=SimpleNamespace(
            approval_disagreement_rules=[
                SimpleNamespace(maximum_approval_score=0.55, require_detector_disagreement=False)
            ]
        ),
        detector_class_mode="class_agnostic",
    )
    result = audit_routing(metadata)
    assert [row["route"] for row in result["issues"]] == [
        "approved_verifier",
        "approved_detail_rule_0",
    ]
    metadata.classifier_verification.ambiguity_maximum_approval_score = 0.85
    metadata.classifier_resolution_fallback.approval_disagreement_rules[
        0
    ].maximum_approval_score = 0.85
    assert audit_routing(metadata)["reachable"]


def _review_inputs():
    source = {
        "detector_manifest_sha256": "locked",
        "multi_count": 20,
        "source_manifest_sha256": "source",
        "evaluation_role": "same_item_diagnostic",
    }
    config = {
        "diagnostic_targets": {
            "maximum_approved_error_count": 0,
            "maximum_false_negative_count": 0,
            "maximum_unknown_top3_miss_count": 0,
            "minimum_correct_approved_rate": 0.99,
            "maximum_cpu_full_path_p95_ms": 250,
        }
    }
    report = {
        "dataset": {"manifest_sha256": "locked", "image_count": 20, "held_out_test_set": False},
        "counts": {
            "approved_misrecognition_count": 1,
            "approved_false_positive_count": 2,
            "false_negative_count": 4,
            "unknown_candidate_out_count": 1,
            "segmentation_image_count": 20,
        },
        "metrics": {"correct_approved_rate": 0.8},
        "performance": {"full_path": {"p95_ms": 300}},
        "environment": {"provider": "cpu"},
    }
    return report, source, config


def test_diagnostics_prioritize_approved_errors_and_never_select_thresholds():
    result = review_measurement(*_review_inputs())
    assert result["diagnostic_rank"] == [3, 4, 1, -0.8, 300]
    assert not result["all_development_targets_met"]
    assert not result["threshold_selection_allowed"]
    assert not result["automatic_model_selection"]
    assert len(result["next_experiments"]) == 5


def test_shared_candidate_history_preserves_stagnation_count(tmp_path, monkeypatch):
    report, source, config = _review_inputs()
    config["iteration"] = {"maximum_candidates": 6, "maximum_consecutive_non_improvements": 2}
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    monkeypatch.setattr(
        "bixolon_scanner.evaluation.limited_source_review.verify_sources", lambda _: source
    )
    history = tmp_path / "history"
    for index in range(3):
        work = tmp_path / f"candidate-{index}"
        write_json(work / "plan.json", {"config": str(config_path)})
        report["performance"]["full_path"]["p95_ms"] = 300 + index
        measurement = work / "evaluation.json"
        write_json(measurement, report)
        result = record_review(work, measurement, history)
        assert result["iteration"] == index + 1
        assert result["consecutive_non_improvements"] == index
        assert result["stop_and_reassess"] is (index == 2)


def test_unselected_images_cannot_enter_the_iteration_loop():
    report, source, config = _review_inputs()
    report["dataset"]["manifest_sha256"] = "another-set"
    with pytest.raises(ValueError, match="different image set"):
        review_measurement(report, source, config)


@pytest.mark.parametrize("latency", [0, None])
def test_abstaining_on_every_image_is_not_success(latency):
    report, source, config = _review_inputs()
    report["counts"] = dict.fromkeys(report["counts"], 0)
    report["metrics"]["correct_approved_rate"] = 0
    report["performance"]["full_path"]["p95_ms"] = latency
    result = review_measurement(report, source, config)
    assert "no_full_path_samples" in result["failed_targets"]
    assert not result["all_development_targets_met"]
