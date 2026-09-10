from copy import deepcopy

import pytest

from bixolon_scanner.evaluation.improvement_selection import assess_model_change


def summary(correct=90, wrong=2, p95=200):
    return {
        "image_count": 20,
        "ground_truth_count": 100,
        "correct_approved_count": correct,
        "wrong_approved_count": wrong,
        "image_status_counts": {"SEGMENTATION": 20},
        "latency": {k: {"p95_ms": p95} for k in ("all", "full_path")},
    }


def inputs():
    report = {"repetitions": [summary()], "summary": summary()}
    policy = {
        "minimum_correct_approval_gain": 11,
        "alternative_minimum_wrong_approval_reduction": 1,
        "maximum_cpu_p95_ratio": 1.05,
        "maximum_cpu_p95_ms_if_baseline_passes": 300,
    }
    return report, deepcopy(report), deepcopy(report), deepcopy(report), policy


def test_fewer_false_approvals_cannot_hide_lost_correct_objects():
    old, new, source_old, source_new, policy = inputs()
    new["repetitions"] = [summary(correct=89, wrong=0)]
    result = assess_model_change(old, new, source_old, source_new, policy)
    assert not result["accepted"]
    assert "repeat_1:correct_approvals_decreased" in result["reasons"]


def test_new_source_mistake_rejects_improvement_on_untrained_images():
    old, new, source_old, source_new, policy = inputs()
    new["repetitions"] = [summary(correct=92, wrong=1)]
    source_new["summary"] = summary(correct=89, wrong=3)
    result = assess_model_change(old, new, source_old, source_new, policy)
    assert not result["accepted"]
    assert "source_wrong_approvals_increased" in result["reasons"]


@pytest.mark.parametrize("latency,accepted", [(210, True), (210.01, False), (float("nan"), False)])
def test_speed_boundary_cannot_silently_accept_regression(latency, accepted):
    old, new, source_old, source_new, policy = inputs()
    new["repetitions"] = [summary(correct=92, wrong=1, p95=latency)]
    assert assess_model_change(old, new, source_old, source_new, policy)["accepted"] is accepted


def test_changed_denominator_is_an_error():
    old, new, source_old, source_new, policy = inputs()
    new["repetitions"][0]["ground_truth_count"] = 99
    with pytest.raises(ValueError, match="denominators"):
        assess_model_change(old, new, source_old, source_new, policy)
