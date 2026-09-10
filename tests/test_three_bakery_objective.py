from copy import deepcopy
from pathlib import Path

import pytest

from bixolon_scanner.evaluation.three_bakery_objective import (
    apply_final_objective,
    load_objective,
    object_accuracy,
    rank_candidate,
    required_correct_count,
)


@pytest.fixture
def objective():
    return load_objective(Path("configs/experiments/bread/three_bakery_objective.json"))


@pytest.mark.parametrize("total,minimum", [(649, 643), (1410, 1396), (100, 99)])
def test_object_threshold_uses_ceiling_without_float_rounding(total, minimum):
    assert required_correct_count(total, 0.99) == minimum


@pytest.mark.parametrize(
    "correct,wrong,passed", [(1395, 0, False), (1396, 0, True), (1410, 1, False)]
)
def test_final_object_accuracy_and_zero_false_approvals(objective, correct, wrong, passed):
    summary = {
        "ground_truth_count": 1410,
        "correct_approved_count": correct,
        "wrong_approved_count": wrong,
        "complete_images": 0,
    }
    result = object_accuracy(summary, objective, final=True)
    assert result["accuracy_target_met"] is passed
    assert result["minimum_correct_approved_count"] == 1396


def test_missing_and_recaptured_objects_cannot_be_removed_from_denominator(objective):
    with pytest.raises(ValueError, match="denominator"):
        object_accuracy(
            {"ground_truth_count": 1396, "correct_approved_count": 1396}, objective, final=True
        )
    result = object_accuracy(
        {"ground_truth_count": 1410, "correct_approved_count": 0, "wrong_approved_count": 0},
        objective,
        final=True,
    )
    assert result["correct_approved_rate"] == 0
    assert result["accuracy_target_met"] is False


def test_each_repetition_must_meet_object_and_existing_speed_targets(objective):
    good = {
        "ground_truth_count": 1410,
        "correct_approved_count": 1396,
        "wrong_approved_count": 0,
        "speed_target_met": True,
    }
    measured = {"repetitions": [good, {**good, "speed_target_met": False}, good]}
    before = deepcopy(measured)
    result = apply_final_objective(measured, objective)
    assert result["target_met_all_repetitions"] is False
    assert result["repetitions"][0]["target_met"] is True
    assert result["repetitions"][1]["target_met"] is False
    assert measured == before


def test_object_success_can_qualify_97_complete_images_but_synthetic_false_approval_cannot(
    objective,
):
    real = {
        "ground_truth_count": 649,
        "correct_approved_count": 645,
        "wrong_approved_count": 0,
        "image_status_counts": {},
        "complete_images": 297,
        "by_difficulty": {"multi": {"complete_images": 97}},
        "missed_count": 1,
        "unknown_top3_miss_count": 0,
    }
    stress = {**real, "ground_truth_count": 1446, "correct_approved_count": 977}
    measured = {
        "diagnostics": {"real": real, "stress": stress},
        "rank": [1, 0, -97, -128, 449, 0, 159.3],
        "cpu_latency": {"target_met": True, "worst_full_path_p95_ms": 159.3},
    }
    result = rank_candidate(measured, objective)
    assert result["rank"][0] == 0
    assert result["rank"][2:4] == (-645, -977)
    measured["diagnostics"]["stress"]["wrong_approved_count"] = 1
    assert rank_candidate(measured, objective)["rank"][0] == 1
