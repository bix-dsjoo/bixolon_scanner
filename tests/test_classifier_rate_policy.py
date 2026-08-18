from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.experiments.bread.classifier_rate_policy import (
    _class_conditional_thresholds,
    _official_classifier_gates,
    unanimous_top1_confirmation,
    validate_group_folds,
)
from bixolon_scanner.experiments.bread.zero_error_classifier import Policy


def _policy() -> Policy:
    return Policy(
        name="fixture",
        predictions=np.asarray([0, 0, 0, 1]),
        approval_score=np.asarray([0.7, 0.8, 0.9, 0.6], dtype=np.float32),
        top3=np.asarray([[0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 0, 2]]),
        top3_safety_score=np.ones(4, dtype=np.float32),
    )


def test_class_thresholds_use_predicted_class_and_guard_failures() -> None:
    thresholds = _class_conditional_thresholds(
        _policy(),
        np.asarray([0, 1, 0, 1]),
        np.ones(4, dtype=bool),
        class_count=2,
        guard_samples=1,
        no_failure_threshold=0.55,
    )

    assert thresholds[0] == pytest.approx(0.9)
    assert thresholds[1] == pytest.approx(0.55)


def test_official_classifier_gate_uses_inclusive_point_rates() -> None:
    gates = _official_classifier_gates(
        {
            "approved_rate": 0.90,
            "approved_misrecognition_rate": 0.001,
            "unknown_top3_candidate_out_rate": 0.001,
        },
        minimum_approved_rate=0.90,
        maximum_approved_misrecognition_rate=0.001,
        maximum_unknown_top3_candidate_out_rate=0.001,
    )

    assert gates == {
        "approved_rate": True,
        "approved_misrecognition_rate": True,
        "unknown_top3_candidate_out_rate": True,
        "all_met": True,
    }


def test_group_folds_reject_leakage() -> None:
    rows = [
        {"group_id": "same", "fold": 0},
        {"group_id": "same", "fold": 1},
    ]

    with pytest.raises(ValueError, match="group-aware folds overlap"):
        validate_group_folds(rows)


def test_group_folds_accept_each_group_in_one_fold() -> None:
    rows = [
        {"group_id": "a", "fold": 0},
        {"group_id": "a", "fold": 0},
        {"group_id": "b", "fold": 1},
    ]

    assert validate_group_folds(rows) == 0


def test_unanimous_confirmation_requires_every_view_to_match() -> None:
    confirmed = unanimous_top1_confirmation(
        np.asarray([0, 1, 2]),
        np.asarray([[0, 0, 0], [1, 0, 1], [2, 2, 2]]),
    )

    assert confirmed.tolist() == [True, False, True]
