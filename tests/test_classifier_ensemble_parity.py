from __future__ import annotations

import numpy as np

from bixolon_scanner.experiments.bread.classifier_ensemble_parity import (
    _decision_parity,
    _model_parity,
    final_policy_decisions,
)


def test_final_policy_confirmation_turns_disagreement_into_unknown() -> None:
    left = np.asarray([[5.0, 1.0, 0.0, -1.0], [1.0, 5.0, 0.0, -1.0]], dtype=np.float32)
    right = left.copy()
    confirmer = np.asarray([[1.0, 5.0, 0.0, -1.0], [1.0, 5.0, 0.0, -1.0]], dtype=np.float32)

    result = final_policy_decisions(
        (left, right),
        [confirmer],
        base_view_names=("left", "right"),
        first_view_weight=0.5,
        ranking_tie_break_bias_span=0.0,
        approval_thresholds=np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        top3_safety_threshold=-10.0,
    )

    assert result["approved"].tolist() == [False, True]
    assert result["unknown"].tolist() == [True, False]
    assert result["confirmation"].tolist() == [False, True]


def test_model_parity_reports_row_mismatch_counts() -> None:
    expected = np.asarray([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    actual = np.asarray([[4.0, 3.0, 1.0, 2.0], [4.0, 3.0, 2.0, 1.0]])

    result = _model_parity(actual, expected)

    assert result["top1"]["equal"] is True
    assert result["top1"]["mismatch_count"] == 0
    assert result["top3"]["equal"] is False
    assert result["top3"]["mismatch_count"] == 1
    assert result["top3"]["mismatch_indices"] == [0]


def test_decision_parity_maps_reference_field_names() -> None:
    actual = {
        "predictions": np.asarray([1, 2]),
        "top3": np.asarray([[1, 2, 0], [2, 1, 0]]),
        "approved": np.asarray([True, False]),
        "unknown": np.asarray([False, True]),
        "confirmation": np.asarray([True, True]),
    }
    expected = {
        "predictions": np.asarray([1, 2]),
        "top3": np.asarray([[1, 2, 0], [2, 1, 0]]),
        "final_approved": np.asarray([False, False]),
        "final_unknown": np.asarray([True, True]),
        "confirmation": np.asarray([True, True]),
    }

    result = _decision_parity(
        actual,
        expected,
        expected_names={"approved": "final_approved", "unknown": "final_unknown"},
    )

    assert result["predictions"]["equal"] is True
    assert result["approved"]["mismatch_count"] == 1
    assert result["unknown"]["mismatch_indices"] == [0]
