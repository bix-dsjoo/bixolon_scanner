from __future__ import annotations

import numpy as np
import pytest

from bixolon_scanner.experiments.bread.classifier_domain_oof import (
    _apply_margin_thresholds,
    assert_group_fold_isolation,
    classification_metrics,
    selective_margin_policy,
    unique_class_predictions,
)


def test_unique_class_predictions_resolve_duplicate_top1_by_global_score() -> None:
    scores = np.asarray(
        [
            [5.0, 4.0, 0.0],
            [4.9, 1.0, 0.0],
            [0.0, 0.1, 6.0],
        ]
    )
    predictions = unique_class_predictions(scores, np.asarray([7, 7, 7]))
    assert predictions.tolist() == [1, 0, 2]


def test_unique_class_predictions_reject_more_rois_than_classes() -> None:
    with pytest.raises(ValueError, match="more ROIs than classes"):
        unique_class_predictions(np.ones((3, 2)), np.asarray([1, 1, 1]))


def test_classification_metrics_report_top1_and_top3() -> None:
    scores = np.asarray([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    metrics = classification_metrics(
        scores,
        np.asarray([0, 3]),
        np.asarray([1, 2]),
        unique_classes=False,
    )
    assert metrics["top1_error_count"] == 1
    assert metrics["top3_miss_count"] == 1


def test_group_fold_isolation_rejects_cross_fold_group() -> None:
    rows = [
        {"group_id": "group-a", "fold": 0},
        {"group_id": "group-a", "fold": 1},
    ]
    with pytest.raises(ValueError, match="group-aware fold overlap"):
        assert_group_fold_isolation(rows)


def test_selective_margin_policy_rejects_high_margin_error_by_predicted_class() -> None:
    scores = np.asarray(
        [
            [10.0, 0.0, -1.0],
            [9.0, 0.0, -1.0],
            [8.0, 7.0, -1.0],
            [0.0, 10.0, 1.0],
        ]
    )
    targets = np.asarray([0, 2, 0, 1])
    policy = selective_margin_policy(
        scores,
        targets,
        maximum_approved_errors=0,
        maximum_unknown_top3_misses=1,
    )
    rejected = _apply_margin_thresholds(scores, policy["thresholds"])
    assert rejected.tolist() == [False, True, True, False]
    assert policy["approved_error_count"] == 0
    assert policy["unknown_top3_miss_count"] == 0


def test_selective_margin_policy_can_leave_one_error_approved() -> None:
    scores = np.asarray([[4.0, 3.0], [3.0, 2.0], [2.0, 4.0]])
    policy = selective_margin_policy(
        scores,
        np.asarray([1, 0, 1]),
        maximum_approved_errors=1,
        maximum_unknown_top3_misses=0,
    )
    assert policy["approved_count"] == 3
    assert policy["approved_error_count"] == 1
