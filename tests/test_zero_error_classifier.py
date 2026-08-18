import numpy as np

from bixolon_scanner.experiments.bread.classifier_guard_search import (
    exclude_image_records,
)
from bixolon_scanner.experiments.bread.zero_error_classifier import (
    Policy,
    _guarded_threshold,
    policy_metrics,
)


def test_guarded_threshold_excludes_failures_and_requested_safer_samples():
    scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    failures = np.asarray([False, True, False, False])
    calibration = np.ones(4, dtype=bool)

    threshold = _guarded_threshold(scores, failures, calibration, guard_samples=1)

    assert threshold == float(scores[2])


def test_zero_guard_moves_past_float32_failure_value():
    scores = np.asarray([0.2, 0.3], dtype=np.float32)
    failures = np.asarray([True, False])

    threshold = _guarded_threshold(scores, failures, np.ones(2, dtype=bool), guard_samples=0)

    assert threshold > float(scores[0])
    assert not bool(scores[0] >= threshold)


def test_policy_metrics_counts_unsafe_top3_as_segment_recapture():
    policy = Policy(
        name="fixture",
        predictions=np.asarray([0, 1, 2]),
        approval_score=np.asarray([0.9, 0.4, 0.3]),
        top3=np.asarray([[0, 1, 2], [1, 0, 2], [0, 1, 3]]),
        top3_safety_score=np.asarray([0.9, 0.8, 0.1]),
    )
    targets = np.asarray([0, 0, 2])

    metrics = policy_metrics(
        policy,
        targets,
        np.ones(3, dtype=bool),
        approval_threshold=0.8,
        safety_threshold=0.5,
    )

    assert metrics["approved_count"] == 1
    assert metrics["approved_error_count"] == 0
    assert metrics["unknown_count"] == 1
    assert metrics["unknown_top3_miss_count"] == 0
    assert metrics["segment_recapture_count"] == 1


def test_exclude_image_records_removes_every_roi_from_an_image():
    targets = np.asarray([0, 1, 2])
    logits = {"view": np.arange(6).reshape(3, 2)}
    rows = [{"image_id": 10}, {"image_id": 11}, {"image_id": 10}]

    retained_targets, retained_logits, retained_rows = exclude_image_records(
        targets, logits, rows, {10}
    )

    assert retained_targets.tolist() == [1]
    assert retained_logits["view"].tolist() == [[2, 3]]
    assert retained_rows == [{"image_id": 11}]
