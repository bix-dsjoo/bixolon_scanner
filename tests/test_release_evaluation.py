import numpy as np

from bixolon_scanner.evaluation.release import (
    Counts,
    _metrics,
    average_precision,
    minimum_zero_error_samples,
    nearest_iou,
)


def test_release_metrics_separate_classifier_errors_from_false_segments():
    metrics = _metrics(
        Counts(
            approved=100,
            approved_correct=97,
            approved_wrong=3,
            approved_classification_wrong=1,
            approved_false_segmentation=2,
        )
    )

    assert metrics["approved_misrecognition_count"] == 3
    assert metrics["approved_classification_misrecognition_count"] == 1
    assert metrics["approved_false_segmentation_count"] == 2


def test_release_recognition_counts_approved_top1_and_unknown_top3():
    metrics = _metrics(
        Counts(
            ground_truth=10,
            classified_matched=10,
            top1_correct=8,
            recognized_correct=9,
        )
    )

    assert metrics["recognition_accuracy_all_ground_truth"] == 0.9
    assert metrics["classifier_top1_accuracy_on_resolved_matches"] == 0.8


def test_average_precision_penalizes_false_positive_and_missed_ground_truth():
    assert average_precision([(0.9, True), (0.8, False)], ground_truth=2) == 0.5


def test_point_one_percent_risk_requires_2995_zero_error_samples_at_95_percent():
    assert minimum_zero_error_samples(0.001) == 2995


def test_nearest_iou_distinguishes_duplicate_like_predictions_from_background():
    ground_truth = [np.asarray([0, 0, 10, 10], dtype=np.float32)]

    assert nearest_iou(np.asarray([1, 1, 11, 11], dtype=np.float32), ground_truth) > 0.6
    assert nearest_iou(np.asarray([20, 20, 30, 30], dtype=np.float32), ground_truth) == 0.0
    assert nearest_iou(np.asarray([0, 0, 1, 1], dtype=np.float32), []) == 0.0
