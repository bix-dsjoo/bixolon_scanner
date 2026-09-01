import numpy as np

from bixolon_scanner.experiments.bread.rfdetr_box_evaluation import (
    _maximum_matched_ground_truth,
    proposal_coverage_metrics,
    select_candidate,
    select_proposal_candidate,
)


def test_maximum_proposal_matching_reassigns_ambiguous_candidate() -> None:
    ious = np.asarray([[0.9, 0.8], [0.7, 0.1]], dtype=np.float32)

    assert _maximum_matched_ground_truth(ious, 0.5) == {0, 1}


def test_rfdetr_box_candidate_minimizes_total_error_then_false_negative():
    candidates = [
        {
            "score_threshold": 0.1,
            "metrics": {
                "false_positive_count": 2,
                "false_negative_count": 1,
                "exact_image_rate": 0.8,
            },
        },
        {
            "score_threshold": 0.2,
            "metrics": {
                "false_positive_count": 1,
                "false_negative_count": 2,
                "exact_image_rate": 0.8,
            },
        },
        {
            "score_threshold": 0.3,
            "metrics": {
                "false_positive_count": 4,
                "false_negative_count": 0,
                "exact_image_rate": 0.9,
            },
        },
    ]

    assert select_candidate(candidates)["score_threshold"] == 0.1


def test_proposal_coverage_ignores_surplus_candidates() -> None:
    records = [
        {
            "image_id": "image-1",
            "annotations": [{"bbox_xywh": [0.0, 0.0, 10.0, 10.0]}],
        }
    ]
    predictions = [
        {
            "image_id": "image-1",
            "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
            "scores": [0.9, 0.8],
            "class_ids": [0, 0],
        }
    ]

    metrics = proposal_coverage_metrics(records, predictions)

    assert metrics["all_ground_truth_covered_image_count"] == 1
    assert metrics["missed_image_count"] == 0
    assert metrics["mean_proposals_per_image"] == 2.0


def test_select_proposal_candidate_prioritizes_zero_misses_over_false_positives() -> None:
    candidates = [
        {
            "name": "low-workload-with-miss",
            "score_threshold": 0.5,
            "metrics": {"false_negative_count": 1, "false_positive_count": 0},
            "proposal_metrics": {
                "false_negative_count": 1,
                "missed_image_count": 1,
                "surplus_proposal_count": 0,
            },
        },
        {
            "name": "broad-zero-miss",
            "score_threshold": 0.01,
            "metrics": {"false_negative_count": 0, "false_positive_count": 20},
            "proposal_metrics": {
                "false_negative_count": 0,
                "missed_image_count": 0,
                "surplus_proposal_count": 20,
            },
        },
    ]

    assert select_proposal_candidate(candidates)["name"] == "broad-zero-miss"
