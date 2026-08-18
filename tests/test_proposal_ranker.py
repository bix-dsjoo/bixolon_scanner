import numpy as np

from bixolon_scanner.experiments.bread.proposal_ranker import (
    proposal_assignment_labels,
    proposal_features,
    proposal_labels,
    proposal_qualities,
    select_ranked_predictions,
)


def test_proposal_features_are_gt_independent_and_labels_use_iou():
    record = {
        "image_id": 1,
        "width": 100,
        "height": 100,
        "annotations": [{"bbox_xywh": [10, 10, 20, 20]}],
    }
    primary = {
        "image_id": 1,
        "boxes_xyxy": [[10, 10, 30, 30]],
        "scores": [0.8],
        "class_ids": [1],
        "top3_class_ids": [[1, 2, 3]],
    }
    recovery = {
        "image_id": 1,
        "boxes_xyxy": [[11, 11, 31, 31]],
        "scores": [0.7],
        "class_ids": [2],
        "top3_class_ids": [[2, 1, 4]],
    }

    boxes, scores, classes, features = proposal_features(record, primary, recovery)
    labels = proposal_labels(record, boxes)

    assert boxes.shape == (2, 4)
    assert scores.tolist() == np.asarray([0.8, 0.7], dtype=np.float32).tolist()
    assert classes.tolist() == [1, 2]
    assert features.shape == (2, 74)
    assert labels.tolist() == [1, 1]


def test_proposal_qualities_preserve_continuous_iou_for_ranking():
    record = {"annotations": [{"bbox_xywh": [0, 0, 10, 10]}]}
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 5, 10]], dtype=np.float32)

    np.testing.assert_allclose(proposal_qualities(record, boxes), [1.0, 0.5])


def test_proposal_assignment_labels_choose_one_proposal_per_target():
    record = {
        "annotations": [
            {"bbox_xywh": [0, 0, 10, 10]},
            {"bbox_xywh": [20, 0, 10, 10]},
        ]
    }
    boxes = np.asarray(
        [
            [0, 0, 10, 10],
            [1, 0, 11, 10],
            [20, 0, 30, 10],
            [40, 0, 50, 10],
        ],
        dtype=np.float32,
    )

    labels = proposal_assignment_labels(record, boxes)

    assert labels.tolist() == [1, 0, 1, 0]


def test_ranked_prediction_selection_is_class_agnostic():
    ranked = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10], [0, 0, 10, 10]],
            "scores": [0.9, 0.8],
            "class_ids": [1, 2],
        }
    ]

    selected = select_ranked_predictions(
        ranked,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
    )

    assert selected[0]["scores"] == [0.9]


def test_ranked_prediction_selection_can_preserve_overlapping_distinct_classes():
    ranked = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10], [0, 0, 10, 10]],
            "scores": [0.9, 0.8],
            "class_ids": [1, 2],
        }
    ]

    selected = select_ranked_predictions(
        ranked,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
        nms_mode="class_aware",
    )

    assert selected[0]["scores"] == [0.9, 0.8]


def test_center_aware_nms_preserves_overlaps_with_distant_centers():
    ranked = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10], [4, 0, 14, 10], [0.5, 0, 10.5, 10]],
            "scores": [0.9, 0.8, 0.7],
            "class_ids": [1, 1, 1],
        }
    ]

    selected = select_ranked_predictions(
        ranked,
        score_threshold=0.5,
        nms_iou_threshold=0.3,
        nms_mode="center_aware",
        nms_center_distance_threshold=0.2,
    )

    assert selected[0]["scores"] == [0.9, 0.8]


def test_ranked_prediction_selection_removes_group_box_but_keeps_single_outer():
    ranked = [
        {
            "image_id": 1,
            "boxes_xyxy": [
                [0, 0, 30, 10],
                [1, 1, 9, 9],
                [21, 1, 29, 9],
                [40, 0, 60, 20],
                [45, 5, 55, 15],
            ],
            "scores": [0.99, 0.9, 0.8, 0.7, 0.6],
            "class_ids": [1, 2, 3, 4, 5],
        }
    ]

    selected = select_ranked_predictions(
        ranked,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
        containment_threshold=0.8,
        group_minimum=2,
    )

    assert [0.0, 0.0, 30.0, 10.0] not in selected[0]["boxes_xyxy"]
    assert [40.0, 0.0, 60.0, 20.0] in selected[0]["boxes_xyxy"]


def test_group_selection_accepts_an_empty_candidate_set():
    ranked = [{"image_id": 1, "boxes_xyxy": [], "scores": [], "class_ids": []}]

    selected = select_ranked_predictions(
        ranked,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
        group_minimum=2,
    )

    assert selected[0]["boxes_xyxy"] == []
