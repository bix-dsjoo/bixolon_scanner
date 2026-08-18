import numpy as np

from bixolon_scanner.experiments.bread.proposal_group_verifier import (
    filter_group_predictions,
    group_content_relation_features,
    group_geometry_features,
    merged_group_labels,
)


def test_merged_group_label_requires_two_covered_targets_and_no_iou_match():
    record = {
        "annotations": [
            {"bbox_xywh": [0, 0, 10, 10]},
            {"bbox_xywh": [12, 0, 10, 10]},
        ]
    }
    boxes = np.asarray([[0, 0, 22, 10], [0, 0, 10, 10]], dtype=np.float32)

    assert merged_group_labels(record, boxes).tolist() == [1, 0]


def test_group_geometry_features_capture_inner_candidate_structure():
    record = {"width": 100, "height": 100}
    prediction = {
        "boxes_xyxy": [[0, 0, 30, 10], [0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8, 0.7],
        "class_ids": [0, 1, 2],
        "source_ids": [1, 0, 0],
    }

    features = group_geometry_features(record, prediction)

    assert features.shape == (3, 24)
    assert features[0, 11] == 2


def test_group_content_relations_measure_inner_class_diversity():
    prediction = {
        "boxes_xyxy": [[0, 0, 30, 10], [0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8, 0.7],
        "class_ids": [0, 0, 0],
    }
    logits = np.asarray([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]], dtype=np.float32)

    features = group_content_relation_features(prediction, logits)

    assert features.shape == (3, 13)
    assert features[0, 2] == 2
    assert features[0, 3] == 1


def test_group_filter_keeps_aligned_source_provenance():
    prediction = {
        "boxes_xyxy": [[0, 0, 10, 10], [20, 0, 30, 10]],
        "scores": [0.9, 0.8],
        "class_ids": [1, 2],
        "source_ids": [0, 1],
    }

    filtered = filter_group_predictions(
        [prediction],
        [np.asarray([0.9, 0.1])],
        group_threshold=0.5,
    )

    assert filtered[0]["scores"] == [0.8]
    assert filtered[0]["source_ids"] == [1]
