import numpy as np

from bixolon_scanner.experiments.bread.proposal_pairwise_nms import (
    distinct_object_targets,
    pairwise_features,
    pairwise_select,
)


def test_pairwise_targets_keep_only_different_matched_objects():
    record = {
        "annotations": [
            {"bbox_xywh": [0, 0, 10, 10]},
            {"bbox_xywh": [5, 0, 10, 10]},
        ]
    }
    boxes = np.asarray([[0, 0, 10, 10], [5, 0, 15, 10], [0, 0, 10, 10]])
    pairs = np.asarray([[0, 1], [0, 2]])

    assert distinct_object_targets(record, boxes, pairs).tolist() == [1, 0]


def test_pairwise_features_and_selection_preserve_likely_distinct_overlap():
    record = {
        "width": 20,
        "height": 10,
        "annotations": [],
    }
    prediction = {
        "image_id": 1,
        "boxes_xyxy": [[0, 0, 10, 10], [5, 0, 15, 10]],
        "scores": [0.9, 0.8],
        "class_ids": [1, 2],
    }
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    logits = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    pairs, features = pairwise_features(
        record,
        prediction,
        embeddings,
        embeddings,
        logits,
        minimum_iou=0.3,
    )
    selected = pairwise_select(
        prediction,
        {(0, 1): 0.9},
        score_threshold=0.5,
        nms_iou_threshold=0.3,
        distinct_threshold=0.5,
    )

    assert pairs.tolist() == [[0, 1]]
    assert features.shape == (1, 16)
    assert selected["scores"] == [0.9, 0.8]
