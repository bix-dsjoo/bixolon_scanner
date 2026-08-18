from bixolon_scanner.experiments.bread.scale_consensus_detector import (
    scale_consensus_candidates,
    scale_consensus_predictions,
    select_scale_consensus_candidates,
)


def test_scale_consensus_uses_conservative_score_and_recovery_geometry():
    primary = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "scores": [0.9, 0.8],
            "class_ids": [1, 2],
        }
    ]
    recovery = [
        {
            "image_id": 1,
            "boxes_xyxy": [[1, 1, 11, 11], [20, 20, 30, 30]],
            "scores": [0.7, 0.2],
            "class_ids": [1, 3],
        }
    ]

    result = scale_consensus_predictions(
        primary,
        recovery,
        agreement_iou_threshold=0.5,
        consensus_score_threshold=0.1,
        nms_iou_threshold=0.5,
        containment_threshold=0.8,
        group_minimum=2,
    )

    assert result[0]["boxes_xyxy"] == [[1, 1, 11, 11]]
    assert result[0]["scores"] == [0.7]
    assert result[0]["class_ids"] == [1]


def test_scale_consensus_filters_before_hierarchical_suppression():
    primary = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10]],
            "scores": [0.9],
            "class_ids": [1],
        }
    ]
    recovery = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10]],
            "scores": [0.2],
            "class_ids": [1],
        }
    ]

    candidates = scale_consensus_candidates(
        primary,
        recovery,
        agreement_iou_threshold=0.5,
    )
    rejected = select_scale_consensus_candidates(
        candidates,
        consensus_score_threshold=0.21,
        nms_iou_threshold=0.5,
        containment_threshold=0.8,
        group_minimum=2,
    )

    assert candidates[0]["scores"] == [0.2]
    assert rejected[0]["boxes_xyxy"] == []


def test_scale_consensus_can_ignore_detector_class_for_spatial_pairing():
    primary = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0, 0, 10, 10]],
            "scores": [0.8],
            "class_ids": [1],
        }
    ]
    recovery = [
        {
            "image_id": 1,
            "boxes_xyxy": [[1, 1, 11, 11]],
            "scores": [0.7],
            "class_ids": [2],
        }
    ]

    candidates = scale_consensus_candidates(
        primary,
        recovery,
        agreement_iou_threshold=0.5,
        class_match="none",
    )

    assert candidates[0]["scores"] == [0.7]
