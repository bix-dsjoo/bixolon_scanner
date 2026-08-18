from bixolon_scanner.experiments.bread.detector_geometry_fusion import (
    replace_with_recovery_geometry,
)


def test_geometry_fusion_uses_highest_score_agreed_recovery_box():
    primary = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0.0, 0.0, 100.0, 100.0]],
            "scores": [0.8],
            "class_ids": [2],
        }
    ]
    recovery = [
        {
            "image_id": 1,
            "boxes_xyxy": [
                [0.0, 0.0, 95.0, 95.0],
                [20.0, 20.0, 100.0, 100.0],
            ],
            "scores": [0.4, 0.9],
            "class_ids": [2, 2],
        }
    ]

    result = replace_with_recovery_geometry(
        primary,
        recovery,
        recovery_score_threshold=0.1,
        agreement_iou_threshold=0.3,
        recovery_nms_threshold=0.95,
        require_same_class=True,
    )

    assert result[0]["boxes_xyxy"] == [[20.0, 20.0, 100.0, 100.0]]
    assert result[0]["scores"] == [0.8]


def test_geometry_fusion_can_require_matching_classes():
    primary = [
        {
            "image_id": 1,
            "boxes_xyxy": [[0.0, 0.0, 100.0, 100.0]],
            "scores": [0.8],
            "class_ids": [2],
        }
    ]
    recovery = [
        {
            "image_id": 1,
            "boxes_xyxy": [[10.0, 10.0, 90.0, 90.0]],
            "scores": [0.9],
            "class_ids": [3],
        }
    ]

    result = replace_with_recovery_geometry(
        primary,
        recovery,
        recovery_score_threshold=0.1,
        agreement_iou_threshold=0.3,
        recovery_nms_threshold=0.5,
        require_same_class=True,
    )

    assert result[0]["boxes_xyxy"] == primary[0]["boxes_xyxy"]
