from bixolon_scanner.experiments.bread.multiscale_detector import (
    consensus_filter_predictions,
)


def test_consensus_filter_keeps_high_singletons_and_agreed_low_boxes():
    primary = [
        {
            "image_id": 1,
            "boxes_xyxy": [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ],
            "scores": [0.9, 0.4, 0.3],
        }
    ]
    recovery = [
        {
            "image_id": 1,
            "boxes_xyxy": [[20.0, 20.0, 30.0, 30.0]],
            "scores": [0.8],
        }
    ]

    result = consensus_filter_predictions(
        primary,
        recovery,
        primary_threshold=0.2,
        primary_singleton_threshold=0.8,
        recovery_threshold=0.5,
        agreement_iou_threshold=0.5,
        nms_iou_threshold=0.5,
    )

    assert result[0]["scores"] == [0.9, 0.4]
