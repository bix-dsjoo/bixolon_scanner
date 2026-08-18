from bixolon_scanner.experiments.bread.detector_disagreement_gate import (
    has_perfect_iou_matching,
    prediction_rows_agree,
)
from bixolon_scanner.pipeline.ports import Detection


def test_perfect_iou_matching_is_one_to_one() -> None:
    primary = [
        Detection(0, 0, 10, 10, 0.9),
        Detection(1, 1, 9, 9, 0.8),
    ]
    recovery = [Detection(0, 0, 10, 10, 0.9), Detection(20, 20, 30, 30, 0.7)]

    assert not has_perfect_iou_matching(primary, recovery, iou_threshold=0.5)


def test_prediction_rows_require_equal_count_and_geometry() -> None:
    primary = {
        "image_id": 1,
        "boxes_xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
        "scores": [0.9, 0.8],
    }
    recovery = {
        "image_id": 1,
        "boxes_xyxy": [[1, 1, 11, 11], [20, 20, 30, 30]],
        "scores": [0.7, 0.6],
    }

    assert prediction_rows_agree(primary, recovery, iou_threshold=0.5)
    recovery["boxes_xyxy"].pop()
    recovery["scores"].pop()
    assert not prediction_rows_agree(primary, recovery, iou_threshold=0.5)
