import numpy as np

from bixolon_scanner.experiments.bread.rfdetr_predictions import detection_row


class _Detections:
    xyxy = np.asarray([[1.5, 2.5, 30.5, 40.5]], dtype=np.float32)
    confidence = np.asarray([0.75], dtype=np.float32)
    class_id = np.asarray([19], dtype=np.int64)


def test_rfdetr_prediction_row_matches_detector_evaluation_contract():
    row = detection_row(42, _Detections())

    assert row == {
        "image_id": 42,
        "boxes_xyxy": [[1.5, 2.5, 30.5, 40.5]],
        "scores": [0.75],
        "class_ids": [19],
    }
