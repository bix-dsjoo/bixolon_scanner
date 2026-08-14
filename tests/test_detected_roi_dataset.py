from __future__ import annotations

import numpy as np
from PIL import Image

from bixolon_scanner.evaluation.detected_roi_dataset import (
    crop_tensor,
    match_detections,
    select_detections,
)


def test_select_and_match_detections_uses_score_nms_and_iou() -> None:
    prediction = {
        "boxes_xyxy": [
            [10.0, 10.0, 50.0, 50.0],
            [11.0, 11.0, 49.0, 49.0],
            [60.0, 60.0, 90.0, 90.0],
        ],
        "scores": [0.9, 0.8, 0.2],
    }
    detections = select_detections(
        prediction,
        score_threshold=0.5,
        nms_iou_threshold=0.7,
        maximum_aspect_ratio=5.0,
    )
    matches = match_detections(
        detections,
        [{"bbox_xywh": [10.0, 10.0, 40.0, 40.0], "category_id": 1}],
        match_iou_threshold=0.5,
    )

    assert len(detections) == 1
    assert matches[0][0] == 0
    assert matches[0][1] == 1.0


def test_crop_tensor_matches_classifier_contract() -> None:
    image = Image.new("RGB", (100, 80), color=(127, 128, 129))
    detection = select_detections(
        {"boxes_xyxy": [[20.0, 10.0, 70.0, 60.0]], "scores": [0.9]},
        score_threshold=0.5,
        nms_iou_threshold=0.7,
        maximum_aspect_ratio=5.0,
    )[0]

    tensor = crop_tensor(image, detection, crop_margin_ratio=0.05, input_size=224)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
