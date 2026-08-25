from pathlib import Path

import numpy as np
import pytest

from bixolon_scanner.training.yolo_objectness_predictions import (
    prediction_row,
    raw_stretch_prediction_row,
)


def test_prediction_row_uses_numeric_image_id_and_one_generic_class() -> None:
    row = prediction_row(
        image_path="C:/dataset/000000042.jpg",
        boxes_xyxy=[[1, 2, 3, 4], [5, 6, 7, 8]],
        scores=[0.9, 0.8],
        end2end=True,
    )

    assert row == {
        "image_id": 42,
        "boxes_xyxy": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        "scores": [0.9, 0.8],
        "class_ids": [0, 0],
        "architecture_contract": "one-class-product-independent-objectness",
        "head": "one-to-one",
    }


def test_prediction_row_rejects_misaligned_values() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        prediction_row(
            image_path="000000001.jpg",
            boxes_xyxy=[[1, 2, 3, 4]],
            scores=[],
            end2end=False,
        )


def test_raw_stretch_prediction_scales_xywh_to_original_pixels() -> None:
    row = raw_stretch_prediction_row(
        image_path=Path("000000007.jpg"),
        raw_output=np.asarray([[320.0], [160.0], [64.0], [32.0], [0.8]]),
        input_size=640,
        original_width=1280,
        original_height=640,
        minimum_score=0.65,
        maximum_detections=300,
    )

    assert row["boxes_xyxy"] == [[576.0, 144.0, 704.0, 176.0]]
    assert row["scores"] == pytest.approx([0.8])
