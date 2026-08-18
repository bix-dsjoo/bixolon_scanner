from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bixolon_scanner.evaluation.onnx_detector import (
    detector_classification_metrics,
    load_records,
    raw_outputs_to_prediction,
)


def test_explicit_annotation_path_keeps_images_under_dataset_root(tmp_path: Path):
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    annotation = tmp_path / "split" / "validation.json"
    annotation.parent.mkdir()
    annotation.write_text(
        '{"images":[{"id":1,"file_name":"images/sample.jpg"}],"annotations":[]}',
        encoding="utf-8",
    )

    records = load_records(tmp_path, "unused.json", annotation_path=annotation)

    assert records[0]["image_path"] == image.resolve()


def test_load_records_normalizes_zero_based_coco_categories(tmp_path: Path):
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    annotation = tmp_path / "validation.json"
    annotation.write_text(
        """{
          "images":[{"id":1,"file_name":"images/sample.jpg"}],
          "annotations":[{"image_id":1,"bbox":[1,2,3,4],"category_id":0}],
          "categories":[{"id":0,"name":"bread-0"},{"id":1,"name":"bread-1"}]
        }""",
        encoding="utf-8",
    )

    records = load_records(tmp_path, "unused.json", annotation_path=annotation)

    assert records[0]["annotations"][0]["category_id"] == 1


def test_raw_outputs_collapse_classes_and_convert_boxes():
    prediction = raw_outputs_to_prediction(
        np.asarray([[0.0, 2.0], [-2.0, -1.0]], dtype=np.float32),
        np.asarray([[0.5, 0.5, 0.5, 0.25], [0.1, 0.1, 0.4, 0.4]], dtype=np.float32),
        image_width=200,
        image_height=100,
    )

    np.testing.assert_allclose(
        prediction["boxes_xyxy"],
        [[50.0, 37.5, 150.0, 62.5], [0.0, 0.0, 60.0, 30.0]],
    )
    assert prediction["scores"] == pytest.approx([0.880797, 0.268941], rel=1e-5)
    assert prediction["class_ids"] == [1, 1]
    assert prediction["top3_class_ids"] == [[1, 0], [1, 0]]


def test_raw_outputs_reject_misaligned_boxes():
    with pytest.raises(ValueError, match="queries, 4"):
        raw_outputs_to_prediction(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            image_width=100,
            image_height=100,
        )


def test_detector_classification_metrics_match_category_ids():
    report = detector_classification_metrics(
        [{"annotations": [{"bbox_xywh": [10.0, 10.0, 20.0, 20.0], "category_id": 2}]}],
        [
            {
                "boxes_xyxy": [[10.0, 10.0, 30.0, 30.0]],
                "scores": [0.9],
                "class_ids": [1],
                "top3_class_ids": [[1, 0, 2]],
            }
        ],
        score_threshold=0.5,
        nms_iou_threshold=0.7,
        match_iou_threshold=0.5,
        max_object_aspect_ratio=5.0,
    )

    assert report["matched_sample_count"] == 1
    assert report["top1_accuracy"] == 1.0
    assert report["top3_accuracy"] == 1.0
