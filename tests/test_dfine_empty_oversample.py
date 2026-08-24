from __future__ import annotations

import pytest

from bixolon_scanner.training.dfine_empty_oversample import oversample_empty_coco


def test_oversample_empty_coco_repeats_only_empty_images():
    payload = {
        "images": [
            {"id": 1, "file_name": "positive.jpg"},
            {"id": 2, "file_name": "empty.jpg"},
        ],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1]}],
        "categories": [{"id": 0, "name": "bread"}],
    }

    result = oversample_empty_coco(payload, repeats=3)

    assert len(result["images"]) == 4
    assert [row["file_name"] for row in result["images"]].count("empty.jpg") == 3
    assert len({int(row["id"]) for row in result["images"]}) == 4
    assert result["annotations"] == payload["annotations"]
    assert result["empty_image_oversampling"]["unique_image_count"] == 2
    assert result["empty_image_oversampling"]["added_training_record_count"] == 2


def test_oversample_empty_coco_rejects_missing_empty_images():
    with pytest.raises(ValueError, match="no empty images"):
        oversample_empty_coco(
            {
                "images": [{"id": 1, "file_name": "positive.jpg"}],
                "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1]}],
            },
            repeats=2,
        )


def test_oversample_empty_coco_rejects_invalid_repeat_count():
    with pytest.raises(ValueError, match="positive"):
        oversample_empty_coco({"images": [], "annotations": []}, repeats=0)
