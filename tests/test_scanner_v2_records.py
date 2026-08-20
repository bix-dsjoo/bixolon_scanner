from __future__ import annotations

import json

import pytest

from bixolon_scanner.evaluation.scanner_v2 import _records


def test_records_loads_coco_json(tmp_path) -> None:
    image = tmp_path / "images" / "one.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "images/one.jpg"}],
                "annotations": [{"image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )

    rows = _records(annotation, tmp_path)

    assert rows == [
        {
            "id": 1,
            "file_name": "images/one.jpg",
            "image_id": 1,
            "image_path": "images/one.jpg",
            "annotations": [{"bbox_xywh": [1.0, 2.0, 3.0, 4.0], "category_id": 7}],
            "resolved_path": image,
        }
    ]


def test_records_rejects_coco_path_escape(tmp_path) -> None:
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "../outside.jpg"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escaped"):
        _records(annotation, tmp_path)
