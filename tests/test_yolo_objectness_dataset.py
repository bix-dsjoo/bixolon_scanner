from __future__ import annotations

import json

from bixolon_scanner.training.yolo_objectness_dataset import export_yolo_objectness_dataset


def test_yolo_export_uses_one_generic_object_class(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "one.jpg").write_bytes(b"one")
    (dataset / "two.jpg").write_bytes(b"two")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "image_id": 1,
            "image_path": "one.jpg",
            "fold": 0,
            "width": 100,
            "height": 200,
            "annotations": [{"category_id": 19, "bbox_xywh": [10, 20, 30, 40]}],
        },
        {
            "image_id": 2,
            "image_path": "two.jpg",
            "fold": 1,
            "width": 100,
            "height": 100,
            "annotations": [],
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = export_yolo_objectness_dataset(
        manifest, dataset, tmp_path / "output", validation_fold=1
    )

    label = (tmp_path / "output" / "labels" / "train" / "000000001.txt").read_text()
    assert label == "0 0.2500000000 0.2000000000 0.3000000000 0.2000000000\n"
    assert (tmp_path / "output" / "labels" / "validation" / "000000002.txt").read_text() == ""
    assert report["class_count"] == 1
    assert report["product_labels_exported"] is False
    assert report["difficulty_metadata_exported"] is False
    validation_manifest = [
        json.loads(line)
        for line in (tmp_path / "output" / "manifests" / "validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert validation_manifest == [
        {
            "image_id": 2,
            "image_path": "two.jpg",
            "width": 100,
            "height": 100,
            "fold": 1,
            "annotations": [],
        }
    ]
