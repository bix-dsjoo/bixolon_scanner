import json

from bixolon_scanner.training.dfine_dataset import export_dfine_coco_splits


def test_dfine_export_preserves_empty_images_and_separates_validation_fold(tmp_path):
    rows = [
        {
            "image_id": 1,
            "image_path": "images/one.jpg",
            "width": 640,
            "height": 640,
            "fold": 0,
            "annotations": [],
        },
        {
            "image_id": 2,
            "image_path": "images/two.jpg",
            "width": 640,
            "height": 640,
            "fold": 2,
            "annotations": [{"category_id": 3, "bbox_xywh": [10, 20, 30, 40], "iscrowd": 0}],
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = export_dfine_coco_splits(manifest, tmp_path / "coco", validation_fold=2)
    training = json.loads((tmp_path / "coco" / "instances_train.json").read_text())
    validation = json.loads((tmp_path / "coco" / "instances_validation.json").read_text())

    assert report["evaluation_images_used"] is False
    assert len(training["images"]) == 1
    assert training["annotations"] == []
    assert validation["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert validation["annotations"][0]["category_id"] == 2
    assert validation["categories"][0]["id"] == 0
    assert validation["categories"][-1]["id"] == 19
    assert len(validation["categories"]) == 20
