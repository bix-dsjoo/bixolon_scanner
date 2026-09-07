import json

import pytest

from bixolon_scanner.training.dfine_dataset import (
    export_dfine_coco_source_sets,
    export_dfine_coco_splits,
)


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
    all_records = json.loads((tmp_path / "coco" / "instances_all.json").read_text())

    assert report["evaluation_images_used"] is False
    assert len(training["images"]) == 1
    assert training["annotations"] == []
    assert validation["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert validation["annotations"][0]["category_id"] == 2
    assert validation["categories"][0]["id"] == 0
    assert validation["categories"][-1]["id"] == 19
    assert len(validation["categories"]) == 20
    assert len(all_records["images"]) == 2


def test_dfine_export_can_collapse_bread_labels_for_detection_only(tmp_path):
    rows = [
        {
            "image_id": 1,
            "image_path": "images/one.jpg",
            "width": 640,
            "height": 640,
            "fold": 0,
            "annotations": [{"category_id": 20, "bbox_xywh": [1, 2, 3, 4]}],
        },
        {
            "image_id": 2,
            "image_path": "images/two.jpg",
            "width": 640,
            "height": 640,
            "fold": 1,
            "annotations": [{"category_id": 4, "bbox_xywh": [5, 6, 7, 8]}],
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = export_dfine_coco_splits(
        manifest,
        tmp_path / "coco",
        validation_fold=1,
        class_agnostic=True,
        evaluation_images_used=True,
    )
    training = json.loads((tmp_path / "coco" / "instances_train.json").read_text())
    validation = json.loads((tmp_path / "coco" / "instances_validation.json").read_text())

    assert report["class_agnostic"] is True
    assert report["evaluation_images_used"] is True
    assert training["annotations"][0]["category_id"] == 0
    assert validation["annotations"][0]["category_id"] == 0
    assert validation["categories"] == [{"id": 0, "name": "bread_object", "supercategory": "bread"}]


def test_dfine_source_sets_resolve_images_under_workspace_and_reject_evaluation(tmp_path):
    train_root = tmp_path / "train"
    validation_root = tmp_path / "validation"
    train_root.mkdir()
    validation_root.mkdir()
    base = {
        "image_id": 1,
        "image_path": "image.jpg",
        "width": 64,
        "height": 64,
        "annotations": [{"category_id": 1, "bbox_xywh": [1, 2, 3, 4]}],
    }
    train_manifest = tmp_path / "train.jsonl"
    validation_manifest = tmp_path / "validation.jsonl"
    train_manifest.write_text(json.dumps(base), encoding="utf-8")
    validation_manifest.write_text(json.dumps(base), encoding="utf-8")

    report = export_dfine_coco_source_sets(
        [(train_manifest, train_root)],
        [(validation_manifest, validation_root)],
        tmp_path / "coco",
        workspace_root=tmp_path,
    )
    training = json.loads((tmp_path / "coco" / "instances_train.json").read_text())
    validation = json.loads((tmp_path / "coco" / "instances_validation.json").read_text())

    assert report["evaluation_images_used"] is False
    assert training["images"][0]["file_name"] == "train/image.jpg"
    assert validation["images"][0]["file_name"] == "validation/image.jpg"
    assert training["images"][0]["id"] != validation["images"][0]["id"]

    blocked = dict(base, training_allowed=False)
    validation_manifest.write_text(json.dumps(blocked), encoding="utf-8")
    with pytest.raises(ValueError, match="training-prohibited"):
        export_dfine_coco_source_sets(
            [(train_manifest, train_root)],
            [(validation_manifest, validation_root)],
            tmp_path / "blocked",
            workspace_root=tmp_path,
        )
