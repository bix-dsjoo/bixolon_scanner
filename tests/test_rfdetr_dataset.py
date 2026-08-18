import json
import os

import pytest

from bixolon_scanner.training.rfdetr_dataset import export_rfdetr_coco_fold


def _row(image_id, path, fold, group, annotations):
    return {
        "image_id": image_id,
        "image_path": path,
        "width": 640,
        "height": 480,
        "fold": fold,
        "perceptual_group_id": group,
        "expected_image_status": "ANNOTATED" if annotations else "RECAPTURE",
        "expected_reason_codes": [] if annotations else ["NO_OBJECT"],
        "annotations": annotations,
    }


def test_rfdetr_export_is_group_aware_and_uses_hardlinks(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "images" / "train.jpg").write_bytes(b"train")
    (dataset / "images" / "valid.jpg").write_bytes(b"valid")
    rows = [
        _row(1, "images/train.jpg", 0, "group-a", []),
        _row(
            2,
            "images/valid.jpg",
            1,
            "group-b",
            [{"category_id": 20, "bbox_xywh": [10, 20, 30, 40]}],
        ),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output = tmp_path / "rfdetr"
    report = export_rfdetr_coco_fold(manifest, dataset, output, validation_fold=1)
    training = json.loads((output / "train" / "_annotations.coco.json").read_text())
    validation = json.loads((output / "valid" / "_annotations.coco.json").read_text())

    assert report["group_fold_overlap_count"] == 0
    assert report["training_empty_image_count"] == 1
    assert training["annotations"] == []
    assert validation["annotations"][0]["category_id"] == 20
    assert validation["categories"][0]["id"] == 1
    assert validation["categories"][-1]["id"] == 20
    assert os.path.samefile(
        dataset / "images" / "train.jpg", output / "train" / "images" / "train.jpg"
    )
    assert os.path.samefile(
        dataset / "images" / "valid.jpg", output / "valid" / "images" / "valid.jpg"
    )


def test_rfdetr_export_rejects_cross_fold_physical_group(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "one.jpg").write_bytes(b"one")
    (dataset / "two.jpg").write_bytes(b"two")
    rows = [
        _row(1, "one.jpg", 0, "same-object", []),
        _row(2, "two.jpg", 1, "same-object", []),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="group-aware fold leakage"):
        export_rfdetr_coco_fold(manifest, dataset, tmp_path / "rfdetr", validation_fold=1)


def test_rfdetr_export_can_use_one_detection_class(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "one.jpg").write_bytes(b"one")
    (dataset / "two.jpg").write_bytes(b"two")
    rows = [
        _row(1, "one.jpg", 0, "group-a", [{"category_id": 4, "bbox_xywh": [1, 2, 3, 4]}]),
        _row(2, "two.jpg", 1, "group-b", [{"category_id": 20, "bbox_xywh": [5, 6, 7, 8]}]),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output = tmp_path / "rfdetr"
    report = export_rfdetr_coco_fold(
        manifest, dataset, output, validation_fold=1, class_agnostic=True
    )
    training = json.loads((output / "train" / "_annotations.coco.json").read_text())
    validation = json.loads((output / "valid" / "_annotations.coco.json").read_text())

    assert report["class_agnostic"] is True
    assert training["annotations"][0]["category_id"] == 1
    assert validation["annotations"][0]["category_id"] == 1
    assert validation["categories"] == [{"id": 1, "name": "bread_object", "supercategory": "bread"}]
