from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from bixolon_scanner.training.ten_shot_manifest import (
    SIDES,
    VIEWS,
    audit_ten_shot_dataset,
    write_ten_shot_manifest,
)


def _labels(path: Path, classes: int = 2) -> Path:
    value = {
        "labels": [
            {
                "category_id": category,
                "class_id": f"bread_{category:02d}",
                "class_name": f"Class {category}",
            }
            for category in range(1, classes + 1)
        ]
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _dataset(root: Path, classes: int = 2) -> None:
    for category in range(1, classes + 1):
        directory = root / f"Bread{category:02d}_Class {category}"
        directory.mkdir(parents=True)
        number = 0
        for side in SIDES:
            for view in VIEWS:
                number += 1
                image = Image.new(
                    "RGB",
                    (240 + number, 230 + category),
                    (category * 20, number * 10, 30),
                )
                image.putpixel((number, category), (255, 255, 255))
                image.save(
                    directory / f"bread{category:02d}_{side}_{view}.jpg",
                    quality=95,
                )


def test_ten_shot_manifest_locks_slots_labels_and_provenance(tmp_path: Path):
    root = tmp_path / "bread_project_3"
    _dataset(root)
    (root / "Bread01_Class 1" / "Thumbs.db").write_bytes(b"system cache")
    labels = _labels(tmp_path / "labels.json")
    records, metadata, audit = audit_ten_shot_dataset(
        root,
        labels_metadata=labels,
        expected_classes=2,
    )
    assert len(records) == 20
    assert metadata["shots_per_class"] == 10
    assert metadata["dataset_version"].startswith("bread-10shot-")
    assert audit["exact_duplicate_count"] == 0
    assert {record["split"] for record in records} == {"train_support"}
    assert {record["source"] for record in records} == {
        "bread_project_3_ten_shot"
    }
    assert {record["class_id"] for record in records} == {"bread_01", "bread_02"}
    assert len({(record["category_id"], record["side"], record["view"]) for record in records}) == 20


def test_ten_shot_manifest_is_deterministic_and_writes_checksum(tmp_path: Path):
    root = tmp_path / "bread_project_3"
    _dataset(root)
    labels = _labels(tmp_path / "labels.json")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_version = write_ten_shot_manifest(
        root, first, labels_metadata=labels, expected_classes=2
    )
    second_version = write_ten_shot_manifest(
        root, second, labels_metadata=labels, expected_classes=2
    )
    assert first_version == second_version
    assert (first / "manifest.jsonl").read_bytes() == (
        second / "manifest.jsonl"
    ).read_bytes()
    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["manifest_sha256"] in metadata["dataset_version"] or (
        metadata["dataset_version"].endswith(metadata["manifest_sha256"][:12])
    )


def test_ten_shot_manifest_rejects_missing_slot(tmp_path: Path):
    root = tmp_path / "bread_project_3"
    _dataset(root)
    labels = _labels(tmp_path / "labels.json")
    (root / "Bread01_Class 1" / "bread01_normal_vertical.jpg").unlink()
    with pytest.raises(ValueError, match="requires 10 files"):
        audit_ten_shot_dataset(root, labels_metadata=labels, expected_classes=2)


def test_ten_shot_manifest_rejects_label_directory_mismatch(tmp_path: Path):
    root = tmp_path / "bread_project_3"
    _dataset(root)
    labels = _labels(tmp_path / "labels.json")
    (root / "Bread02_Class 2").rename(root / "Bread02_Wrong")
    with pytest.raises(ValueError, match="class name mismatch"):
        audit_ten_shot_dataset(root, labels_metadata=labels, expected_classes=2)


def test_ten_shot_manifest_rejects_exact_duplicate_across_classes(tmp_path: Path):
    root = tmp_path / "bread_project_3"
    _dataset(root)
    labels = _labels(tmp_path / "labels.json")
    source = root / "Bread01_Class 1" / "bread01_normal_vertical.jpg"
    target = root / "Bread02_Class 2" / "bread02_normal_vertical.jpg"
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="exact duplicate"):
        audit_ten_shot_dataset(root, labels_metadata=labels, expected_classes=2)
