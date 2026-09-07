from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from bixolon_scanner.training.raw_bread_dataset import (
    EXPECTED_CLASS_NAMES,
    audit_raw_bread_classifier_source,
    write_raw_bread_classifier_registry,
)


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "raw_data"
    for category_id, name in EXPECTED_CLASS_NAMES.items():
        directory = root / f"Bread{category_id:02d}_{name}"
        directory.mkdir(parents=True)
        for capture_index in range(84):
            Image.new(
                "RGB",
                (24 + category_id, 24 + capture_index),
                (category_id * 9, capture_index * 3, 90),
            ).save(directory / f"{category_id} ({capture_index + 1}).jpg", quality=100)
    return root


def test_raw_bread_audit_locks_complete_unique_source(tmp_path):
    records, metadata = audit_raw_bread_classifier_source(_source(tmp_path))

    assert len(records) == 1680
    assert metadata["class_count"] == 20
    assert metadata["images_per_class"] == 84
    assert len({record["image_sha256"] for record in records}) == 1680
    assert {record["source_group"] for record in records} == {
        f"bread_{category_id:02d}:raw_data:item" for category_id in range(1, 21)
    }


def test_raw_bread_audit_rejects_non_jpeg_and_duplicate(tmp_path):
    root = _source(tmp_path)
    duplicate = root / "Bread01_Walnut Donut" / "1 (84).jpg"
    duplicate.write_bytes((root / "Bread01_Walnut Donut" / "1 (1).jpg").read_bytes())
    with pytest.raises(ValueError, match="duplicate raw bread image"):
        audit_raw_bread_classifier_source(root)


def test_raw_bread_audit_ignores_only_windows_thumbnail_cache(tmp_path):
    root = _source(tmp_path)
    thumbnail = root / "Bread01_Walnut Donut" / "Thumbs.db"
    thumbnail.write_bytes(b"cache")

    _, metadata = audit_raw_bread_classifier_source(root)

    assert metadata["ignored_system_files"] == ["Bread01_Walnut Donut/Thumbs.db"]
    (root / "Bread01_Walnut Donut" / "notes.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="JPEG images and Thumbs.db only"):
        audit_raw_bread_classifier_source(root)


def test_raw_bread_registry_refuses_to_overwrite(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "registry"
    version = write_raw_bread_classifier_registry(source, output)

    assert version.startswith("raw-bread-")
    assert (output / "manifest.jsonl").read_text(encoding="utf-8").count("\n") == 1680
    with pytest.raises(FileExistsError, match="must be empty"):
        write_raw_bread_classifier_registry(source, output)
