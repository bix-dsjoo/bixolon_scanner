from __future__ import annotations

import json
from pathlib import Path

import pytest

from bixolon_scanner.training.bread_dataset import (
    _safe_annotation_image,
    audit_bread_dataset,
    build_detection_evaluation_records,
)

DATASET_ROOT = Path(__file__).parents[1] / "datasets" / "bread_dataset"


def test_canonical_bread_dataset_locks_training_and_evaluation_sources():
    records, metadata = audit_bread_dataset(DATASET_ROOT)

    assert len(records) == 200
    assert {record["split"] for record in records} == {"train_support"}
    assert {Path(record["image_path"]).parts[0] for record in records} == {"single_objects"}
    assert metadata["ignored_top_level_entries"] == [
        "single_objects_1",
        "single_objects_2",
        "single_objects_3",
    ]
    assert metadata["training_contract"] == {
        "allowed_directory": "single_objects",
        "class_count": 20,
        "shots_per_class": 10,
        "original_image_count": 200,
        "derived_evaluation_images_are_training_forbidden": True,
    }
    assert metadata["evaluation_sets"]["multi_object_scenes"]["image_count"] == 300
    assert metadata["evaluation_sets"]["multi_object_scenes"]["annotation_count"] == 1410
    assert len(metadata["evaluation_sets"]["multi_object_scenes"]["image_content_sha256"]) == 64
    assert metadata["evaluation_sets"]["scan_log_samples"]["status_counts"] == {
        "ANNOTATED": 69,
        "RECAPTURE": 31,
    }


@pytest.mark.parametrize(
    ("training_source", "shots_per_class"),
    [
        ("single_objects", 10),
        ("single_objects_1", 7),
        ("single_objects_2", 10),
        ("single_objects_3", 12),
    ],
)
def test_training_sources_are_audited_independently(training_source, shots_per_class):
    records, metadata = audit_bread_dataset(DATASET_ROOT, training_source=training_source)

    assert len(records) == 20 * shots_per_class
    assert {Path(record["image_path"]).parts[0] for record in records} == {training_source}
    assert metadata["training_contract"]["allowed_directory"] == training_source
    assert metadata["training_contract"]["shots_per_class"] == shots_per_class
    assert training_source not in metadata["ignored_top_level_entries"]


def test_evaluation_manifest_is_grouped_and_detector_training_forbidden():
    records = build_detection_evaluation_records(DATASET_ROOT)

    assert len(records) == 300
    assert sum(len(row["annotations"]) for row in records) == 1410
    assert {row["split"] for row in records} == {"development"}
    assert all(row["exclude_from_detector_training"] for row in records)
    groups: dict[str, set[int]] = {}
    for row in records:
        groups.setdefault(row["capture_session_id"], set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in groups.values())


def test_annotation_path_cannot_escape_dataset(tmp_path):
    root = tmp_path / "bread_dataset"
    annotation = root / "annotations" / "instances.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "outside.jpg").write_bytes(b"outside")

    with pytest.raises(ValueError, match="escapes bread_dataset"):
        _safe_annotation_image(root.resolve(), annotation, "../../outside.jpg")
