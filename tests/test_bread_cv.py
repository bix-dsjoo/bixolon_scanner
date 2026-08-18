from pathlib import Path

import pytest

from bixolon_scanner.training.bread_cv import (
    assign_balanced_folds,
    build_bread_cross_validation_registry,
    hamming_distance,
    write_bread_cross_validation_registry,
)

DATASET_ROOT = Path(__file__).parents[1] / "datasets" / "bread_dataset"


def test_registry_uses_single_objects_and_multi_object_scenes_only(tmp_path):
    version = write_bread_cross_validation_registry(
        DATASET_ROOT, tmp_path, classifier_source="single_objects", fold_count=3
    )
    classifier, detector, metadata = build_bread_cross_validation_registry(
        DATASET_ROOT, classifier_source="single_objects", fold_count=3
    )

    assert version.startswith("bread-1.1-")
    assert len(classifier) == 200
    assert {Path(row["image_path"]).parts[0] for row in classifier} == {"single_objects"}
    assert metadata["classifier"]["mixed_sources"] is False
    assert len(detector) == 300
    assert {row["evaluation_set"] for row in detector} == {"multi_object_scenes"}
    assert metadata["detector"]["annotated_image_count"] == 300
    assert metadata["detector"]["expected_recapture_image_count"] == 0
    assert metadata["detector"]["annotation_count"] == 1410
    assert (tmp_path / "classifier_manifest.jsonl").is_file()
    assert (tmp_path / "detector_manifest.jsonl").is_file()


def test_perceptual_groups_never_cross_detector_folds():
    _, detector, metadata = build_bread_cross_validation_registry(
        DATASET_ROOT, classifier_source="single_objects", fold_count=3
    )
    folds_by_group: dict[str, set[int]] = {}
    for row in detector:
        folds_by_group.setdefault(row["perceptual_group_id"], set()).add(row["fold"])

    assert all(len(folds) == 1 for folds in folds_by_group.values())
    assert sorted(metadata["detector"]["folds"]["image_counts"]) == [99, 100, 101]


def test_disallowed_classifier_source_is_rejected():
    with pytest.raises(ValueError, match="exactly single_objects or single_objects_2"):
        build_bread_cross_validation_registry(
            DATASET_ROOT, classifier_source="single_objects_3", fold_count=3
        )


def test_fold_assignment_keeps_duplicate_group_together():
    rows = [
        {
            "perceptual_group_id": "same",
            "evaluation_set": "multi_object_scenes",
            "expected_image_status": "ANNOTATED",
            "difficulty": "EASY",
            "annotations": [],
        },
        {
            "perceptual_group_id": "same",
            "evaluation_set": "multi_object_scenes",
            "expected_image_status": "ANNOTATED",
            "difficulty": "EASY",
            "annotations": [],
        },
        {
            "perceptual_group_id": "other",
            "evaluation_set": "multi_object_scenes",
            "expected_image_status": "ANNOTATED",
            "difficulty": "HARD",
            "annotations": [],
        },
    ]

    assign_balanced_folds(rows, fold_count=2)

    assert rows[0]["fold"] == rows[1]["fold"]
    assert hamming_distance(0b1010, 0b0011) == 2
