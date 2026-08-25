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
    assert metadata["classifier"]["allowed_sources"] == [
        "single_objects",
        "single_objects_3",
    ]
    assert metadata["classifier"]["folds"]["assignment_key"] == "physical_item_id"
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


def test_registry_accepts_single_objects_3_without_mixing_sources():
    classifier, detector, metadata = build_bread_cross_validation_registry(
        DATASET_ROOT, classifier_source="single_objects_3", fold_count=3
    )

    assert len(classifier) == 240
    assert {Path(row["image_path"]).parts[0] for row in classifier} == {"single_objects_3"}
    assert len(detector) == 300
    assert metadata["classifier"]["mixed_sources"] is False


@pytest.mark.parametrize("classifier_source", ["single_objects", "single_objects_3"])
def test_classifier_physical_session_and_perceptual_groups_never_cross_folds(
    classifier_source,
):
    classifier, _, metadata = build_bread_cross_validation_registry(
        DATASET_ROOT, classifier_source=classifier_source, fold_count=3
    )

    for key in ("physical_item_id", "capture_session_id", "perceptual_group_id"):
        folds_by_group: dict[str, set[int]] = {}
        for row in classifier:
            folds_by_group.setdefault(row[key], set()).add(row["fold"])
        assert all(len(folds) == 1 for folds in folds_by_group.values())
    assert sorted(metadata["classifier"]["folds"]["group_counts"]) == [6, 7, 7]


def test_single_objects_2_is_not_a_classifier_candidate():
    with pytest.raises(ValueError, match="supported single-object collection"):
        build_bread_cross_validation_registry(
            DATASET_ROOT, classifier_source="single_objects_2", fold_count=3
        )


def test_registry_can_add_explicit_detector_operational_collection():
    classifier, detector, metadata = build_bread_cross_validation_registry(
        DATASET_ROOT,
        classifier_source="single_objects_3",
        detector_operational_collection=Path("2026-08-18"),
        fold_count=3,
    )

    assert len(classifier) == 240
    assert len(detector) == 415
    assert sum(len(row["annotations"]) for row in detector) == 1914
    assert {row["evaluation_set"] for row in detector} == {
        "multi_object_scenes",
        "operational_collections/2026-08-18",
    }
    assert metadata["detector"]["sources"] == [
        "multi_object_scenes",
        "operational_collections/2026-08-18",
    ]
    assert metadata["evaluation_policy"]["held_out_test_set"] is False


def test_registry_rejects_detector_collection_outside_operational_root(tmp_path):
    with pytest.raises(ValueError, match="inside operational_collections"):
        build_bread_cross_validation_registry(
            DATASET_ROOT,
            detector_operational_collection=tmp_path,
        )


def test_disallowed_classifier_source_is_rejected():
    with pytest.raises(ValueError, match="supported single-object collection"):
        build_bread_cross_validation_registry(
            DATASET_ROOT, classifier_source="not_a_collection", fold_count=3
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
