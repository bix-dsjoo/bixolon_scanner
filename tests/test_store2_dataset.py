from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from bixolon_scanner.training.store2_dataset import (
    Store2SyntheticRecipe,
    audit_store2_source,
    generate_store2_synthetic_dataset,
    partition_complete_classifier_support,
)

DATASET_ROOT = Path(__file__).parents[1] / "datasets" / "bread_dataset"


def test_store2_source_audit_locks_exact_allowed_images():
    records, metadata = audit_store2_source(DATASET_ROOT / "single_objects_4")

    assert len(records) == 200
    assert metadata["object_image_count"] == 200
    assert metadata["background_image_count"] == 10
    assert metadata["class_count"] == 20
    assert metadata["shots_per_class"] == 10
    assert metadata["source_image_set_sha256"] == (
        "155af9f0aa3eb6a130f78f8a0586f7785a7ff9dec715e105565352b85f897b87"
    )
    assert {Path(row["image_path"]).parts[0] for row in records} == {
        path.name
        for path in (DATASET_ROOT / "single_objects_4").iterdir()
        if path.is_dir() and path.name.startswith("bread_")
    }


def test_bix_bakery_source_audit_supports_new_dataset_layout(tmp_path):
    root = tmp_path / "bix_bakery_dataset"
    single_root = root / "single_object"
    background_root = root / "background"
    background_root.mkdir(parents=True)
    for category_id in range(1, 21):
        class_root = single_root / f"bread_{category_id:02d}_class_{category_id:02d}"
        class_root.mkdir(parents=True)
        for capture_index in range(10):
            image = Image.new(
                "RGB",
                (32 + category_id, 32 + capture_index),
                (category_id * 10, capture_index * 20, 80),
            )
            image.save(class_root / f"capture_{capture_index:02d}.jpg", quality=100)
    for capture_index in range(10):
        image = Image.new("RGB", (48, 48 + capture_index), (240, 240, 240))
        image.save(background_root / f"background_{capture_index:02d}.jpg", quality=100)

    records, metadata = audit_store2_source(root)

    assert len(records) == 200
    assert metadata["source_directory"] == "bix_bakery_dataset"
    assert metadata["single_object_directory"] == "single_object"
    assert metadata["background_directory"] == "background"
    assert all(Path(row["image_path"]).parts[0] == "single_object" for row in records)
    assert {row["source_dataset"] for row in records} == {"bix_bakery_dataset"}


def test_synthetic_generation_refuses_unreviewed_annotations(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "metadata.json").write_text(
        json.dumps({"source_image_set_sha256": "source"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="approved annotation-review.json"):
        generate_store2_synthetic_dataset(
            tmp_path,
            prepared,
            tmp_path / "synthetic",
            recipe=Store2SyntheticRecipe(image_count=1),
        )


def test_store2_synthetic_recipe_covers_dense_and_edge_variation():
    recipe = Store2SyntheticRecipe()

    assert recipe.image_count == 6000
    assert recipe.maximum_objects == 7
    assert recipe.maximum_rotation_degrees == 180.0
    assert recipe.hard_scene_probability > 0
    assert recipe.border_probability > 0
    assert recipe.duplicate_class_probability > 0
    assert recipe.hard_maximum_overlap > recipe.medium_maximum_overlap
    assert recipe.hard_cluster_probability > 0
    assert 0 < recipe.minimum_visibility_fraction < 1
    assert 0 < recipe.side_view_minimum_compression < recipe.side_view_maximum_compression <= 1


def test_store2_synthetic_recipe_rejects_invalid_side_view_compression(tmp_path):
    with pytest.raises(ValueError, match="side-view compression range"):
        generate_store2_synthetic_dataset(
            tmp_path,
            tmp_path / "prepared",
            tmp_path / "synthetic",
            recipe=Store2SyntheticRecipe(
                image_count=1,
                side_view_minimum_compression=0.8,
                side_view_maximum_compression=0.4,
            ),
        )


def test_classifier_support_separates_source_frame_clipping():
    classifier = [
        {"original_image_path": "inside.jpg", "category_id": 1},
        {"original_image_path": "right.jpg", "category_id": 1},
        {"original_image_path": "top.jpg", "category_id": 2},
    ]
    detector = [
        {
            "image_path": "inside.jpg",
            "width": 100,
            "height": 80,
            "annotations": [{"bbox_xywh": [10, 10, 50, 50]}],
        },
        {
            "image_path": "right.jpg",
            "width": 100,
            "height": 80,
            "annotations": [{"bbox_xywh": [60, 10, 40, 50]}],
        },
        {
            "image_path": "top.jpg",
            "width": 100,
            "height": 80,
            "annotations": [{"bbox_xywh": [10, 0, 50, 50]}],
        },
    ]

    complete, clipped = partition_complete_classifier_support(classifier, detector)

    assert [row["original_image_path"] for row in complete] == ["inside.jpg"]
    assert [row["original_image_path"] for row in clipped] == ["right.jpg", "top.jpg"]
