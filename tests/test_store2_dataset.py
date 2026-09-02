from __future__ import annotations

import json
from pathlib import Path

import pytest

from bixolon_scanner.training.store2_dataset import (
    Store2SyntheticRecipe,
    audit_store2_source,
    generate_store2_synthetic_dataset,
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
