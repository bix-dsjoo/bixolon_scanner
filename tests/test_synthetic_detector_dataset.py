import json
from pathlib import Path

from bixolon_scanner.training.synthetic_detector_dataset import (
    MultiObjectRecipe,
    generate_synthetic_detector_dataset,
)

DATASET_ROOT = Path(__file__).parents[1] / "datasets" / "bread_dataset"


def test_synthetic_detector_dataset_uses_only_registered_originals(tmp_path):
    output = tmp_path / "synthetic"
    metadata = generate_synthetic_detector_dataset(
        DATASET_ROOT,
        output,
        seed=7,
        recipe=MultiObjectRecipe(
            image_count=3,
            image_size=320,
            maximum_objects=2,
            empty_image_probability=0.0,
        ),
    )

    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["training_source_policy"] == "single_objects-only"
    assert metadata["source_original_count"] == 200
    assert metadata["synthetic_empty_image_count"] == sum(not row["annotations"] for row in rows)
    assert len(rows) == 3
    assert {row["source"] for row in rows} == {"bread_dataset_single_original_composite"}
    assert {row["fold"] for row in rows} == {0, 1, 2}
    assert all(row["annotations"] for row in rows)


def test_synthetic_detector_dataset_selects_one_training_source(tmp_path):
    output = tmp_path / "synthetic-seven-shot"
    metadata = generate_synthetic_detector_dataset(
        DATASET_ROOT,
        output,
        seed=17,
        training_source="single_objects_1",
        recipe=MultiObjectRecipe(
            image_count=3,
            image_size=320,
            maximum_objects=2,
            empty_image_probability=0.0,
        ),
    )

    assert metadata["training_source_policy"] == "single_objects_1-only"
    assert metadata["source_original_count"] == 140


def test_synthetic_detector_dataset_can_generate_background_negatives(tmp_path):
    output = tmp_path / "synthetic-negatives"
    metadata = generate_synthetic_detector_dataset(
        DATASET_ROOT,
        output,
        seed=11,
        recipe=MultiObjectRecipe(
            image_count=3,
            image_size=320,
            empty_image_probability=1.0,
        ),
    )

    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metadata["synthetic_empty_image_count"] == 3
    assert all(not row["annotations"] for row in rows)
    assert all(not row["physical_item_ids"] for row in rows)
    assert metadata["recipe"]["structured_distractor_probability"] == 0.7


def test_structured_hard_negatives_are_deterministic_and_versioned(tmp_path):
    recipe = MultiObjectRecipe(
        image_count=3,
        image_size=320,
        empty_image_probability=1.0,
        structured_distractor_probability=1.0,
        maximum_structured_distractors=4,
    )
    first = generate_synthetic_detector_dataset(
        DATASET_ROOT, tmp_path / "first", seed=23, recipe=recipe
    )
    second = generate_synthetic_detector_dataset(
        DATASET_ROOT, tmp_path / "second", seed=23, recipe=recipe
    )

    assert first["recipe_sha256"] == second["recipe_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["recipe"]["maximum_structured_distractors"] == 4
