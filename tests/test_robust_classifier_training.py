from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from bixolon_scanner.experiments.bread.robust_classifier_training import (
    hard_clutter_recipe,
    load_single_objects_records,
    mild_clutter_recipe,
    moderate_clutter_recipe,
    prepare_clutter_tensor,
)
from bixolon_scanner.training.synthetic_roi import ClutterRoiRecipe, augment_clutter_roi


def _row(path: str, digest: str, fold: int) -> dict[str, object]:
    return {
        "record_type": "classification",
        "split": "development",
        "image_path": path,
        "image_sha256": digest,
        "fold": fold,
        "category_id": 1,
    }


def test_classifier_source_rejects_mixed_single_object_roots(tmp_path: Path) -> None:
    rows = [
        _row("single_objects/a.jpg", "a" * 64, 0),
        _row("single_objects/b.jpg", "b" * 64, 1),
        _row("single_objects_2/c.jpg", "c" * 64, 2),
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="single_objects only"):
        load_single_objects_records(manifest)


def test_hard_clutter_recipe_exercises_occlusion_and_multiple_distractors() -> None:
    recipe = hard_clutter_recipe()

    recipe.validate()
    assert recipe.distractor_count_max >= 4
    assert recipe.foreground_distractor_probability >= 0.5
    assert recipe.maximum_target_occlusion >= 0.25


def test_moderate_recipe_keeps_target_more_visible_than_hard_recipe() -> None:
    moderate = moderate_clutter_recipe()
    hard = hard_clutter_recipe()

    moderate.validate()
    assert moderate.target_scale_min > hard.target_scale_min
    assert moderate.maximum_target_occlusion < hard.maximum_target_occlusion


def test_mild_recipe_matches_dominant_target_training_contract() -> None:
    mild = mild_clutter_recipe()
    moderate = moderate_clutter_recipe()

    mild.validate()
    assert mild.target_scale_min > moderate.target_scale_min
    assert mild.distractor_count_max < moderate.distractor_count_max
    assert mild.maximum_target_occlusion < moderate.maximum_target_occlusion


def test_neighbor_masked_clutter_tensor_uses_production_shape() -> None:
    target = Image.new("RGBA", (80, 56), (0, 0, 0, 0))
    ImageDraw.Draw(target).ellipse((8, 8, 72, 48), fill=(190, 110, 45, 255))
    distractor = Image.new("RGBA", (52, 72), (0, 0, 0, 0))
    ImageDraw.Draw(distractor).rectangle((8, 5, 44, 67), fill=(115, 65, 32, 255))
    sample = augment_clutter_roi(
        target,
        target_sha256="a" * 64,
        target_category_id=1,
        distractors=[(distractor, "b" * 64, 2)],
        seed=42,
        recipe=ClutterRoiRecipe(output_size=224),
    )

    tensor = prepare_clutter_tensor(sample, apply_neighbor_mask=True)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype.name == "float32"
    assert tensor.flags.c_contiguous
