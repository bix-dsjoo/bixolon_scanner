from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bixolon_scanner.training.synthetic_roi import (
    ClutterRoiRecipe,
    DirectRoiRecipe,
    SyntheticRecipe,
    augment_clutter_roi,
    augment_direct_roi,
    border_connected_background_alpha,
    clutter_roi_recipe_sha256,
    compose_synthetic_frame,
    crop_with_margin,
    prepare_direct_roi_source,
    validate_single_detector_crop,
    white_background_alpha,
)


def _support() -> Image.Image:
    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 15, 140, 110), fill=(150, 80, 30))
    draw.ellipse((65, 45, 95, 75), fill="white")
    return image


def test_white_background_alpha_removes_white_and_keeps_object():
    masked = white_background_alpha(_support(), transparent_threshold=8, opaque_threshold=45)
    alpha = np.asarray(masked.getchannel("A"))
    assert alpha[0, 0] == 0
    assert alpha[50, 30] == 255


def test_border_connected_mask_preserves_enclosed_pale_foreground():
    image = Image.new("RGB", (9, 9), "white")
    pixels = image.load()
    for y in range(2, 7):
        for x in range(2, 7):
            pixels[x, y] = (210, 205, 190)
    pixels[4, 4] = (252, 252, 248)
    masked = border_connected_background_alpha(image, color_distance=20, feather_radius=0)
    assert masked.getpixel((0, 0))[3] == 0
    assert masked.getpixel((4, 4))[3] == 255


def test_border_connected_direct_roi_is_reproducible_and_records_crop_contract():
    recipe = DirectRoiRecipe(
        crop_mode="border_connected_composite",
        padding_ratio=0.07,
        procedural_shadow=True,
        blur_probability=0,
    )
    values = dict(source_sha256="a" * 64, category_id=1, seed=23, recipe=recipe)
    first = augment_direct_roi(_support(), **values)
    second = augment_direct_roi(_support(), **values)
    assert first.image.tobytes() == second.image.tobytes()
    assert first.provenance["parameters"]["crop_mode"] == "border_connected_composite"
    assert first.provenance["parameters"]["padding_ratio"] == 0.07
    prepared = prepare_direct_roi_source(_support(), recipe)
    reused = augment_direct_roi(_support(), prepared_cutout=prepared, **values)
    assert reused.image.tobytes() == first.image.tobytes()
    assert reused.provenance == first.provenance


def test_synthetic_frame_is_seeded_and_tracks_source_provenance():
    recipe = SyntheticRecipe(
        frame_width=320,
        frame_height=240,
        object_scale_min=0.35,
        object_scale_max=0.35,
        rotation_degrees=20,
        blur_probability=0,
    )
    kwargs = {
        "source_sha256": "a" * 64,
        "category_id": 3,
        "seed": 20260812,
        "recipe": recipe,
    }
    first = compose_synthetic_frame(_support(), **kwargs)
    second = compose_synthetic_frame(_support(), **kwargs)
    assert first.bbox_xyxy == second.bbox_xyxy
    assert first.provenance == second.provenance
    assert first.image.tobytes() == second.image.tobytes()
    assert first.image.size == (320, 240)
    assert first.provenance["source_sha256"] == "a" * 64
    assert first.provenance["category_id"] == 3


def test_direct_roi_is_seeded_and_never_claims_detector_provenance():
    recipe = DirectRoiRecipe(output_size=224, blur_probability=0)
    values = dict(source_sha256="b" * 64, category_id=2, seed=7, recipe=recipe)
    first = augment_direct_roi(_support(), **values)
    second = augment_direct_roi(_support(), **values)
    assert first.image.size == (224, 224)
    assert first.image.tobytes() == second.image.tobytes()
    assert first.provenance["mode"] == "ground_truth_foreground_roi"
    assert first.provenance["background_id"] == "procedural-neutral"
    assert "detector" not in str(first.provenance).lower()


@dataclass(frozen=True)
class _Detection:
    x1: float
    y1: float
    x2: float
    y2: float


def test_detector_validation_rejects_hard_gate_count_and_iou():
    expected = (10, 10, 110, 110)
    valid = _Detection(12, 12, 108, 108)
    assert (
        validate_single_detector_crop(
            [valid], expected_bbox_xyxy=expected, hard_reasons=[], minimum_iou=0.8
        )
        == valid
    )
    with pytest.raises(ValueError, match="hard gate"):
        validate_single_detector_crop(
            [valid],
            expected_bbox_xyxy=expected,
            hard_reasons=["DETECTOR_UNCERTAIN_OBJECT"],
            minimum_iou=0.8,
        )
    with pytest.raises(ValueError, match="exactly one"):
        validate_single_detector_crop(
            [], expected_bbox_xyxy=expected, hard_reasons=[], minimum_iou=0.8
        )
    with pytest.raises(ValueError, match="below"):
        validate_single_detector_crop(
            [_Detection(200, 200, 220, 220)],
            expected_bbox_xyxy=expected,
            hard_reasons=[],
            minimum_iou=0.8,
        )


def test_crop_with_margin_matches_expected_clamped_geometry():
    image = Image.new("RGB", (100, 80), "black")
    crop = crop_with_margin(image, (10, 20, 50, 60), margin_ratio=0.05)
    assert crop.size == (44, 44)
    edge = crop_with_margin(image, (0, 0, 20, 20), margin_ratio=0.05)
    assert edge.size == (21, 21)


def test_clutter_roi_is_deterministic_and_preserves_source_provenance():
    target = Image.new("RGBA", (80, 56), (0, 0, 0, 0))
    ImageDraw.Draw(target).ellipse((8, 8, 72, 48), fill=(190, 110, 45, 255))
    distractor = Image.new("RGBA", (52, 72), (0, 0, 0, 0))
    ImageDraw.Draw(distractor).rectangle((8, 5, 44, 67), fill=(115, 65, 32, 255))
    recipe = ClutterRoiRecipe(output_size=96, distractor_count_min=2, distractor_count_max=2)

    kwargs = {
        "target_sha256": "a" * 64,
        "target_category_id": 1,
        "distractors": [(distractor, "b" * 64, 2)],
        "seed": 42,
        "recipe": recipe,
    }
    first = augment_clutter_roi(target, **kwargs)
    second = augment_clutter_roi(target, **kwargs)

    assert first.image.tobytes() == second.image.tobytes()
    assert first.image.size == (96, 96)
    assert first.provenance["target_source_sha256"] == "a" * 64
    assert len(first.provenance["distractors"]) == 2
    assert all(row["source_sha256"] == "b" * 64 for row in first.provenance["distractors"])
    assert all(row["target_occlusion"] <= 0.12 for row in first.provenance["distractors"])
    assert first.provenance["recipe_sha256"] == clutter_roi_recipe_sha256(recipe)


def test_clutter_roi_rejects_same_class_distractors():
    cutout = Image.new("RGBA", (32, 32), (120, 80, 40, 255))
    with pytest.raises(ValueError, match="different category"):
        augment_clutter_roi(
            cutout,
            target_sha256="a" * 64,
            target_category_id=1,
            distractors=[(cutout, "b" * 64, 1)],
            seed=1,
            recipe=ClutterRoiRecipe(),
        )
