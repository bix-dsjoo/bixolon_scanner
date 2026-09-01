import json
from pathlib import Path

from PIL import Image, ImageDraw

from bixolon_scanner.training.operational_compositor import (
    OperationalCompositeRecipe,
    extract_annotated_cutout,
    generate_operational_composites,
)


def _fixture_collection(root: Path) -> Path:
    images = root / "images"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    background = Image.new("RGB", (180, 160), (236, 234, 229))
    background.save(images / "empty.jpg", quality=100)

    first = background.copy()
    draw = ImageDraw.Draw(first)
    draw.ellipse((30, 42, 95, 107), fill=(185, 91, 25))
    draw.ellipse((52, 64, 73, 85), fill=(236, 234, 229))
    first.save(images / "first.jpg", quality=100)

    second = background.copy()
    draw = ImageDraw.Draw(second)
    draw.rounded_rectangle((91, 38, 153, 113), radius=15, fill=(212, 132, 48))
    second.save(images / "second.jpg", quality=100)

    payload = {
        "images": [
            {"id": 1, "file_name": "../images/empty.jpg", "width": 180, "height": 160},
            {"id": 2, "file_name": "../images/first.jpg", "width": 180, "height": 160},
            {"id": 3, "file_name": "../images/second.jpg", "width": 180, "height": 160},
        ],
        "annotations": [
            {"id": 1, "image_id": 2, "category_id": 3, "bbox": [29, 41, 67, 67]},
            {"id": 2, "image_id": 3, "category_id": 6, "bbox": [90, 37, 64, 77]},
        ],
        "categories": [
            {"id": 3, "name": "Waffle"},
            {"id": 6, "name": "Croissant"},
        ],
    }
    annotation_path = annotations / "instances.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    return annotation_path


def test_background_difference_keeps_an_enclosed_hole(tmp_path):
    annotation_path = _fixture_collection(tmp_path / "collection")
    image_root = annotation_path.parent.parent / "images"
    with (
        Image.open(image_root / "first.jpg") as source,
        Image.open(image_root / "empty.jpg") as background,
    ):
        cutout = extract_annotated_cutout(
            source,
            background,
            bbox_xywh=(29, 41, 67, 67),
            transparent_distance=12,
            opaque_distance=36,
            feather_radius=0.0,
        )

    alpha = cutout.getchannel("A")
    center = alpha.getpixel((cutout.width // 2, cutout.height // 2))
    assert center == 0
    assert alpha.getextrema() == (0, 255)


def test_operational_composites_are_deterministic_and_export_coco(tmp_path):
    annotation_path = _fixture_collection(tmp_path / "collection")
    recipe = OperationalCompositeRecipe(
        image_count=2,
        minimum_objects=2,
        maximum_objects=2,
        minimum_source_scale=0.72,
        maximum_source_scale=0.78,
        placement_margin_fraction=0.02,
        maximum_occlusion_fraction=0.0,
        shadow_probability=1.0,
        mask_transparent_distance=12,
        mask_opaque_distance=36,
    )
    first = generate_operational_composites(
        annotation_path,
        tmp_path / "first-output",
        seed=31,
        recipe=recipe,
        source_image_names=["first.jpg", "second.jpg"],
        background_image_names=["empty.jpg"],
    )
    second = generate_operational_composites(
        annotation_path,
        tmp_path / "second-output",
        seed=31,
        recipe=recipe,
        source_image_names=["first.jpg", "second.jpg"],
        background_image_names=["empty.jpg"],
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["instances_sha256"] == second["instances_sha256"]
    assert first["synthetic_image_count"] == 2
    assert first["synthetic_annotation_count"] == 4
    coco = json.loads((tmp_path / "first-output" / "instances.json").read_text())
    assert {row["category_id"] for row in coco["annotations"]} == {3, 6}
    assert all(row["area"] > 0 for row in coco["annotations"])
    assert all(
        0 <= row["bbox"][0] < 180
        and 0 <= row["bbox"][1] < 160
        and row["bbox"][0] + row["bbox"][2] <= 180
        and row["bbox"][1] + row["bbox"][3] <= 160
        for row in coco["annotations"]
    )
    provenance = [
        json.loads(line)
        for line in (tmp_path / "first-output" / "provenance.jsonl").read_text().splitlines()
    ]
    assert all(row["background_image"] == "empty.jpg" for row in provenance)
    assert all(len(row["objects"]) == 2 for row in provenance)


def test_operational_compositor_rejects_nonempty_background_selection(tmp_path):
    annotation_path = _fixture_collection(tmp_path / "collection")
    try:
        generate_operational_composites(
            annotation_path,
            tmp_path / "output",
            seed=1,
            recipe=OperationalCompositeRecipe(image_count=1),
            background_image_names=["first.jpg"],
        )
    except ValueError as error:
        assert "not empty" in str(error)
    else:
        raise AssertionError("a nonempty background must be rejected")
