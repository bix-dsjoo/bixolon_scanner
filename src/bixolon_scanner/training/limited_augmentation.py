"""Generate training derivatives exclusively from a frozen original-image manifest."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from ..contracts.catalog import sha256_file
from .limited_source import read_jsonl, verify_sources, write_json, write_jsonl
from .synthetic_roi import DirectRoiRecipe, prepare_direct_roi_source


def procedural_background(size: int, rng: np.random.Generator, style: str) -> Image.Image:
    """Draw surfaces without reading extra photographs; used for positives and negatives."""
    if style not in {"gradient", "surfaces"}:
        raise ValueError("unsupported procedural background style")
    color = rng.integers(80, 240, size=3)
    gradient = np.linspace(-20, 20, size, dtype=np.float32)[None, :, None]
    pixels = np.broadcast_to(np.clip(color + gradient, 0, 255), (size, size, 3)).copy()
    if style == "surfaces":
        yy, xx = np.mgrid[:size, :size]
        grain = np.sin((yy + xx * rng.uniform(-0.2, 0.2)) * rng.uniform(0.1, 1.0))
        pixels += grain[:, :, None] * rng.uniform(2, 18)
    canvas = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).convert("RGBA")
    if style == "surfaces":
        draw = ImageDraw.Draw(canvas)
        for _ in range(int(rng.integers(0, 3))):
            x, y = (int(v) for v in rng.integers(0, max(1, size // 4), size=2))
            right, bottom = (int(v) for v in rng.integers(3 * size // 4, size, size=2))
            tone = int(rng.integers(130, 251))
            radius = int(rng.integers(0, max(1, size // 8)))
            draw.rounded_rectangle(
                (x, y, right, bottom),
                radius=radius,
                fill=(tone, tone, tone, 255),
                outline=(tone - 25,) * 3 + (255,),
                width=max(1, size // 100),
            )
            inset = max(2, size // 50)
            draw.rounded_rectangle(
                (x + inset, y + inset, right - inset, bottom - inset),
                radius=max(0, radius - inset),
                outline=(min(255, tone + 12),) * 3 + (255,),
                width=max(1, size // 160),
            )
    return canvas


def generate_training_derivatives(prepared: Path, output: Path, recipe: dict) -> dict:
    report = verify_sources(prepared)
    if output.exists():
        raise FileExistsError(output)
    size = int(recipe["image_size"])
    count = int(recipe["image_count"])
    maximum_objects = int(recipe["maximum_objects"])
    minimum_visible = float(recipe["minimum_visible_fraction"])
    empty_probability = float(recipe["empty_probability"])
    background_style = recipe.get("background_style", "gradient")
    if background_style not in {"gradient", "surfaces"}:
        raise ValueError("unsupported procedural background style")
    if size < 64 or count < 1 or not 1 <= maximum_objects <= 16:
        raise ValueError("invalid synthetic image dimensions or count")
    if not 0 < minimum_visible <= 1 or not 0 <= empty_probability < 1:
        raise ValueError("invalid synthetic visibility or empty-image probability")
    output.mkdir(parents=True)
    (output / "scenes").mkdir()
    (output / "crops").mkdir()
    root = Path(report["dataset_root"])
    originals = read_jsonl(prepared / "classifier.jsonl")
    cutouts = []
    cutout_recipe = DirectRoiRecipe(crop_mode="white_alpha_composite")
    for row in originals:
        with Image.open(root / row["image_path"]) as opened:
            cutout = prepare_direct_roi_source(opened, cutout_recipe)
            cutout.thumbnail((size, size), Image.Resampling.LANCZOS)
            cutouts.append(cutout)
    rng = np.random.default_rng(recipe["seed"])
    scene_records = []
    try:
        for index in range(count):
            for _attempt in range(100):
                n = (
                    0
                    if rng.random() < empty_probability
                    else int(rng.integers(1, maximum_objects + 1))
                )
                # Procedural backgrounds consume no additional original photographs.
                canvas = procedural_background(size, rng, background_style)
                ownership = np.zeros((size, size), dtype=np.int16)
                areas = []
                sources = []
                grid = max(1, math.ceil(math.sqrt(n)))
                cells = rng.permutation(grid * grid)[:n]
                for ordinal, cell in enumerate(cells, start=1):
                    source_index = int(rng.integers(len(originals)))
                    sources.append(originals[source_index])
                    cutout = cutouts[source_index].rotate(
                        float(rng.uniform(-180, 180)), expand=True
                    )
                    target = min(size, max(8, round(size / grid * rng.uniform(0.65, 1.18))))
                    cutout.thumbnail((target, target), Image.Resampling.LANCZOS)
                    cutout = ImageEnhance.Brightness(cutout).enhance(float(rng.uniform(0.8, 1.2)))
                    cx = (int(cell) % grid + 0.5) * size / grid
                    cy = (int(cell) // grid + 0.5) * size / grid
                    x = max(
                        0,
                        min(
                            size - cutout.width,
                            round(cx - cutout.width / 2 + rng.uniform(-0.1, 0.1) * size / grid),
                        ),
                    )
                    y = max(
                        0,
                        min(
                            size - cutout.height,
                            round(cy - cutout.height / 2 + rng.uniform(-0.1, 0.1) * size / grid),
                        ),
                    )
                    alpha = np.asarray(cutout.getchannel("A")) > 127
                    areas.append(int(alpha.sum()))
                    ownership[y : y + cutout.height, x : x + cutout.width][alpha] = ordinal
                    canvas.alpha_composite(cutout, (x, y))
                    cutout.close()
                visible_areas = [int(np.count_nonzero(ownership == i + 1)) for i in range(n)]
                if all(
                    a > 0 and v / a >= minimum_visible
                    for a, v in zip(areas, visible_areas, strict=True)
                ):
                    break
                canvas.close()
            else:
                raise RuntimeError("could not generate a sufficiently visible synthetic scene")
            annotations = []
            for ordinal, source in enumerate(sources, start=1):
                yy, xx = np.where(ownership == ordinal)
                bbox = [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max() - xx.min() + 1),
                    int(yy.max() - yy.min() + 1),
                ]
                annotations.append(
                    {
                        "annotation_id": index * maximum_objects + ordinal,
                        "category_id": source["category_id"],
                        "bbox_xywh": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "visible_fraction": visible_areas[ordinal - 1] / areas[ordinal - 1],
                        "original_image_sha256": source["image_sha256"],
                    }
                )
            relative = f"scenes/{index:05d}.jpg"
            canvas.convert("RGB").save(output / relative, quality=92)
            canvas.close()
            scene_records.append(
                {
                    "record_type": "detection",
                    "image_id": index + 1,
                    "image_path": relative,
                    "image_sha256": sha256_file(output / relative),
                    "width": size,
                    "height": size,
                    "split": "train",
                    "source": "limited220_procedural_composition",
                    "source_group": "shared_physical_item_collection",
                    "capture_session_id": "derived:shared_physical_item_collection",
                    "physical_item_ids": sorted(
                        {v for r in sources for v in r["physical_item_ids"]}
                    ),
                    "parent_sha256s": sorted({r["image_sha256"] for r in sources}),
                    "annotations": annotations,
                }
            )
    finally:
        for cutout in cutouts:
            cutout.close()

    crops = []
    for scene in read_jsonl(prepared / "detector.jsonl"):
        with Image.open(root / scene["image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        try:
            for annotation in scene["annotations"]:
                x, y, w, h = annotation["bbox_xywh"]
                relative = f"crops/{scene['image_id']}_{annotation['annotation_id']}.png"
                crop = image.crop((round(x), round(y), round(x + w), round(y + h)))
                crop.save(output / relative)
                crop.close()
                category = int(annotation["category_id"])
                crops.append(
                    {
                        "record_type": "classification",
                        "image_path": relative,
                        "image_sha256": sha256_file(output / relative),
                        "category_id": category,
                        "class_id": f"bread_{category:02d}",
                        "class_name": report["labels"][category - 1]["name"],
                        "original_image_sha256": scene["image_sha256"],
                        "parent_sha256s": [scene["image_sha256"]],
                        "split": "train",
                        "capture_session_id": scene["capture_session_id"],
                        "physical_item_id": f"bread_{category:02d}:shared_item",
                        "source_group": scene["source_group"],
                    }
                )
        finally:
            image.close()
    write_jsonl(output / "synthetic.jsonl", scene_records)
    write_jsonl(output / "multi-crops.jsonl", crops)
    result = {
        "schema_version": "1.0",
        "source_manifest_sha256": report["source_manifest_sha256"],
        "original_image_count": report["original_count"],
        "synthetic_count": len(scene_records),
        "multi_crop_count": len(crops),
        "recipe": recipe,
        "synthetic_manifest_sha256": sha256_file(output / "synthetic.jsonl"),
        "crop_manifest_sha256": sha256_file(output / "multi-crops.jsonl"),
        "external_background_images_used": False,
        "evaluation_role": "training_derivatives_not_independent_validation",
    }
    write_json(output / "derivative-report.json", result)
    return result
