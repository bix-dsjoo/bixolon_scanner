from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .bread_dataset import audit_bread_dataset
from .synthetic_roi import border_connected_background_alpha


@dataclass(frozen=True)
class MultiObjectRecipe:
    image_size: int = 640
    image_count: int = 1200
    minimum_objects: int = 1
    maximum_objects: int = 7
    empty_image_probability: float = 0.25
    minimum_object_scale: float = 0.12
    maximum_object_scale: float = 0.34
    maximum_rotation_degrees: float = 32.0
    maximum_overlap_iou: float = 0.18
    placement_attempts: int = 80
    background_min: int = 172
    background_max: int = 242
    structured_distractor_probability: float = 0.7
    maximum_structured_distractors: int = 6
    shadow_probability: float = 0.75
    blur_probability: float = 0.12
    jpeg_quality_min: int = 84
    jpeg_quality_max: int = 97

    def validate(self) -> None:
        if self.image_size < 320 or self.image_count < 3:
            raise ValueError("synthetic detector dataset is too small")
        if not 1 <= self.minimum_objects <= self.maximum_objects:
            raise ValueError("invalid synthetic object count range")
        if not 0 <= self.empty_image_probability <= 1:
            raise ValueError("invalid empty image probability")
        if not 0 < self.minimum_object_scale <= self.maximum_object_scale < 1:
            raise ValueError("invalid synthetic object scale range")
        if not 0 <= self.maximum_overlap_iou < 1:
            raise ValueError("invalid overlap threshold")
        if not 0 <= self.structured_distractor_probability <= 1:
            raise ValueError("invalid structured distractor probability")
        if self.maximum_structured_distractors < 0:
            raise ValueError("invalid structured distractor count")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _recipe_sha256(recipe: MultiObjectRecipe) -> str:
    recipe.validate()
    return hashlib.sha256(_canonical_json(asdict(recipe)).encode("utf-8")).hexdigest()


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _background(size: int, rng: np.random.Generator, recipe: MultiObjectRecipe) -> Image.Image:
    base = int(rng.integers(recipe.background_min, recipe.background_max + 1))
    tint = np.asarray(
        [int(np.clip(base + rng.integers(-9, 10), 0, 255)) for _ in range(3)],
        dtype=np.float32,
    )
    yy, xx = np.mgrid[0:size, 0:size]
    gradient = float(rng.uniform(-14, 14)) * (xx / max(size - 1, 1) - 0.5)
    gradient += float(rng.uniform(-14, 14)) * (yy / max(size - 1, 1) - 0.5)
    radial = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / size
    tray_delta = np.where(radial < float(rng.uniform(0.42, 0.58)), 6.0, -5.0)
    noise = rng.normal(0, 1.6, size=(size, size))
    array = np.empty((size, size, 3), dtype=np.uint8)
    for channel in range(3):
        array[..., channel] = np.clip(tint[channel] + gradient + tray_delta + noise, 0, 255).astype(
            np.uint8
        )
    image = Image.fromarray(array, mode="RGB")
    if (
        recipe.maximum_structured_distractors
        and rng.random() < recipe.structured_distractor_probability
    ):
        _draw_structured_distractors(image, rng, recipe.maximum_structured_distractors)
    return image


def _draw_structured_distractors(
    image: Image.Image, rng: np.random.Generator, maximum_count: int
) -> None:
    """Draw tray rails and seams that are negatives, never bread annotations."""
    draw = ImageDraw.Draw(image, mode="RGB")
    size = min(image.size)
    count = int(rng.integers(1, maximum_count + 1))
    for _ in range(count):
        base = int(rng.integers(118, 226))
        color = tuple(int(np.clip(base + rng.integers(-12, 13), 0, 255)) for _ in range(3))
        width = int(rng.integers(max(2, size // 240), max(4, size // 45) + 1))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            x = int(rng.integers(0, image.width))
            draw.line((x, 0, x + int(rng.integers(-12, 13)), image.height), fill=color, width=width)
        elif kind == 1:
            y = int(rng.integers(0, image.height))
            draw.line((0, y, image.width, y + int(rng.integers(-12, 13))), fill=color, width=width)
        else:
            margin_x = int(rng.integers(0, max(1, image.width // 4)))
            margin_y = int(rng.integers(0, max(1, image.height // 4)))
            right = int(rng.integers(max(margin_x + 2, image.width // 2), image.width + 1))
            bottom = int(rng.integers(max(margin_y + 2, image.height // 2), image.height + 1))
            draw.rounded_rectangle(
                (margin_x, margin_y, right - 1, bottom - 1),
                radius=max(2, size // 40),
                outline=color,
                width=width,
            )


def _prepare_cutout(path: Path) -> Image.Image:
    with Image.open(path) as source:
        cutout = border_connected_background_alpha(source, color_distance=46, feather_radius=0.9)
    bbox = cutout.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError(f"training original has no foreground: {path}")
    return cutout.crop(bbox)


def _transform_cutout(
    source: Image.Image,
    rng: np.random.Generator,
    recipe: MultiObjectRecipe,
) -> Image.Image:
    alpha = source.getchannel("A")
    rgb = ImageEnhance.Brightness(source.convert("RGB")).enhance(float(rng.uniform(0.82, 1.18)))
    rgb = ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(0.86, 1.14)))
    rgb = ImageEnhance.Color(rgb).enhance(float(rng.uniform(0.88, 1.12)))
    transformed = rgb.convert("RGBA")
    transformed.putalpha(alpha)
    if rng.random() < 0.5:
        transformed = ImageOps.mirror(transformed)
    transformed = transformed.rotate(
        float(rng.uniform(-recipe.maximum_rotation_degrees, recipe.maximum_rotation_degrees)),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    bbox = transformed.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("transformed cutout is empty")
    transformed = transformed.crop(bbox)
    target_long = max(
        8,
        round(
            recipe.image_size
            * float(rng.uniform(recipe.minimum_object_scale, recipe.maximum_object_scale))
        ),
    )
    ratio = target_long / max(transformed.size)
    return transformed.resize(
        (
            max(2, round(transformed.width * ratio)),
            max(2, round(transformed.height * ratio)),
        ),
        Image.Resampling.LANCZOS,
    )


def generate_synthetic_detector_dataset(
    dataset_root: Path,
    output_root: Path,
    *,
    seed: int,
    recipe: MultiObjectRecipe,
    training_source: str = "single_objects",
) -> dict[str, Any]:
    recipe.validate()
    records, dataset_metadata = audit_bread_dataset(
        dataset_root,
        training_source=training_source,
    )
    sources = [
        {
            **record,
            "path": dataset_root.resolve() / record["image_path"],
        }
        for record in records
    ]
    cutouts = [_prepare_cutout(source["path"]) for source in sources]
    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "images"
    image_root.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    recipe_digest = _recipe_sha256(recipe)
    annotation_id = 1
    for image_index in range(recipe.image_count):
        canvas = _background(recipe.image_size, rng, recipe)
        object_count = (
            0
            if rng.random() < recipe.empty_image_probability
            else int(rng.integers(recipe.minimum_objects, recipe.maximum_objects + 1))
        )
        selected = rng.choice(len(sources), size=object_count, replace=False)
        boxes: list[tuple[int, int, int, int]] = []
        annotations: list[dict[str, Any]] = []
        source_provenance: list[dict[str, Any]] = []
        for source_index in selected:
            cutout = _transform_cutout(cutouts[int(source_index)], rng, recipe)
            max_x = recipe.image_size - cutout.width
            max_y = recipe.image_size - cutout.height
            if max_x < 0 or max_y < 0:
                raise ValueError("synthetic cutout exceeds the frame")
            chosen: tuple[int, int, int, int] | None = None
            for _ in range(recipe.placement_attempts):
                left = int(rng.integers(0, max_x + 1))
                top = int(rng.integers(0, max_y + 1))
                candidate = (left, top, left + cutout.width, top + cutout.height)
                if all(
                    _iou(candidate, existing) <= recipe.maximum_overlap_iou for existing in boxes
                ):
                    chosen = candidate
                    break
            if chosen is None:
                continue
            left, top, right, bottom = chosen
            if rng.random() < recipe.shadow_probability:
                shadow = Image.new("L", canvas.size, 0)
                shadow.paste(cutout.getchannel("A"), (min(left + 4, max_x), min(top + 6, max_y)))
                shadow = shadow.filter(ImageFilter.GaussianBlur(5.0))
                shade = Image.new("RGB", canvas.size, (110, 105, 98))
                canvas.paste(shade, (0, 0), shadow.point(lambda value: round(value * 0.22)))
            canvas.paste(cutout.convert("RGB"), (left, top), cutout.getchannel("A"))
            boxes.append(chosen)
            width, height = right - left, bottom - top
            source = sources[int(source_index)]
            annotations.append(
                {
                    "annotation_id": annotation_id,
                    "category_id": int(source["category_id"]),
                    "bbox_xywh": [left, top, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            source_provenance.append(
                {
                    "image_sha256": source["image_sha256"],
                    "category_id": source["category_id"],
                    "bbox_xywh": [left, top, width, height],
                }
            )
            annotation_id += 1
        if object_count and not annotations:
            raise RuntimeError("synthetic frame contains no placed objects")
        if rng.random() < recipe.blur_probability:
            canvas = canvas.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 0.85))))
        quality = int(rng.integers(recipe.jpeg_quality_min, recipe.jpeg_quality_max + 1))
        file_name = f"synthetic_{image_index + 1:05d}.jpg"
        image_path = image_root / file_name
        canvas.save(image_path, format="JPEG", quality=quality)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        image_id = image_index + 1
        manifest_rows.append(
            {
                "record_type": "detection",
                "source": "bread_dataset_single_original_composite",
                "source_dataset": dataset_metadata["dataset_version"],
                "image_id": image_id,
                "image_path": f"images/{file_name}",
                "image_sha256": digest,
                "capture_session_id": f"synthetic:{image_id // 3:05d}",
                "physical_item_ids": [
                    f"bread_{row['category_id']:02d}:original_item" for row in source_provenance
                ],
                "split": "development",
                "fold": image_index % 3,
                "width": recipe.image_size,
                "height": recipe.image_size,
                "annotations": annotations,
                "generation_recipe_sha256": recipe_digest,
            }
        )
        provenance_rows.append(
            {
                "image_id": image_id,
                "image_sha256": digest,
                "seed": seed,
                "sources": source_provenance,
            }
        )
    manifest_body = "".join(_canonical_json(row) + "\n" for row in manifest_rows)
    (output_root / "manifest.jsonl").write_text(manifest_body, encoding="utf-8", newline="\n")
    (output_root / "provenance.jsonl").write_text(
        "".join(_canonical_json(row) + "\n" for row in provenance_rows),
        encoding="utf-8",
        newline="\n",
    )
    metadata = {
        "schema_version": "1.0",
        "dataset_version": dataset_metadata["dataset_version"],
        "training_source_policy": f"{training_source}-only",
        "source_original_count": len(sources),
        "synthetic_image_count": len(manifest_rows),
        "synthetic_annotation_count": sum(len(row["annotations"]) for row in manifest_rows),
        "synthetic_empty_image_count": sum(not row["annotations"] for row in manifest_rows),
        "seed": seed,
        "recipe": asdict(recipe),
        "recipe_sha256": recipe_digest,
        "manifest_sha256": hashlib.sha256(manifest_body.encode("utf-8")).hexdigest(),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate detector training composites from canonical 10-shot originals"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-count", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--empty-image-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--training-source", default="single_objects")
    args = parser.parse_args()
    metadata = generate_synthetic_detector_dataset(
        args.dataset_root,
        args.output_root,
        seed=args.seed,
        training_source=args.training_source,
        recipe=MultiObjectRecipe(
            image_size=args.image_size,
            image_count=args.image_count,
            empty_image_probability=args.empty_image_probability,
        ),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
