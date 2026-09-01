from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class OperationalCompositeRecipe:
    """Parameters for composing annotated single breads over real empty trays."""

    image_count: int = 100
    minimum_objects: int = 2
    maximum_objects: int = 4
    minimum_source_scale: float = 0.82
    maximum_source_scale: float = 1.05
    maximum_object_long_side_fraction: float = 0.68
    maximum_rotation_degrees: float = 180.0
    horizontal_flip_probability: float = 0.5
    minimum_brightness: float = 0.90
    maximum_brightness: float = 1.10
    minimum_contrast: float = 0.92
    maximum_contrast: float = 1.08
    minimum_saturation: float = 0.92
    maximum_saturation: float = 1.08
    placement_margin_fraction: float = 0.07
    maximum_occlusion_fraction: float = 0.06
    placement_attempts: int = 160
    shadow_probability: float = 0.9
    minimum_shadow_opacity: float = 0.12
    maximum_shadow_opacity: float = 0.24
    minimum_shadow_blur: float = 4.0
    maximum_shadow_blur: float = 11.0
    minimum_shadow_offset: float = 3.0
    maximum_shadow_offset: float = 10.0
    mask_transparent_distance: int = 16
    mask_opaque_distance: int = 48
    mask_feather_radius: float = 0.8
    jpeg_quality_min: int = 90
    jpeg_quality_max: int = 97

    def validate(self) -> None:
        if self.image_count < 1:
            raise ValueError("image_count must be positive")
        if not 1 <= self.minimum_objects <= self.maximum_objects:
            raise ValueError("invalid object count range")
        if not 0 < self.minimum_source_scale <= self.maximum_source_scale:
            raise ValueError("invalid source-relative scale range")
        if not 0 < self.maximum_object_long_side_fraction <= 1:
            raise ValueError("maximum_object_long_side_fraction must be in (0, 1]")
        if self.maximum_rotation_degrees < 0:
            raise ValueError("maximum_rotation_degrees cannot be negative")
        for name, value in (
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("placement_margin_fraction", self.placement_margin_fraction),
            ("maximum_occlusion_fraction", self.maximum_occlusion_fraction),
            ("shadow_probability", self.shadow_probability),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for minimum, maximum, name in (
            (self.minimum_brightness, self.maximum_brightness, "brightness"),
            (self.minimum_contrast, self.maximum_contrast, "contrast"),
            (self.minimum_saturation, self.maximum_saturation, "saturation"),
            (self.minimum_shadow_opacity, self.maximum_shadow_opacity, "shadow opacity"),
            (self.minimum_shadow_blur, self.maximum_shadow_blur, "shadow blur"),
            (self.minimum_shadow_offset, self.maximum_shadow_offset, "shadow offset"),
        ):
            if minimum < 0 or minimum > maximum:
                raise ValueError(f"invalid {name} range")
        if self.placement_attempts < 1:
            raise ValueError("placement_attempts must be positive")
        if not 0 <= self.mask_transparent_distance < self.mask_opaque_distance <= 255:
            raise ValueError("invalid foreground mask thresholds")
        if self.mask_feather_radius < 0:
            raise ValueError("mask_feather_radius cannot be negative")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("invalid JPEG quality range")


@dataclass(frozen=True)
class _SourceRecord:
    image_id: int
    path: Path
    image_sha256: str
    category_id: int
    category_name: str
    bbox_xywh: tuple[float, float, float, float]
    frame_size: tuple[int, int]


@dataclass(frozen=True)
class _BackgroundRecord:
    image_id: int
    path: Path
    image_sha256: str


@dataclass
class _PlacedObject:
    cutout: Image.Image
    left: int
    top: int
    bbox_xywh: tuple[int, int, int, int]
    mask_area: int
    source: _SourceRecord
    parameters: dict[str, Any]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recipe_sha256(recipe: OperationalCompositeRecipe) -> str:
    recipe.validate()
    return hashlib.sha256(_canonical_json(asdict(recipe)).encode("utf-8")).hexdigest()


def _load_coco(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read COCO annotations: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("COCO annotations must contain an object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"COCO annotations are missing {key}")
    return value


def _resolve_image(annotation_path: Path, file_name: str) -> Path:
    collection_root = annotation_path.resolve().parent.parent
    candidate = (annotation_path.resolve().parent / file_name).resolve()
    try:
        candidate.relative_to(collection_root)
    except ValueError as error:
        raise ValueError(f"COCO image escapes the collection root: {file_name}") from error
    if not candidate.is_file():
        raise ValueError(f"COCO image is missing: {file_name}")
    return candidate


def _selected_name(path: Path, names: set[str] | None) -> bool:
    return names is None or path.name in names


def _read_collection(
    annotation_path: Path,
    *,
    source_image_names: Sequence[str] | None,
    background_image_names: Sequence[str] | None,
) -> tuple[list[_SourceRecord], list[_BackgroundRecord], list[dict[str, Any]]]:
    coco = _load_coco(annotation_path)
    categories = {int(row["id"]): str(row["name"]) for row in coco["categories"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for row in coco["annotations"]:
        annotations_by_image.setdefault(int(row["image_id"]), []).append(row)

    requested_sources = set(source_image_names) if source_image_names else None
    requested_backgrounds = set(background_image_names) if background_image_names else None
    found_sources: set[str] = set()
    found_backgrounds: set[str] = set()
    sources: list[_SourceRecord] = []
    backgrounds: list[_BackgroundRecord] = []
    for image_row in coco["images"]:
        image_id = int(image_row["id"])
        path = _resolve_image(annotation_path, str(image_row["file_name"]))
        annotations = annotations_by_image.get(image_id, [])
        if len(annotations) == 1 and _selected_name(path, requested_sources):
            annotation = annotations[0]
            bbox = tuple(float(value) for value in annotation["bbox"])
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"invalid single-object bbox: {path.name}")
            category_id = int(annotation["category_id"])
            if category_id not in categories:
                raise ValueError(f"unknown category {category_id}: {path.name}")
            sources.append(
                _SourceRecord(
                    image_id=image_id,
                    path=path,
                    image_sha256=_sha256_file(path),
                    category_id=category_id,
                    category_name=categories[category_id],
                    bbox_xywh=bbox,
                    frame_size=(int(image_row["width"]), int(image_row["height"])),
                )
            )
            found_sources.add(path.name)
        if not annotations and _selected_name(path, requested_backgrounds):
            backgrounds.append(
                _BackgroundRecord(
                    image_id=image_id,
                    path=path,
                    image_sha256=_sha256_file(path),
                )
            )
            found_backgrounds.add(path.name)

    if requested_sources is not None and found_sources != requested_sources:
        missing = sorted(requested_sources - found_sources)
        raise ValueError(
            f"requested single-object images are missing or not single-object: {missing}"
        )
    if requested_backgrounds is not None and found_backgrounds != requested_backgrounds:
        missing = sorted(requested_backgrounds - found_backgrounds)
        raise ValueError(f"requested empty background images are missing or not empty: {missing}")
    if not sources:
        raise ValueError("the COCO collection has no selected single-object sources")
    if not backgrounds:
        raise ValueError("the COCO collection has no selected empty backgrounds")
    output_categories = [
        {"id": category_id, "name": name} for category_id, name in sorted(categories.items())
    ]
    return sources, backgrounds, output_categories


def _largest_seeded_component(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    height, width = weak.shape
    visited = np.zeros_like(weak, dtype=bool)
    best_coordinates: list[tuple[int, int]] = []
    best_score = (-1, -1)
    for start_y, start_x in np.argwhere(strong):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x] or not weak[y, x]:
            continue
        queue: deque[tuple[int, int]] = deque([(y, x)])
        coordinates: list[tuple[int, int]] = []
        strong_count = 0
        while queue:
            current_y, current_x = queue.popleft()
            if visited[current_y, current_x] or not weak[current_y, current_x]:
                continue
            visited[current_y, current_x] = True
            coordinates.append((current_y, current_x))
            strong_count += int(strong[current_y, current_x])
            if current_y:
                queue.append((current_y - 1, current_x))
            if current_y + 1 < height:
                queue.append((current_y + 1, current_x))
            if current_x:
                queue.append((current_y, current_x - 1))
            if current_x + 1 < width:
                queue.append((current_y, current_x + 1))
        score = (strong_count, len(coordinates))
        if score > best_score:
            best_score = score
            best_coordinates = coordinates
    result = np.zeros_like(weak, dtype=bool)
    if best_coordinates:
        yy, xx = zip(*best_coordinates, strict=True)
        result[np.asarray(yy), np.asarray(xx)] = True
    return result


def _bbox_with_margin(
    bbox_xywh: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox_xywh
    margin = max(3, int(round(max(width, height) * 0.04)))
    return (
        max(0, int(math.floor(x)) - margin),
        max(0, int(math.floor(y)) - margin),
        min(image_size[0], int(math.ceil(x + width)) + margin),
        min(image_size[1], int(math.ceil(y + height)) + margin),
    )


def extract_annotated_cutout(
    source_image: Image.Image,
    background_image: Image.Image,
    *,
    bbox_xywh: tuple[float, float, float, float],
    transparent_distance: int,
    opaque_distance: int,
    feather_radius: float,
) -> Image.Image:
    """Extract one annotated object by differencing an aligned empty-tray image."""
    source = ImageOps.exif_transpose(source_image).convert("RGB")
    background = ImageOps.exif_transpose(background_image).convert("RGB")
    if source.size != background.size:
        raise ValueError("source and empty background dimensions must match")
    bounds = _bbox_with_margin(bbox_xywh, image_size=source.size)
    source_array = np.asarray(source.crop(bounds), dtype=np.int16)
    background_array = np.asarray(background.crop(bounds), dtype=np.int16)

    border_source = np.concatenate(
        (source_array[0], source_array[-1], source_array[:, 0], source_array[:, -1]), axis=0
    )
    border_background = np.concatenate(
        (
            background_array[0],
            background_array[-1],
            background_array[:, 0],
            background_array[:, -1],
        ),
        axis=0,
    )
    exposure_delta = np.median(border_source - border_background, axis=0)
    adjusted_background = np.clip(background_array + exposure_delta, 0, 255)
    distance = np.max(np.abs(source_array - adjusted_background), axis=-1).astype(np.float32)
    weak = distance >= transparent_distance
    strong = distance >= opaque_distance
    connected = _largest_seeded_component(weak, strong)
    if not connected.any():
        raise ValueError("foreground mask is empty after background subtraction")

    alpha = np.clip(
        (distance - transparent_distance) / max(opaque_distance - transparent_distance, 1),
        0.0,
        1.0,
    )
    alpha = np.rint(alpha * connected * 255.0).astype(np.uint8)
    alpha_image = Image.fromarray(alpha, mode="L")
    if feather_radius:
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(feather_radius))
    rgba = Image.fromarray(source_array.astype(np.uint8), mode="RGB").convert("RGBA")
    rgba.putalpha(alpha_image)
    visible = alpha_image.point(lambda value: 255 if value >= 4 else 0).getbbox()
    if visible is None:
        raise ValueError("foreground mask disappeared after feathering")
    return rgba.crop(visible)


def _load_cutouts(
    sources: Sequence[_SourceRecord],
    reference_background: _BackgroundRecord,
    recipe: OperationalCompositeRecipe,
) -> dict[int, Image.Image]:
    with Image.open(reference_background.path) as opened_background:
        background = ImageOps.exif_transpose(opened_background).convert("RGB")
    cutouts: dict[int, Image.Image] = {}
    for source in sources:
        with Image.open(source.path) as opened_source:
            cutouts[source.image_id] = extract_annotated_cutout(
                opened_source,
                background,
                bbox_xywh=source.bbox_xywh,
                transparent_distance=recipe.mask_transparent_distance,
                opaque_distance=recipe.mask_opaque_distance,
                feather_radius=recipe.mask_feather_radius,
            )
    return cutouts


def _trim_alpha(image: Image.Image) -> Image.Image:
    bbox = image.getchannel("A").point(lambda value: 255 if value >= 4 else 0).getbbox()
    if bbox is None:
        raise ValueError("transformed foreground is empty")
    return image.crop(bbox)


def _transform_cutout(
    source: Image.Image,
    *,
    source_frame_size: tuple[int, int],
    canvas_size: tuple[int, int],
    crowd_scale: float,
    rng: np.random.Generator,
    recipe: OperationalCompositeRecipe,
) -> tuple[Image.Image, dict[str, Any]]:
    alpha = source.getchannel("A")
    brightness = float(rng.uniform(recipe.minimum_brightness, recipe.maximum_brightness))
    contrast = float(rng.uniform(recipe.minimum_contrast, recipe.maximum_contrast))
    saturation = float(rng.uniform(recipe.minimum_saturation, recipe.maximum_saturation))
    rgb = ImageEnhance.Brightness(source.convert("RGB")).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    transformed = rgb.convert("RGBA")
    transformed.putalpha(alpha)
    mirrored = bool(rng.random() < recipe.horizontal_flip_probability)
    if mirrored:
        transformed = ImageOps.mirror(transformed)

    source_scale = float(rng.uniform(recipe.minimum_source_scale, recipe.maximum_source_scale))
    frame_scale = min(canvas_size) / min(source_frame_size)
    resize_ratio = source_scale * crowd_scale * frame_scale
    maximum_long_side = min(canvas_size) * recipe.maximum_object_long_side_fraction
    resize_ratio = min(resize_ratio, maximum_long_side / max(transformed.size))
    transformed = transformed.resize(
        (
            max(2, int(round(transformed.width * resize_ratio))),
            max(2, int(round(transformed.height * resize_ratio))),
        ),
        Image.Resampling.LANCZOS,
    )
    rotation = float(rng.uniform(-recipe.maximum_rotation_degrees, recipe.maximum_rotation_degrees))
    transformed = transformed.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    transformed = _trim_alpha(transformed)
    return transformed, {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "mirrored": mirrored,
        "rotation_degrees": rotation,
        "source_scale": source_scale,
        "crowd_scale": crowd_scale,
        "effective_resize_ratio": resize_ratio,
    }


def _place_cutout(
    cutout: Image.Image,
    *,
    occupancy: np.ndarray,
    rng: np.random.Generator,
    recipe: OperationalCompositeRecipe,
) -> tuple[int, int, np.ndarray] | None:
    height, width = occupancy.shape
    margin_x = int(round(width * recipe.placement_margin_fraction))
    margin_y = int(round(height * recipe.placement_margin_fraction))
    min_left = margin_x
    min_top = margin_y
    max_left = width - margin_x - cutout.width
    max_top = height - margin_y - cutout.height
    if max_left < min_left or max_top < min_top:
        return None
    local_mask = np.asarray(cutout.getchannel("A")) >= 24
    local_area = max(1, int(np.count_nonzero(local_mask)))
    for _ in range(recipe.placement_attempts):
        left = int(rng.integers(min_left, max_left + 1))
        top = int(rng.integers(min_top, max_top + 1))
        region = occupancy[top : top + cutout.height, left : left + cutout.width]
        overlap = int(np.count_nonzero(region & local_mask)) / local_area
        if overlap <= recipe.maximum_occlusion_fraction:
            return left, top, local_mask
    return None


def _full_mask(
    local_alpha: Image.Image,
    *,
    canvas_size: tuple[int, int],
    left: int,
    top: int,
) -> Image.Image:
    result = Image.new("L", canvas_size, 0)
    result.paste(local_alpha, (left, top))
    return result


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="JPEG", quality=quality, optimize=False)
    return stream.getvalue()


def generate_operational_composites(
    annotation_path: Path,
    output_root: Path,
    *,
    seed: int,
    recipe: OperationalCompositeRecipe,
    source_image_names: Sequence[str] | None = None,
    background_image_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create reproducible multi-bread scenes and COCO/provenance artifacts."""
    recipe.validate()
    sources, backgrounds, categories = _read_collection(
        annotation_path.resolve(),
        source_image_names=source_image_names,
        background_image_names=background_image_names,
    )
    if recipe.minimum_objects > len(sources):
        raise ValueError("minimum_objects exceeds the selected source count")
    maximum_objects = min(recipe.maximum_objects, len(sources))
    cutouts = _load_cutouts(sources, backgrounds[0], recipe)
    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "images"
    image_root.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    recipe_digest = _recipe_sha256(recipe)
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    annotation_id = 1

    for image_index in range(recipe.image_count):
        background = backgrounds[int(rng.integers(0, len(backgrounds)))]
        with Image.open(background.path) as opened_background:
            canvas = ImageOps.exif_transpose(opened_background).convert("RGB")
        canvas_size = canvas.size
        occupancy = np.zeros((canvas.height, canvas.width), dtype=bool)
        object_count = int(rng.integers(recipe.minimum_objects, maximum_objects + 1))
        crowd_scale = min(1.0, math.sqrt(2.6 / object_count))
        selected_indices = rng.choice(len(sources), size=object_count, replace=False)
        placed: list[_PlacedObject] = []
        transformed_sources: list[tuple[_SourceRecord, Image.Image, dict[str, Any]]] = []
        for selected_index in selected_indices:
            source = sources[int(selected_index)]
            transformed, parameters = _transform_cutout(
                cutouts[source.image_id],
                source_frame_size=source.frame_size,
                canvas_size=canvas_size,
                crowd_scale=crowd_scale,
                rng=rng,
                recipe=recipe,
            )
            transformed_sources.append((source, transformed, parameters))
        transformed_sources.sort(
            key=lambda row: row[1].width * row[1].height,
            reverse=True,
        )
        for source, transformed, parameters in transformed_sources:
            placement = _place_cutout(
                transformed,
                occupancy=occupancy,
                rng=rng,
                recipe=recipe,
            )
            if placement is None:
                continue
            left, top, local_mask = placement
            occupancy[top : top + transformed.height, left : left + transformed.width] |= local_mask
            local_bbox = (
                transformed.getchannel("A").point(lambda value: 255 if value >= 24 else 0).getbbox()
            )
            if local_bbox is None:
                continue
            bbox_left = left + local_bbox[0]
            bbox_top = top + local_bbox[1]
            bbox_width = local_bbox[2] - local_bbox[0]
            bbox_height = local_bbox[3] - local_bbox[1]
            parameters["position"] = [left, top]
            placed.append(
                _PlacedObject(
                    cutout=transformed,
                    left=left,
                    top=top,
                    bbox_xywh=(bbox_left, bbox_top, bbox_width, bbox_height),
                    mask_area=int(np.count_nonzero(local_mask)),
                    source=source,
                    parameters=parameters,
                )
            )
        if len(placed) < recipe.minimum_objects:
            raise RuntimeError(
                f"could not place at least {recipe.minimum_objects} objects in image {image_index + 1}"
            )

        shadow_angle = float(rng.uniform(0, math.tau))
        shadow_opacity = float(
            rng.uniform(recipe.minimum_shadow_opacity, recipe.maximum_shadow_opacity)
        )
        shadow_blur = float(rng.uniform(recipe.minimum_shadow_blur, recipe.maximum_shadow_blur))
        shadow_offset = float(
            rng.uniform(recipe.minimum_shadow_offset, recipe.maximum_shadow_offset)
        )
        shadow_enabled = bool(rng.random() < recipe.shadow_probability)
        if shadow_enabled:
            combined_shadow = np.zeros((canvas.height, canvas.width), dtype=np.float32)
            offset_x = int(round(math.cos(shadow_angle) * shadow_offset))
            offset_y = int(round(math.sin(shadow_angle) * shadow_offset))
            for item in placed:
                shadow = _full_mask(
                    item.cutout.getchannel("A"),
                    canvas_size=canvas_size,
                    left=item.left + offset_x,
                    top=item.top + offset_y,
                ).filter(ImageFilter.GaussianBlur(shadow_blur))
                combined_shadow = np.maximum(
                    combined_shadow, np.asarray(shadow, dtype=np.float32) / 255.0
                )
            combined_shadow *= shadow_opacity
            combined_shadow[occupancy] = 0.0
            shadow_alpha = Image.fromarray(
                np.rint(np.clip(combined_shadow, 0.0, 1.0) * 255).astype(np.uint8), mode="L"
            )
            shadow_color = Image.new("RGB", canvas_size, (72, 66, 58))
            canvas.paste(shadow_color, (0, 0), shadow_alpha)

        for item in placed:
            canvas.paste(
                item.cutout.convert("RGB"),
                (item.left, item.top),
                item.cutout.getchannel("A"),
            )

        quality = int(rng.integers(recipe.jpeg_quality_min, recipe.jpeg_quality_max + 1))
        file_name = f"composite_{image_index + 1:05d}.jpg"
        image_path = image_root / file_name
        image_bytes = _jpeg_bytes(canvas, quality)
        image_path.write_bytes(image_bytes)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        image_id = image_index + 1
        coco_images.append(
            {
                "id": image_id,
                "file_name": f"images/{file_name}",
                "width": canvas.width,
                "height": canvas.height,
            }
        )
        row_annotations: list[dict[str, Any]] = []
        object_provenance: list[dict[str, Any]] = []
        for item in placed:
            bbox = list(item.bbox_xywh)
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": item.source.category_id,
                "bbox": bbox,
                "area": item.mask_area,
                "iscrowd": 0,
            }
            coco_annotations.append(annotation)
            row_annotations.append(
                {
                    "annotation_id": annotation_id,
                    "category_id": item.source.category_id,
                    "bbox_xywh": bbox,
                    "area": item.mask_area,
                    "iscrowd": 0,
                }
            )
            object_provenance.append(
                {
                    "source_image": item.source.path.name,
                    "source_image_sha256": item.source.image_sha256,
                    "source_image_id": item.source.image_id,
                    "category_id": item.source.category_id,
                    "category_name": item.source.category_name,
                    "bbox_xywh": bbox,
                    "parameters": item.parameters,
                }
            )
            annotation_id += 1
        manifest_rows.append(
            {
                "record_type": "detection",
                "source": "operational_single_object_composite",
                "image_id": image_id,
                "image_path": f"images/{file_name}",
                "image_sha256": image_sha256,
                "capture_session_id": f"synthetic-operational:{image_id:05d}",
                "physical_item_ids": [f"operational:{item.source.image_sha256}" for item in placed],
                "split": "development",
                "width": canvas.width,
                "height": canvas.height,
                "annotations": row_annotations,
                "generation_recipe_sha256": recipe_digest,
            }
        )
        provenance_rows.append(
            {
                "image_id": image_id,
                "image_sha256": image_sha256,
                "seed": seed,
                "background_image": background.path.name,
                "background_image_sha256": background.image_sha256,
                "shadow": {
                    "enabled": shadow_enabled,
                    "angle_radians": shadow_angle,
                    "opacity": shadow_opacity,
                    "blur_radius": shadow_blur,
                    "offset_pixels": shadow_offset,
                },
                "jpeg_quality": quality,
                "objects": object_provenance,
            }
        )

    manifest_body = "".join(_canonical_json(row) + "\n" for row in manifest_rows)
    provenance_body = "".join(_canonical_json(row) + "\n" for row in provenance_rows)
    coco_payload = {
        "info": {
            "description": "Operational single-object bread composites",
            "version": "1.0",
            "generation_recipe_sha256": recipe_digest,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    coco_body = json.dumps(coco_payload, ensure_ascii=False, indent=2) + "\n"
    (output_root / "manifest.jsonl").write_text(manifest_body, encoding="utf-8", newline="\n")
    (output_root / "provenance.jsonl").write_text(provenance_body, encoding="utf-8", newline="\n")
    (output_root / "instances.json").write_text(coco_body, encoding="utf-8", newline="\n")
    metadata = {
        "schema_version": "1.0",
        "source_annotation_file": annotation_path.name,
        "source_annotation_sha256": _sha256_file(annotation_path),
        "source_image_count": len(sources),
        "background_image_count": len(backgrounds),
        "synthetic_image_count": len(coco_images),
        "synthetic_annotation_count": len(coco_annotations),
        "seed": seed,
        "recipe": asdict(recipe),
        "recipe_sha256": recipe_digest,
        "manifest_sha256": hashlib.sha256(manifest_body.encode("utf-8")).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_body.encode("utf-8")).hexdigest(),
        "instances_sha256": hashlib.sha256(coco_body.encode("utf-8")).hexdigest(),
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose annotated single breads over real empty operational backgrounds"
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-image", action="append", dest="source_images")
    parser.add_argument("--background-image", action="append", dest="background_images")
    parser.add_argument("--image-count", type=int, default=100)
    parser.add_argument("--minimum-objects", type=int, default=2)
    parser.add_argument("--maximum-objects", type=int, default=4)
    parser.add_argument("--minimum-source-scale", type=float, default=0.82)
    parser.add_argument("--maximum-source-scale", type=float, default=1.05)
    parser.add_argument("--maximum-rotation-degrees", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    metadata = generate_operational_composites(
        args.annotations,
        args.output_root,
        seed=args.seed,
        source_image_names=args.source_images,
        background_image_names=args.background_images,
        recipe=OperationalCompositeRecipe(
            image_count=args.image_count,
            minimum_objects=args.minimum_objects,
            maximum_objects=args.maximum_objects,
            minimum_source_scale=args.minimum_source_scale,
            maximum_source_scale=args.maximum_source_scale,
            maximum_rotation_degrees=args.maximum_rotation_degrees,
        ),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
