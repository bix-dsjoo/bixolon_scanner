from __future__ import annotations

import hashlib
import io
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class SyntheticRecipe:
    frame_width: int = 1280
    frame_height: int = 720
    object_scale_min: float = 0.20
    object_scale_max: float = 0.48
    rotation_degrees: float = 180.0
    perspective_fraction: float = 0.06
    brightness_min: float = 0.80
    brightness_max: float = 1.20
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    saturation_min: float = 0.85
    saturation_max: float = 1.15
    blur_probability: float = 0.20
    blur_radius_max: float = 0.8
    jpeg_quality_min: int = 82
    jpeg_quality_max: int = 96
    white_transparent_threshold: int = 8
    white_opaque_threshold: int = 45

    def validate(self) -> None:
        if self.frame_width < 64 or self.frame_height < 64:
            raise ValueError("synthetic frame dimensions must be at least 64 pixels")
        if not 0 < self.object_scale_min <= self.object_scale_max <= 0.95:
            raise ValueError("synthetic object scale range is invalid")
        for name, value in (
            ("brightness_min", self.brightness_min),
            ("contrast_min", self.contrast_min),
            ("saturation_min", self.saturation_min),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.blur_probability <= 1.0:
            raise ValueError("blur_probability must be between zero and one")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("JPEG quality range is invalid")
        if not 0 <= self.white_transparent_threshold < self.white_opaque_threshold <= 255:
            raise ValueError("white-background alpha thresholds are invalid")


@dataclass(frozen=True)
class SyntheticSample:
    image: Image.Image
    bbox_xyxy: tuple[int, int, int, int]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DirectRoiRecipe:
    """Classifier-only augmentation; it never creates detector input."""

    output_size: int = 224
    canvas_scale_min: float = 0.62
    canvas_scale_max: float = 0.94
    rotation_degrees: float = 25.0
    perspective_fraction: float = 0.04
    brightness_min: float = 0.80
    brightness_max: float = 1.20
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    saturation_min: float = 0.85
    saturation_max: float = 1.15
    blur_probability: float = 0.15
    blur_radius_max: float = 0.7
    jpeg_quality_min: int = 82
    jpeg_quality_max: int = 96
    white_transparent_threshold: int = 8
    white_opaque_threshold: int = 45
    crop_mode: str = "white_alpha_composite"
    padding_ratio: float = 0.0
    border_color_distance: int = 42
    mask_feather_radius: float = 0.8
    procedural_gradient: bool = False
    procedural_shadow: bool = False

    def validate(self) -> None:
        if self.output_size < 64:
            raise ValueError("ROI output_size must be at least 64")
        if not 0 < self.canvas_scale_min <= self.canvas_scale_max <= 1:
            raise ValueError("ROI canvas scale range is invalid")
        if not 0 <= self.blur_probability <= 1:
            raise ValueError("ROI blur_probability is invalid")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("ROI JPEG quality range is invalid")
        if self.crop_mode not in {
            "white_alpha_composite",
            "padded_letterbox",
            "border_connected_composite",
        }:
            raise ValueError("unsupported direct ROI crop mode")
        if not 0.0 <= self.padding_ratio <= 0.25:
            raise ValueError("ROI padding_ratio must be in [0, 0.25]")
        if not 1 <= self.border_color_distance <= 255:
            raise ValueError("border_color_distance must be in [1, 255]")
        if self.mask_feather_radius < 0:
            raise ValueError("mask_feather_radius cannot be negative")


def direct_roi_recipe_sha256(recipe: DirectRoiRecipe) -> str:
    recipe.validate()
    return hashlib.sha256(_canonical_json(asdict(recipe)).encode("utf-8")).hexdigest()


def augment_direct_roi(
    support_image: Image.Image,
    *,
    source_sha256: str,
    category_id: int,
    seed: int,
    recipe: DirectRoiRecipe,
    prepared_cutout: Image.Image | None = None,
) -> SyntheticSample:
    """Make a known-foreground ROI without invoking or imitating the detector."""
    recipe.validate()
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a full SHA-256 digest")
    rng = np.random.default_rng(seed)
    cutout = (
        prepare_direct_roi_source(support_image, recipe)
        if prepared_cutout is None
        else prepared_cutout.convert("RGBA").copy()
    )
    cutout = cutout.crop(_nonempty_alpha_bbox(cutout))
    alpha = cutout.getchannel("A")
    brightness = float(rng.uniform(recipe.brightness_min, recipe.brightness_max))
    contrast = float(rng.uniform(recipe.contrast_min, recipe.contrast_max))
    saturation = float(rng.uniform(recipe.saturation_min, recipe.saturation_max))
    rgb = ImageEnhance.Brightness(cutout.convert("RGB")).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    cutout = rgb.convert("RGBA")
    cutout.putalpha(alpha)
    cutout, perspective = _perspective_transform(cutout, recipe.perspective_fraction, rng)
    rotation = float(rng.uniform(-recipe.rotation_degrees, recipe.rotation_degrees))
    cutout = cutout.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    cutout = cutout.crop(_nonempty_alpha_bbox(cutout))
    scale = float(rng.uniform(recipe.canvas_scale_min, recipe.canvas_scale_max))
    if recipe.padding_ratio:
        scale = min(scale, 1.0 / (1.0 + 2.0 * recipe.padding_ratio))
    long_side = max(1, round(recipe.output_size * scale))
    resize = long_side / max(cutout.size)
    cutout = cutout.resize(
        (max(1, round(cutout.width * resize)), max(1, round(cutout.height * resize))),
        Image.Resampling.LANCZOS,
    )
    if recipe.procedural_gradient:
        canvas, tint = _procedural_roi_background(recipe.output_size, rng)
    else:
        neutral = int(rng.integers(185, 236))
        tint = tuple(
            int(np.clip(neutral + rng.integers(-10, 11), 0, 255)) for _ in range(3)
        )
        canvas = Image.new("RGB", (recipe.output_size, recipe.output_size), tint)
    max_x, max_y = recipe.output_size - cutout.width, recipe.output_size - cutout.height
    left = int(rng.integers(0, max_x + 1))
    top = int(rng.integers(0, max_y + 1))
    if recipe.procedural_shadow and recipe.crop_mode != "padded_letterbox":
        shadow = Image.new("L", canvas.size, 0)
        shadow.paste(
            cutout.getchannel("A"),
            (min(left + 2, recipe.output_size - 1), min(top + 3, recipe.output_size - 1)),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(2.0))
        shade = Image.new(
            "RGB", canvas.size, tuple(max(0, value - 28) for value in tint)
        )
        canvas.paste(shade, (0, 0), shadow.point(lambda value: round(value * 0.22)))
    canvas.paste(cutout.convert("RGB"), (left, top), cutout.getchannel("A"))
    blur_radius = 0.0
    if rng.random() < recipe.blur_probability:
        blur_radius = float(rng.uniform(0.05, recipe.blur_radius_max))
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
    jpeg_quality = int(rng.integers(recipe.jpeg_quality_min, recipe.jpeg_quality_max + 1))
    canvas = _jpeg_roundtrip(canvas, jpeg_quality)
    bbox = (left, top, left + cutout.width, top + cutout.height)
    return SyntheticSample(
        image=canvas,
        bbox_xyxy=bbox,
        provenance={
            "schema_version": "1.0",
            "mode": "ground_truth_foreground_roi",
            "source_sha256": source_sha256,
            "category_id": category_id,
            "seed": seed,
            "recipe_sha256": direct_roi_recipe_sha256(recipe),
            "background_id": "procedural-neutral",
            "bbox_xyxy": list(bbox),
            "parameters": {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "perspective": perspective,
                "rotation_degrees": rotation,
                "crop_mode": recipe.crop_mode,
                "padding_ratio": recipe.padding_ratio,
                "canvas_scale": scale,
                "position": [left, top],
                "blur_radius": blur_radius,
                "jpeg_quality": jpeg_quality,
            },
        },
    )


def prepare_direct_roi_source(
    support_image: Image.Image, recipe: DirectRoiRecipe
) -> Image.Image:
    """Extract the deterministic foreground once for all views of a source."""
    recipe.validate()
    if recipe.crop_mode == "border_connected_composite":
        cutout = border_connected_background_alpha(
            support_image,
            color_distance=recipe.border_color_distance,
            feather_radius=recipe.mask_feather_radius,
        )
    elif recipe.crop_mode == "padded_letterbox":
        cutout = ImageOps.exif_transpose(support_image).convert("RGBA")
    else:
        cutout = white_background_alpha(
            support_image,
            transparent_threshold=recipe.white_transparent_threshold,
            opaque_threshold=recipe.white_opaque_threshold,
        )
    return cutout.crop(_nonempty_alpha_bbox(cutout))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def recipe_sha256(recipe: SyntheticRecipe) -> str:
    recipe.validate()
    return hashlib.sha256(_canonical_json(asdict(recipe)).encode("utf-8")).hexdigest()


def white_background_alpha(
    image: Image.Image,
    *,
    transparent_threshold: int,
    opaque_threshold: int,
) -> Image.Image:
    if not 0 <= transparent_threshold < opaque_threshold <= 255:
        raise ValueError("white-background alpha thresholds are invalid")
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    values = np.asarray(rgba, dtype=np.uint8)
    existing_alpha = values[..., 3].astype(np.float32) / 255.0
    rgb = values[..., :3].astype(np.int16)
    distance = (255 - rgb.min(axis=-1)).astype(np.float32)
    inferred = np.clip(
        (distance - transparent_threshold) / (opaque_threshold - transparent_threshold),
        0.0,
        1.0,
    )
    alpha = np.rint(existing_alpha * inferred * 255.0).astype(np.uint8)
    result = values.copy()
    result[..., 3] = alpha
    return Image.fromarray(result, mode="RGBA")


def border_connected_background_alpha(
    image: Image.Image,
    *,
    color_distance: int = 42,
    feather_radius: float = 0.8,
) -> Image.Image:
    """Remove only background-colored pixels connected to the image border."""
    if not 1 <= color_distance <= 255:
        raise ValueError("color_distance must be in [1, 255]")
    if feather_radius < 0:
        raise ValueError("feather_radius cannot be negative")
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    values = np.asarray(rgba, dtype=np.uint8)
    rgb = values[..., :3].astype(np.int16)
    height, width = rgb.shape[:2]
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background_color = np.median(border, axis=0)
    distance = np.max(np.abs(rgb - background_color[None, None, :]), axis=-1)
    candidate = distance <= color_distance
    connected = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if candidate[0, x]:
            queue.append((0, x))
        if candidate[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if candidate[y, 0]:
            queue.append((y, 0))
        if candidate[y, width - 1]:
            queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if connected[y, x] or not candidate[y, x]:
            continue
        connected[y, x] = True
        if y:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))
    alpha = np.where(connected, 0, values[..., 3]).astype(np.uint8)
    alpha_image = Image.fromarray(alpha, mode="L")
    if feather_radius:
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(feather_radius))
    result = rgba.copy()
    result.putalpha(alpha_image)
    return result


def _procedural_roi_background(
    size: int, rng: np.random.Generator
) -> tuple[Image.Image, tuple[int, int, int]]:
    neutral = int(rng.integers(185, 236))
    tint = tuple(int(np.clip(neutral + rng.integers(-10, 11), 0, 255)) for _ in range(3))
    yy, xx = np.mgrid[0:size, 0:size]
    direction_x = float(rng.uniform(-8.0, 8.0))
    direction_y = float(rng.uniform(-8.0, 8.0))
    gradient = direction_x * (xx / max(size - 1, 1) - 0.5)
    gradient += direction_y * (yy / max(size - 1, 1) - 0.5)
    noise = rng.normal(0.0, 1.25, size=(size, size))
    array = np.empty((size, size, 3), dtype=np.uint8)
    for channel, value in enumerate(tint):
        array[..., channel] = np.clip(value + gradient + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB"), tint


def _nonempty_alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("support cutout contains no foreground after mask extraction")
    return bbox


def _prepare_background(
    background: Image.Image | None, recipe: SyntheticRecipe, rng: np.random.Generator
) -> tuple[Image.Image, str]:
    size = (recipe.frame_width, recipe.frame_height)
    if background is None:
        base = int(rng.integers(180, 236))
        tint = tuple(
            int(np.clip(base + int(rng.integers(-12, 13)), 0, 255)) for _ in range(3)
        )
        return Image.new("RGB", size, tint), "procedural-neutral"
    normalized = ImageOps.exif_transpose(background).convert("RGB")
    fitted = ImageOps.fit(normalized, size, method=Image.Resampling.LANCZOS)
    digest = hashlib.sha256(fitted.tobytes()).hexdigest()
    return fitted, f"pixels:{digest}"


def _perspective_transform(
    image: Image.Image, fraction: float, rng: np.random.Generator
) -> tuple[Image.Image, list[float]]:
    if fraction <= 0:
        return image, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # PIL expects the inverse output-to-input mapping. Small projective terms
    # produce camera-like skew while keeping the foreground identity intact.
    horizontal = float(rng.uniform(-fraction, fraction)) / max(image.width, 1)
    vertical = float(rng.uniform(-fraction, fraction)) / max(image.height, 1)
    coefficients = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, horizontal, vertical]
    transformed = image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
    )
    return transformed, coefficients


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="JPEG", quality=quality)
    stream.seek(0)
    with Image.open(stream) as source:
        return source.convert("RGB")


def compose_synthetic_frame(
    support_image: Image.Image,
    *,
    source_sha256: str,
    category_id: int,
    seed: int,
    recipe: SyntheticRecipe,
    background: Image.Image | None = None,
    background_id: str | None = None,
) -> SyntheticSample:
    recipe.validate()
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a full SHA-256 digest")
    if category_id < 1:
        raise ValueError("category_id must be positive")
    rng = np.random.default_rng(seed)
    frame, inferred_background_id = _prepare_background(background, recipe, rng)
    cutout = white_background_alpha(
        support_image,
        transparent_threshold=recipe.white_transparent_threshold,
        opaque_threshold=recipe.white_opaque_threshold,
    )
    cutout = cutout.crop(_nonempty_alpha_bbox(cutout))
    cutout_alpha = cutout.getchannel("A")

    brightness = float(rng.uniform(recipe.brightness_min, recipe.brightness_max))
    contrast = float(rng.uniform(recipe.contrast_min, recipe.contrast_max))
    saturation = float(rng.uniform(recipe.saturation_min, recipe.saturation_max))
    rgb = ImageEnhance.Brightness(cutout.convert("RGB")).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    cutout = rgb.convert("RGBA")
    cutout.putalpha(cutout_alpha)
    cutout, perspective = _perspective_transform(
        cutout, recipe.perspective_fraction, rng
    )
    rotation = float(rng.uniform(-recipe.rotation_degrees, recipe.rotation_degrees))
    cutout = cutout.rotate(
        rotation, resample=Image.Resampling.BICUBIC, expand=True
    )
    cutout = cutout.crop(_nonempty_alpha_bbox(cutout))

    scale = float(rng.uniform(recipe.object_scale_min, recipe.object_scale_max))
    target_long_side = max(1, int(round(min(recipe.frame_width, recipe.frame_height) * scale)))
    resize_scale = target_long_side / max(cutout.width, cutout.height)
    resized = cutout.resize(
        (
            max(1, int(round(cutout.width * resize_scale))),
            max(1, int(round(cutout.height * resize_scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    max_x = recipe.frame_width - resized.width
    max_y = recipe.frame_height - resized.height
    if max_x < 0 or max_y < 0:
        raise ValueError("synthetic object does not fit inside the frame")
    left = int(rng.integers(0, max_x + 1))
    top = int(rng.integers(0, max_y + 1))
    frame.paste(resized.convert("RGB"), (left, top), resized.getchannel("A"))

    blurred = bool(rng.random() < recipe.blur_probability)
    blur_radius = (
        float(rng.uniform(0.05, recipe.blur_radius_max)) if blurred else 0.0
    )
    if blurred:
        frame = frame.filter(ImageFilter.GaussianBlur(blur_radius))
    jpeg_quality = int(
        rng.integers(recipe.jpeg_quality_min, recipe.jpeg_quality_max + 1)
    )
    frame = _jpeg_roundtrip(frame, jpeg_quality)
    bbox = (left, top, left + resized.width, top + resized.height)
    provenance = {
        "schema_version": "1.0",
        "source_sha256": source_sha256,
        "category_id": category_id,
        "seed": int(seed),
        "recipe_sha256": recipe_sha256(recipe),
        "background_id": background_id or inferred_background_id,
        "parameters": {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "perspective": perspective,
            "rotation_degrees": rotation,
            "object_scale": scale,
            "position": [left, top],
            "blur_radius": blur_radius,
            "jpeg_quality": jpeg_quality,
        },
        "bbox_xyxy": list(bbox),
    }
    return SyntheticSample(image=frame, bbox_xyxy=bbox, provenance=provenance)


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection_left = max(float(left[0]), float(right[0]))
    intersection_top = max(float(left[1]), float(right[1]))
    intersection_right = min(float(left[2]), float(right[2]))
    intersection_bottom = min(float(left[3]), float(right[3]))
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def validate_single_detector_crop(
    detections: Sequence[Any],
    *,
    expected_bbox_xyxy: Sequence[float],
    hard_reasons: Sequence[str],
    minimum_iou: float,
) -> Any:
    if hard_reasons:
        raise ValueError(f"synthetic frame failed detector hard gate: {list(hard_reasons)}")
    if len(detections) != 1:
        raise ValueError(
            f"synthetic frame requires exactly one detection, got {len(detections)}"
        )
    detection = detections[0]
    actual = (detection.x1, detection.y1, detection.x2, detection.y2)
    overlap = _iou(actual, expected_bbox_xyxy)
    if overlap < minimum_iou:
        raise ValueError(
            f"synthetic detector IoU {overlap:.6f} is below {minimum_iou:.6f}"
        )
    return detection


def crop_with_margin(
    image: Image.Image, bbox_xyxy: Sequence[float], *, margin_ratio: float
) -> Image.Image:
    if margin_ratio < 0:
        raise ValueError("crop margin ratio must be non-negative")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("crop bbox is empty")
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    bounds = (
        max(0, int(np.floor(x1 - margin_x))),
        max(0, int(np.floor(y1 - margin_y))),
        min(image.width, int(np.ceil(x2 + margin_x))),
        min(image.height, int(np.ceil(y2 + margin_y))),
    )
    result = image.crop(bounds)
    if result.width == 0 or result.height == 0:
        raise ValueError("crop is empty after clamping")
    return result
