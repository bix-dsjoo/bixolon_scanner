from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

from .operational_compositor import extract_annotated_cutout

GRID_POSITIONS = (
    "top_left",
    "top_right",
    "center",
    "bottom_left",
    "bottom_right",
)
SIDES = ("normal", "flipped")
POSITION_ANCHORS: dict[str, tuple[float, float]] = {
    "top_left": (0.24, 0.26),
    "top_right": (0.76, 0.26),
    "center": (0.50, 0.51),
    "bottom_left": (0.24, 0.76),
    "bottom_right": (0.76, 0.76),
}
SCENARIOS = ("NORMAL", "DENSE", "OCCLUSION", "EDGE", "LIGHTING", "RECAPTURE")
RECAPTURE_SUBTYPES = (
    "EMPTY_TRAY",
    "SEVERE_BLUR",
    "OVEREXPOSED",
    "UNDEREXPOSED",
    "SEVERE_OCCLUSION",
)
CLASS_DIRECTORY_PATTERN = re.compile(r"^bread_(?P<category>\d{2})(?:_(?P<name>.+))?$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SOURCE_DATASET = "bread-grid-20x2x5"
POSE_SOURCE_DATASET = "bread-pose-cutouts-20x2x5"
POSE_VIEW_TO_POSITION = {
    "ground_30_dir_01": "top_left",
    "ground_30_dir_02": "top_right",
    "vertical": "center",
    "ground_30_dir_03": "bottom_left",
    "ground_30_dir_04": "bottom_right",
}


@dataclass(frozen=True)
class GridScenePlan:
    normal: int = 600
    dense: int = 300
    occlusion: int = 225
    edge: int = 150
    lighting: int = 125
    recapture: int = 100

    def validate(self) -> None:
        values = asdict(self)
        if any(int(value) < 0 for value in values.values()):
            raise ValueError("scene counts cannot be negative")
        if self.total < 1:
            raise ValueError("scene plan must generate at least one image")

    @property
    def total(self) -> int:
        return sum(int(value) for value in asdict(self).values())

    def expanded(self) -> list[str]:
        self.validate()
        return [
            *("NORMAL" for _ in range(self.normal)),
            *("DENSE" for _ in range(self.dense)),
            *("OCCLUSION" for _ in range(self.occlusion)),
            *("EDGE" for _ in range(self.edge)),
            *("LIGHTING" for _ in range(self.lighting)),
            *("RECAPTURE" for _ in range(self.recapture)),
        ]


@dataclass(frozen=True)
class GridCompositeRecipe:
    plan: GridScenePlan = field(default_factory=GridScenePlan)
    tray_roi_xyxy: tuple[float, float, float, float] = (0.06, 0.08, 0.94, 0.95)
    minimum_objects_normal: int = 2
    maximum_objects_normal: int = 4
    minimum_objects_dense: int = 5
    maximum_objects_dense: int = 8
    minimum_objects_occlusion: int = 3
    maximum_objects_occlusion: int = 6
    minimum_occlusion_fraction: float = 0.10
    maximum_occlusion_fraction: float = 0.30
    minimum_visibility_fraction: float = 0.55
    maximum_rotation_jitter_degrees: float = 18.0
    placement_attempts: int = 240
    scene_attempts: int = 30
    mask_transparent_distance: int = 16
    mask_opaque_distance: int = 48
    mask_feather_radius: float = 0.8
    shadow_probability: float = 0.92
    jpeg_quality_min: int = 88
    jpeg_quality_max: int = 97

    def validate(self) -> None:
        self.plan.validate()
        left, top, right, bottom = self.tray_roi_xyxy
        if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
            raise ValueError("tray_roi_xyxy must be normalized and non-empty")
        for minimum, maximum, name in (
            (self.minimum_objects_normal, self.maximum_objects_normal, "normal object count"),
            (self.minimum_objects_dense, self.maximum_objects_dense, "dense object count"),
            (
                self.minimum_objects_occlusion,
                self.maximum_objects_occlusion,
                "occlusion object count",
            ),
        ):
            if not 1 <= minimum <= maximum <= 20:
                raise ValueError(f"invalid {name} range")
        if not 0 < self.minimum_occlusion_fraction <= self.maximum_occlusion_fraction < 1:
            raise ValueError("invalid occlusion fraction range")
        if not 0 < self.minimum_visibility_fraction <= 1:
            raise ValueError("minimum_visibility_fraction must be in (0, 1]")
        if self.maximum_rotation_jitter_degrees < 0:
            raise ValueError("maximum_rotation_jitter_degrees cannot be negative")
        if self.placement_attempts < 1 or self.scene_attempts < 1:
            raise ValueError("placement and scene attempts must be positive")
        if not 0 <= self.mask_transparent_distance < self.mask_opaque_distance <= 255:
            raise ValueError("invalid mask distance thresholds")
        if self.mask_feather_radius < 0:
            raise ValueError("mask_feather_radius cannot be negative")
        if not 0 <= self.shadow_probability <= 1:
            raise ValueError("shadow_probability must be in [0, 1]")
        if not 1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("invalid JPEG quality range")


@dataclass(frozen=True)
class CaptureAsset:
    category_id: int
    class_name: str
    side: str
    grid_position: str
    path: Path
    relative_path: str
    sha256: str
    width: int
    height: int
    physical_item_id: str
    capture_session_id: str
    capture_view: str = ""
    extraction_mode: str = "aligned_background"


@dataclass(frozen=True)
class CaptureGrid:
    root: Path
    background_path: Path
    background_relative_path: str
    background_sha256: str
    width: int
    height: int
    assets: tuple[CaptureAsset, ...]
    contract: str = "bread-grid-20x2x5"
    source_dataset: str = SOURCE_DATASET


@dataclass(frozen=True)
class LoadedAsset:
    source: CaptureAsset
    cutout: Image.Image
    foreground_area: int


@dataclass
class CandidateObject:
    category_id: int
    side: str
    target_center: tuple[float, float]
    asset: LoadedAsset
    cutout: Image.Image
    transform: dict[str, Any]


@dataclass
class PlacedObject:
    candidate: CandidateObject
    left: int
    top: int
    full_area: int
    in_frame_mask: np.ndarray


@dataclass
class GeneratedScene:
    image: Image.Image
    annotations: list[dict[str, Any]]
    object_provenance: list[dict[str, Any]]
    scenario: str
    subtype: str | None
    condition_tags: list[str]
    expected_status: str
    expected_reason_codes: list[str]
    scene_parameters: dict[str, Any]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recipe_sha256(recipe: GridCompositeRecipe) -> str:
    recipe.validate()
    return hashlib.sha256(_canonical_json(asdict(recipe)).encode("utf-8")).hexdigest()


def _image_identity(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as source:
            normalized = ImageOps.exif_transpose(source)
            width, height = normalized.size
            normalized.load()
    except Exception as error:
        raise ValueError(f"cannot decode capture image: {path.name}") from error
    return width, height, _sha256_file(path)


def _find_named_image(directory: Path, stem: str) -> Path:
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem == stem
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {stem} image in {directory.name}")
    return matches[0]


def discover_capture_grid(capture_root: Path) -> CaptureGrid:
    """Validate the fixed 20 x 2 x 5 capture tree and lock all source hashes."""
    root = capture_root.resolve()
    if not root.is_dir():
        raise ValueError(f"capture root is not a directory: {capture_root}")
    background_matches = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem == "background"
    )
    if len(background_matches) != 1:
        raise ValueError("capture root must contain exactly one background image")
    background_path = background_matches[0]
    root_images = {
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if root_images != {background_path.resolve()}:
        unexpected = sorted(path.name for path in root_images - {background_path.resolve()})
        raise ValueError(f"unexpected root-level capture images: {unexpected}")
    width, height, background_sha256 = _image_identity(background_path)

    directories: dict[int, tuple[Path, str]] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = CLASS_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected capture directory: {path.name}")
        category_id = int(match.group("category"))
        if category_id in directories:
            raise ValueError(f"duplicate category directory: {category_id}")
        name = (match.group("name") or f"bread_{category_id:02d}").replace("_", " ")
        directories[category_id] = (path, name)
    if set(directories) != set(range(1, 21)):
        raise ValueError("capture root must contain bread_01 through bread_20 directories")

    assets: list[CaptureAsset] = []
    seen_hashes: set[str] = {background_sha256}
    for category_id in range(1, 21):
        directory, class_name = directories[category_id]
        expected_stems = {f"{side}_{position}" for side in SIDES for position in GRID_POSITIONS}
        actual_images = {
            path.stem
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if actual_images != expected_stems:
            raise ValueError(
                f"{directory.name} capture names mismatch: "
                f"missing={sorted(expected_stems - actual_images)}, "
                f"unexpected={sorted(actual_images - expected_stems)}"
            )
        for side in SIDES:
            for position in GRID_POSITIONS:
                path = _find_named_image(directory, f"{side}_{position}")
                image_width, image_height, digest = _image_identity(path)
                if (image_width, image_height) != (width, height):
                    raise ValueError(f"capture dimensions differ from background: {path.name}")
                if digest in seen_hashes:
                    raise ValueError(f"duplicate capture pixels: {path.name}")
                seen_hashes.add(digest)
                assets.append(
                    CaptureAsset(
                        category_id=category_id,
                        class_name=class_name,
                        side=side,
                        grid_position=position,
                        path=path,
                        relative_path=path.relative_to(root).as_posix(),
                        sha256=digest,
                        width=width,
                        height=height,
                        physical_item_id=f"bread_{category_id:02d}:capture_item",
                        capture_session_id=f"bread_{category_id:02d}:grid_capture",
                        capture_view=position,
                    )
                )
    if len(assets) != 200:
        raise RuntimeError("validated capture grid must contain exactly 200 bread images")
    return CaptureGrid(
        root=root,
        background_path=background_path,
        background_relative_path=background_path.relative_to(root).as_posix(),
        background_sha256=background_sha256,
        width=width,
        height=height,
        assets=tuple(assets),
    )


def discover_pose_cutouts(capture_root: Path, background_image: Path) -> CaptureGrid:
    """Validate 20 x 2 x (four directions + vertical) white-background cutouts."""
    root = capture_root.resolve()
    if not root.is_dir():
        raise ValueError(f"capture root is not a directory: {capture_root}")
    background_path = background_image.resolve()
    if not background_path.is_file():
        raise ValueError(f"background image does not exist: {background_image}")
    width, height, background_sha256 = _image_identity(background_path)

    directories: dict[int, tuple[Path, str]] = {}
    for path in root.iterdir():
        if not path.is_dir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                raise ValueError(f"unexpected root-level capture image: {path.name}")
            continue
        match = CLASS_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected capture directory: {path.name}")
        category_id = int(match.group("category"))
        if category_id in directories:
            raise ValueError(f"duplicate category directory: {category_id}")
        name = (match.group("name") or f"bread_{category_id:02d}").replace("_", " ")
        directories[category_id] = (path, name)
    if set(directories) != set(range(1, 21)):
        raise ValueError("capture root must contain bread_01 through bread_20 directories")

    assets: list[CaptureAsset] = []
    seen_hashes: set[str] = {background_sha256}
    for category_id in range(1, 21):
        directory, class_name = directories[category_id]
        expected_stems = {
            f"bread_{category_id:02d}_{side}_{view}"
            for side in SIDES
            for view in POSE_VIEW_TO_POSITION
        }
        actual_images = {
            path.stem
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if actual_images != expected_stems:
            raise ValueError(
                f"{directory.name} capture names mismatch: "
                f"missing={sorted(expected_stems - actual_images)}, "
                f"unexpected={sorted(actual_images - expected_stems)}"
            )
        for side in SIDES:
            for view, position in POSE_VIEW_TO_POSITION.items():
                stem = f"bread_{category_id:02d}_{side}_{view}"
                path = _find_named_image(directory, stem)
                image_width, image_height, digest = _image_identity(path)
                if digest in seen_hashes:
                    raise ValueError(f"duplicate capture pixels: {path.name}")
                seen_hashes.add(digest)
                assets.append(
                    CaptureAsset(
                        category_id=category_id,
                        class_name=class_name,
                        side=side,
                        grid_position=position,
                        path=path,
                        relative_path=path.relative_to(root).as_posix(),
                        sha256=digest,
                        width=image_width,
                        height=image_height,
                        physical_item_id=f"bread_{category_id:02d}:capture_item",
                        capture_session_id=f"bread_{category_id:02d}:pose_cutout_capture",
                        capture_view=view,
                        extraction_mode="white_background_cutout",
                    )
                )
    if len(assets) != 200:
        raise RuntimeError("validated pose collection must contain exactly 200 bread images")
    return CaptureGrid(
        root=root,
        background_path=background_path,
        background_relative_path=f"external-background/{background_path.name}",
        background_sha256=background_sha256,
        width=width,
        height=height,
        assets=tuple(assets),
        contract="bread-pose-cutouts-20x2x5",
        source_dataset=POSE_SOURCE_DATASET,
    )


def _capture_window(position: str, width: int, height: int) -> tuple[float, float, float, float]:
    anchor_x, anchor_y = POSITION_ANCHORS[position]
    window_width = width * 0.78
    window_height = height * 0.78
    left = max(0.0, min(width - window_width, anchor_x * width - window_width * 0.5))
    top = max(0.0, min(height - window_height, anchor_y * height - window_height * 0.5))
    return left, top, window_width, window_height


def _extract_white_background_cutout(
    source_image: Image.Image,
    *,
    transparent_distance: int,
    opaque_distance: int,
    feather_radius: float,
) -> Image.Image:
    source = ImageOps.exif_transpose(source_image).convert("RGB")
    source_array = np.asarray(source, dtype=np.int16)
    border = np.concatenate(
        (source_array[0], source_array[-1], source_array[:, 0], source_array[:, -1]), axis=0
    )
    background_color = np.median(border, axis=0)
    distance = np.max(np.abs(source_array - background_color), axis=-1).astype(np.float32)
    weak = distance >= transparent_distance
    strong = distance >= opaque_distance
    labels, component_count = ndimage.label(weak)
    if component_count < 1:
        raise ValueError("foreground mask is empty after white-background removal")
    component_ids = np.arange(1, component_count + 1)
    strong_counts = ndimage.sum(strong, labels=labels, index=component_ids)
    area_counts = ndimage.sum(weak, labels=labels, index=component_ids)
    best_index = max(
        range(component_count),
        key=lambda index: (float(strong_counts[index]), float(area_counts[index])),
    )
    connected = labels == int(component_ids[best_index])
    if not np.any(strong & connected):
        raise ValueError("white-background foreground has no opaque seed")
    alpha = np.clip(
        (distance - transparent_distance) / max(opaque_distance - transparent_distance, 1),
        0.0,
        1.0,
    )
    alpha = np.rint(alpha * connected * 255.0).astype(np.uint8)
    alpha_image = Image.fromarray(alpha, mode="L")
    if feather_radius:
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(feather_radius))
    rgba = source.convert("RGBA")
    rgba.putalpha(alpha_image)
    visible = alpha_image.point(lambda value: 255 if value >= 4 else 0).getbbox()
    if visible is None:
        raise ValueError("foreground mask disappeared after feathering")
    return rgba.crop(visible)


def _load_assets(
    grid: CaptureGrid, recipe: GridCompositeRecipe
) -> tuple[Image.Image, dict[tuple[int, str, str], LoadedAsset]]:
    with Image.open(grid.background_path) as opened:
        background = ImageOps.exif_transpose(opened).convert("RGB")
    result: dict[tuple[int, str, str], LoadedAsset] = {}
    for source in grid.assets:
        with Image.open(source.path) as opened:
            if source.extraction_mode == "white_background_cutout":
                cutout = _extract_white_background_cutout(
                    opened,
                    transparent_distance=recipe.mask_transparent_distance,
                    opaque_distance=recipe.mask_opaque_distance,
                    feather_radius=recipe.mask_feather_radius,
                )
            else:
                cutout = extract_annotated_cutout(
                    opened,
                    background,
                    bbox_xywh=_capture_window(
                        source.grid_position,
                        source.width,
                        source.height,
                    ),
                    transparent_distance=recipe.mask_transparent_distance,
                    opaque_distance=recipe.mask_opaque_distance,
                    feather_radius=recipe.mask_feather_radius,
                )
        foreground_area = int(np.count_nonzero(np.asarray(cutout.getchannel("A")) >= 24))
        if foreground_area < max(64, int(grid.width * grid.height * 0.003)):
            raise ValueError(f"extracted foreground is implausibly small: {source.relative_path}")
        result[(source.category_id, source.side, source.grid_position)] = LoadedAsset(
            source=source,
            cutout=cutout,
            foreground_area=foreground_area,
        )
    return background, result


def _nearest_grid_position(center: tuple[float, float]) -> str:
    return min(
        GRID_POSITIONS,
        key=lambda name: (
            (center[0] - POSITION_ANCHORS[name][0]) ** 2
            + (center[1] - POSITION_ANCHORS[name][1]) ** 2
        ),
    )


def _patch_luminance(image: Image.Image, center: tuple[float, float]) -> float:
    array = np.asarray(image, dtype=np.float32)
    x = int(round(center[0] * (image.width - 1)))
    y = int(round(center[1] * (image.height - 1)))
    radius = max(4, min(image.size) // 32)
    patch = array[
        max(0, y - radius) : min(image.height, y + radius + 1),
        max(0, x - radius) : min(image.width, x + radius + 1),
    ]
    return float(np.mean(0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]))


def _scenario_scale_range(scenario: str) -> tuple[float, float]:
    if scenario == "DENSE":
        return 0.58, 0.80
    if scenario == "OCCLUSION":
        return 0.70, 0.94
    if scenario == "EDGE":
        return 0.78, 1.02
    return 0.80, 1.03


def _trim_alpha(image: Image.Image) -> Image.Image:
    bbox = image.getchannel("A").point(lambda value: 255 if value >= 4 else 0).getbbox()
    if bbox is None:
        raise ValueError("transformed cutout has no visible foreground")
    return image.crop(bbox)


def _transform_asset(
    asset: LoadedAsset,
    *,
    background: Image.Image,
    target_center: tuple[float, float],
    scenario: str,
    scene_object_count: int,
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> tuple[Image.Image, dict[str, Any]]:
    source = asset.cutout
    alpha = source.getchannel("A")
    source_luminance = _patch_luminance(background, POSITION_ANCHORS[asset.source.grid_position])
    target_luminance = _patch_luminance(background, target_center)
    harmonization = float(np.clip(target_luminance / max(source_luminance, 1.0), 0.92, 1.08))
    brightness = harmonization * float(rng.uniform(0.96, 1.04))
    contrast = float(rng.uniform(0.96, 1.04))
    saturation = float(rng.uniform(0.95, 1.05))
    rgb = ImageEnhance.Brightness(source.convert("RGB")).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    transformed = rgb.convert("RGBA")
    transformed.putalpha(alpha)

    minimum_scale, maximum_scale = _scenario_scale_range(scenario)
    source_scale = float(rng.uniform(minimum_scale, maximum_scale))
    native_size_normalization = 1.0
    if asset.source.extraction_mode == "white_background_cutout":
        maximum_fraction_by_count = {
            2: 0.54,
            3: 0.47,
            4: 0.40,
            5: 0.36,
            6: 0.32,
            7: 0.29,
            8: 0.27,
        }
        maximum_fraction = maximum_fraction_by_count.get(scene_object_count, 0.52)
        maximum_native_dimension = min(background.size) * maximum_fraction
        native_size_normalization = min(
            1.0,
            maximum_native_dimension / max(transformed.size),
        )
    applied_scale = source_scale * native_size_normalization
    transformed = transformed.resize(
        (
            max(2, int(round(transformed.width * applied_scale))),
            max(2, int(round(transformed.height * applied_scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    rotation = float(
        rng.uniform(
            -recipe.maximum_rotation_jitter_degrees,
            recipe.maximum_rotation_jitter_degrees,
        )
    )
    transformed = transformed.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    transformed = _trim_alpha(transformed)
    return transformed, {
        "source_scale": source_scale,
        "native_size_normalization": native_size_normalization,
        "applied_scale": applied_scale,
        "rotation_jitter_degrees": rotation,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "local_luminance_harmonization": harmonization,
        "source_grid_position": asset.source.grid_position,
        "source_capture_view": asset.source.capture_view,
        "target_center_normalized": [target_center[0], target_center[1]],
    }


def _balanced_categories(
    category_counts: Counter[int], count: int, rng: np.random.Generator
) -> list[int]:
    categories = list(range(1, 21))
    rng.shuffle(categories)
    categories.sort(key=lambda category_id: category_counts[category_id])
    return categories[:count]


def _balanced_side(
    category_id: int,
    side_counts: Counter[tuple[int, str]],
    rng: np.random.Generator,
) -> str:
    normal = side_counts[(category_id, "normal")]
    flipped = side_counts[(category_id, "flipped")]
    if normal == flipped:
        return SIDES[int(rng.integers(0, len(SIDES)))]
    return "normal" if normal < flipped else "flipped"


def _tray_roi_pixels(
    recipe: GridCompositeRecipe, canvas_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    left, top, right, bottom = recipe.tray_roi_xyxy
    return (
        int(round(left * canvas_size[0])),
        int(round(top * canvas_size[1])),
        int(round(right * canvas_size[0])),
        int(round(bottom * canvas_size[1])),
    )


def _sample_center(
    scenario: str,
    *,
    index: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if scenario == "DENSE":
        angle = float(rng.uniform(0.0, math.tau))
        radius = float(np.clip(abs(rng.normal(0.14, 0.08)), 0.02, 0.33))
        return (
            float(np.clip(0.5 + math.cos(angle) * radius, 0.12, 0.88)),
            float(np.clip(0.52 + math.sin(angle) * radius, 0.14, 0.88)),
        )
    if scenario == "EDGE" and index == 0:
        edge = int(rng.integers(0, 4))
        along = float(rng.uniform(0.20, 0.80))
        return (
            (0.07 if edge == 0 else 0.93 if edge == 1 else along),
            (0.09 if edge == 2 else 0.94 if edge == 3 else along),
        )
    return float(rng.uniform(0.14, 0.86)), float(rng.uniform(0.16, 0.86))


def _object_count(
    scenario: str,
    subtype: str | None,
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> int:
    if subtype == "EMPTY_TRAY":
        return 0
    if scenario == "DENSE":
        return int(rng.integers(recipe.minimum_objects_dense, recipe.maximum_objects_dense + 1))
    if scenario == "OCCLUSION":
        return int(
            rng.integers(recipe.minimum_objects_occlusion, recipe.maximum_objects_occlusion + 1)
        )
    return int(rng.integers(recipe.minimum_objects_normal, recipe.maximum_objects_normal + 1))


def _build_candidates(
    category_ids: Sequence[int],
    *,
    scenario: str,
    background: Image.Image,
    assets: dict[tuple[int, str, str], LoadedAsset],
    side_counts: Counter[tuple[int, str]],
    view_counts: Counter[tuple[int, str, str]],
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> tuple[
    list[CandidateObject],
    Counter[tuple[int, str]],
    Counter[tuple[int, str, str]],
]:
    local_side_counts = side_counts.copy()
    local_view_counts = view_counts.copy()
    candidates: list[CandidateObject] = []
    for index, category_id in enumerate(category_ids):
        target_center = _sample_center(scenario, index=index, rng=rng)
        side = _balanced_side(category_id, local_side_counts, rng)
        center_asset = assets[(category_id, side, "center")]
        if center_asset.source.extraction_mode == "white_background_cutout":
            positions = list(GRID_POSITIONS)
            rng.shuffle(positions)
            positions.sort(key=lambda name: local_view_counts[(category_id, side, name)])
            position = positions[0]
        else:
            position = _nearest_grid_position(target_center)
        asset = assets[(category_id, side, position)]
        cutout, transform = _transform_asset(
            asset,
            background=background,
            target_center=target_center,
            scenario=scenario,
            scene_object_count=len(category_ids),
            rng=rng,
            recipe=recipe,
        )
        candidates.append(
            CandidateObject(
                category_id=category_id,
                side=side,
                target_center=target_center,
                asset=asset,
                cutout=cutout,
                transform=transform,
            )
        )
        local_side_counts[(category_id, side)] += 1
        local_view_counts[(category_id, side, position)] += 1
    if scenario == "EDGE":
        candidates = [
            candidates[0],
            *sorted(
                candidates[1:],
                key=lambda row: row.cutout.width * row.cutout.height,
                reverse=True,
            ),
        ]
    else:
        candidates.sort(key=lambda row: row.cutout.width * row.cutout.height, reverse=True)
    return candidates, local_side_counts, local_view_counts


def _mask_on_canvas(
    cutout: Image.Image,
    *,
    left: int,
    top: int,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    result = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
    source = np.asarray(cutout.getchannel("A")) >= 24
    destination_left = max(0, left)
    destination_top = max(0, top)
    destination_right = min(canvas_size[0], left + cutout.width)
    destination_bottom = min(canvas_size[1], top + cutout.height)
    if destination_right <= destination_left or destination_bottom <= destination_top:
        return result
    source_left = destination_left - left
    source_top = destination_top - top
    result[destination_top:destination_bottom, destination_left:destination_right] = source[
        source_top : source_top + destination_bottom - destination_top,
        source_left : source_left + destination_right - destination_left,
    ]
    return result


def _mask_overlap_fraction(candidate: np.ndarray, existing: np.ndarray) -> float:
    denominator = max(1, min(int(np.count_nonzero(candidate)), int(np.count_nonzero(existing))))
    return int(np.count_nonzero(candidate & existing)) / denominator


def _position_near_target(
    candidate: CandidateObject,
    *,
    canvas_size: tuple[int, int],
    tray_roi: tuple[int, int, int, int],
    rng: np.random.Generator,
    attempt: int,
) -> tuple[int, int] | None:
    minimum_left, minimum_top, maximum_right, maximum_bottom = tray_roi
    maximum_left = maximum_right - candidate.cutout.width
    maximum_top = maximum_bottom - candidate.cutout.height
    if maximum_left < minimum_left or maximum_top < minimum_top:
        return None
    if attempt == 0:
        center_x = int(round(candidate.target_center[0] * canvas_size[0]))
        center_y = int(round(candidate.target_center[1] * canvas_size[1]))
        left = center_x - candidate.cutout.width // 2
        top = center_y - candidate.cutout.height // 2
    else:
        radius_divisor = 3 if attempt >= 64 else 7
        radius_x = max(4, canvas_size[0] // radius_divisor)
        radius_y = max(4, canvas_size[1] // radius_divisor)
        center_x = int(round(candidate.target_center[0] * canvas_size[0]))
        center_y = int(round(candidate.target_center[1] * canvas_size[1]))
        left = center_x - candidate.cutout.width // 2 + int(rng.integers(-radius_x, radius_x + 1))
        top = center_y - candidate.cutout.height // 2 + int(rng.integers(-radius_y, radius_y + 1))
    return (
        int(np.clip(left, minimum_left, maximum_left)),
        int(np.clip(top, minimum_top, maximum_top)),
    )


def _place_nonoverlapping(
    candidate: CandidateObject,
    *,
    placed: Sequence[PlacedObject],
    canvas_size: tuple[int, int],
    tray_roi: tuple[int, int, int, int],
    maximum_overlap: float,
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> PlacedObject | None:
    full_area = int(np.count_nonzero(np.asarray(candidate.cutout.getchannel("A")) >= 24))
    for attempt in range(recipe.placement_attempts):
        position = _position_near_target(
            candidate,
            canvas_size=canvas_size,
            tray_roi=tray_roi,
            rng=rng,
            attempt=attempt,
        )
        if position is None:
            return None
        left, top = position
        mask = _mask_on_canvas(
            candidate.cutout,
            left=left,
            top=top,
            canvas_size=canvas_size,
        )
        if int(np.count_nonzero(mask)) < full_area * 0.98:
            continue
        if all(
            _mask_overlap_fraction(mask, row.in_frame_mask) <= maximum_overlap for row in placed
        ):
            return PlacedObject(
                candidate=candidate,
                left=left,
                top=top,
                full_area=full_area,
                in_frame_mask=mask,
            )
    return None


def _place_forced_overlap(
    candidate: CandidateObject,
    *,
    target: PlacedObject,
    placed: Sequence[PlacedObject],
    canvas_size: tuple[int, int],
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> PlacedObject | None:
    local_area = int(np.count_nonzero(np.asarray(candidate.cutout.getchannel("A")) >= 24))
    target_y, target_x = np.nonzero(target.in_frame_mask)
    target_center_x = int(round(float(np.mean(target_x))))
    target_center_y = int(round(float(np.mean(target_y))))
    horizontal_radius = (target.candidate.cutout.width + candidate.cutout.width) * 0.32
    vertical_radius = (target.candidate.cutout.height + candidate.cutout.height) * 0.32
    for _ in range(recipe.placement_attempts):
        angle = float(rng.uniform(0.0, math.tau))
        distance_scale = float(rng.uniform(0.65, 1.05))
        center_x = target_center_x + math.cos(angle) * horizontal_radius * distance_scale
        center_y = target_center_y + math.sin(angle) * vertical_radius * distance_scale
        left = int(round(center_x - candidate.cutout.width * 0.5))
        top = int(round(center_y - candidate.cutout.height * 0.5))
        mask = _mask_on_canvas(
            candidate.cutout,
            left=left,
            top=top,
            canvas_size=canvas_size,
        )
        in_frame_area = int(np.count_nonzero(mask))
        if in_frame_area < local_area * 0.98:
            continue
        target_overlap = int(np.count_nonzero(mask & target.in_frame_mask)) / max(
            1, int(np.count_nonzero(target.in_frame_mask))
        )
        if (
            not recipe.minimum_occlusion_fraction
            <= target_overlap
            <= recipe.maximum_occlusion_fraction
        ):
            continue
        if any(
            int(np.count_nonzero(mask & row.in_frame_mask))
            / max(1, int(np.count_nonzero(row.in_frame_mask)))
            > recipe.maximum_occlusion_fraction
            for row in placed
        ):
            continue
        return PlacedObject(
            candidate=candidate,
            left=left,
            top=top,
            full_area=local_area,
            in_frame_mask=mask,
        )
    return None


def _place_edge_object(
    candidate: CandidateObject,
    *,
    canvas_size: tuple[int, int],
    tray_roi: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> tuple[PlacedObject, str] | None:
    full_area = int(np.count_nonzero(np.asarray(candidate.cutout.getchannel("A")) >= 24))
    frame_clipped = bool(rng.random() < 0.5)
    edge = int(rng.integers(0, 4))
    if frame_clipped:
        clipped_fraction = float(rng.uniform(0.10, 0.28))
        if edge == 0:
            left = -int(round(candidate.cutout.width * clipped_fraction))
            top = int(rng.integers(0, max(1, canvas_size[1] - candidate.cutout.height + 1)))
        elif edge == 1:
            left = canvas_size[0] - int(round(candidate.cutout.width * (1.0 - clipped_fraction)))
            top = int(rng.integers(0, max(1, canvas_size[1] - candidate.cutout.height + 1)))
        elif edge == 2:
            left = int(rng.integers(0, max(1, canvas_size[0] - candidate.cutout.width + 1)))
            top = -int(round(candidate.cutout.height * clipped_fraction))
        else:
            left = int(rng.integers(0, max(1, canvas_size[0] - candidate.cutout.width + 1)))
            top = canvas_size[1] - int(round(candidate.cutout.height * (1.0 - clipped_fraction)))
        tag = "FRAME_CLIPPED"
    else:
        tray_left, tray_top, tray_right, tray_bottom = tray_roi
        crossing = float(rng.uniform(0.05, 0.18))
        if edge == 0:
            left = tray_left - int(round(candidate.cutout.width * crossing))
            top = int(
                rng.integers(tray_top, max(tray_top + 1, tray_bottom - candidate.cutout.height))
            )
        elif edge == 1:
            left = tray_right - int(round(candidate.cutout.width * (1.0 - crossing)))
            top = int(
                rng.integers(tray_top, max(tray_top + 1, tray_bottom - candidate.cutout.height))
            )
        elif edge == 2:
            left = int(
                rng.integers(tray_left, max(tray_left + 1, tray_right - candidate.cutout.width))
            )
            top = tray_top - int(round(candidate.cutout.height * crossing))
        else:
            left = int(
                rng.integers(tray_left, max(tray_left + 1, tray_right - candidate.cutout.width))
            )
            top = tray_bottom - int(round(candidate.cutout.height * (1.0 - crossing)))
        left = int(np.clip(left, 0, max(0, canvas_size[0] - candidate.cutout.width)))
        top = int(np.clip(top, 0, max(0, canvas_size[1] - candidate.cutout.height)))
        tag = "TRAY_EDGE_TOUCH"
    mask = _mask_on_canvas(
        candidate.cutout,
        left=left,
        top=top,
        canvas_size=canvas_size,
    )
    frame_visibility = int(np.count_nonzero(mask)) / max(1, full_area)
    if tag == "FRAME_CLIPPED" and not 0.60 <= frame_visibility <= 0.95:
        return None
    return (
        PlacedObject(
            candidate=candidate,
            left=left,
            top=top,
            full_area=full_area,
            in_frame_mask=mask,
        ),
        tag,
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    left = int(xx.min())
    top = int(yy.min())
    right = int(xx.max()) + 1
    bottom = int(yy.max()) + 1
    return left, top, right - left, bottom - top


def _offset_blurred_mask(
    mask: np.ndarray,
    *,
    offset_x: int,
    offset_y: int,
    blur_radius: float,
) -> Image.Image:
    source = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    shifted = Image.new("L", source.size, 0)
    shifted.paste(source, (offset_x, offset_y))
    return shifted.filter(ImageFilter.GaussianBlur(blur_radius))


def _paste_cutout(canvas: Image.Image, placed: PlacedObject) -> None:
    canvas.paste(
        placed.candidate.cutout.convert("RGB"),
        (placed.left, placed.top),
        placed.candidate.cutout.getchannel("A"),
    )


def _external_occluder(
    placed: Sequence[PlacedObject],
    *,
    canvas_size: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    union = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
    for row in placed:
        union |= row.in_frame_mask
    bbox = _bbox_from_mask(union)
    overlay = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    external_mask = np.zeros_like(union)
    if bbox is None:
        return overlay, external_mask, {}
    left, top, width, height = bbox
    occluder_width = max(16, int(round(width * float(rng.uniform(0.42, 0.58)))))
    occluder_height = max(16, int(round(height * float(rng.uniform(0.48, 0.68)))))
    center_x = left + width // 2 + int(rng.integers(-max(1, width // 8), max(2, width // 8 + 1)))
    center_y = top + height // 2 + int(rng.integers(-max(1, height // 8), max(2, height // 8 + 1)))
    polygon = [
        (center_x - occluder_width // 2, center_y - occluder_height // 2),
        (center_x + occluder_width // 2, center_y - occluder_height // 2 + 5),
        (center_x + occluder_width // 2 - 4, center_y + occluder_height // 2),
        (center_x - occluder_width // 2 + 3, center_y + occluder_height // 2 - 3),
    ]
    mask_image = Image.new("L", canvas_size, 0)
    ImageDraw.Draw(mask_image).polygon(polygon, fill=255)
    external_mask = np.asarray(mask_image) > 0
    shadow = mask_image.filter(ImageFilter.GaussianBlur(7.0))
    shadow_layer = Image.new("RGBA", canvas_size, (70, 66, 60, 70))
    overlay.alpha_composite(Image.composite(shadow_layer, Image.new("RGBA", canvas_size), shadow))
    paper_color = int(rng.integers(205, 236))
    ImageDraw.Draw(overlay).polygon(
        polygon,
        fill=(paper_color, paper_color, max(0, paper_color - 3), 255),
    )
    return overlay, external_mask, {"polygon": [list(point) for point in polygon]}


def _apply_spatial_lighting(
    image: Image.Image, rng: np.random.Generator
) -> tuple[Image.Image, dict[str, Any]]:
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    gradient_x = float(rng.uniform(-0.20, 0.20))
    gradient_y = float(rng.uniform(-0.20, 0.20))
    exposure = float(rng.uniform(0.82, 1.18))
    contrast = float(rng.uniform(0.90, 1.10))
    factor = 1.0 + gradient_x * (xx / max(width - 1, 1) - 0.5)
    factor += gradient_y * (yy / max(height - 1, 1) - 0.5)
    color_temperature = float(rng.uniform(-0.06, 0.06))
    array = (array - 127.5) * contrast + 127.5
    array[..., 0] *= factor * exposure * (1.0 + color_temperature)
    array[..., 1] *= factor * exposure
    array[..., 2] *= factor * exposure * (1.0 - color_temperature)
    result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    glare = bool(rng.random() < 0.42)
    glare_parameters: dict[str, Any] | None = None
    if glare:
        glare_mask = Image.new("L", result.size, 0)
        center_x = int(rng.integers(width // 5, max(width // 5 + 1, width * 4 // 5)))
        center_y = int(rng.integers(height // 5, max(height // 5 + 1, height * 4 // 5)))
        radius_x = int(rng.integers(max(8, width // 12), max(9, width // 4)))
        radius_y = int(rng.integers(max(8, height // 18), max(9, height // 8)))
        ImageDraw.Draw(glare_mask).ellipse(
            (center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y),
            fill=int(rng.integers(28, 68)),
        )
        glare_mask = glare_mask.filter(ImageFilter.GaussianBlur(max(radius_x, radius_y) * 0.45))
        result.paste(Image.new("RGB", result.size, (255, 252, 244)), (0, 0), glare_mask)
        glare_parameters = {
            "center": [center_x, center_y],
            "radius": [radius_x, radius_y],
        }
    weak_blur = bool(rng.random() < 0.24)
    blur_radius = float(rng.uniform(0.25, 0.75)) if weak_blur else 0.0
    if weak_blur:
        result = result.filter(ImageFilter.GaussianBlur(blur_radius))
    return result, {
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "exposure": exposure,
        "contrast": contrast,
        "color_temperature": color_temperature,
        "glare": glare_parameters,
        "weak_blur_radius": blur_radius,
    }


def _motion_blur(image: Image.Image, *, length: int, angle: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    radius = max(1, length // 2)
    padded = np.pad(array, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    accumulator = np.zeros_like(array, dtype=np.float32)
    samples = 0
    for distance in range(-radius, radius + 1):
        offset_x = int(round(math.cos(angle) * distance))
        offset_y = int(round(math.sin(angle) * distance))
        start_y = radius + offset_y
        start_x = radius + offset_x
        accumulator += padded[
            start_y : start_y + image.height,
            start_x : start_x + image.width,
        ]
        samples += 1
    return Image.fromarray(np.clip(accumulator / samples, 0, 255).astype(np.uint8), mode="RGB")


def _recapture_subtypes(count: int) -> list[str]:
    if count < 1:
        return []
    weights = {
        "EMPTY_TRAY": 0.25,
        "SEVERE_BLUR": 0.25,
        "OVEREXPOSED": 0.20,
        "UNDEREXPOSED": 0.15,
        "SEVERE_OCCLUSION": 0.15,
    }
    counts = {name: 0 for name in RECAPTURE_SUBTYPES}
    remaining = count
    if count >= len(RECAPTURE_SUBTYPES):
        counts = {name: 1 for name in RECAPTURE_SUBTYPES}
        remaining -= len(RECAPTURE_SUBTYPES)
    raw = {name: remaining * weight for name, weight in weights.items()}
    for name in RECAPTURE_SUBTYPES:
        allocated = int(math.floor(raw[name]))
        counts[name] += allocated
        remaining -= allocated
    order = sorted(
        RECAPTURE_SUBTYPES,
        key=lambda name: (raw[name] - math.floor(raw[name]), weights[name]),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    return [name for name in RECAPTURE_SUBTYPES for _ in range(counts[name])]


def _scenario_schedule(
    plan: GridScenePlan, rng: np.random.Generator
) -> list[tuple[str, str | None]]:
    schedule: list[tuple[str, str | None]] = []
    for scenario in plan.expanded():
        if scenario != "RECAPTURE":
            schedule.append((scenario, None))
    recapture = _recapture_subtypes(plan.recapture)
    rng.shuffle(recapture)
    schedule.extend(("RECAPTURE", subtype) for subtype in recapture)
    rng.shuffle(schedule)
    return schedule


def _place_candidates(
    candidates: Sequence[CandidateObject],
    *,
    scenario: str,
    canvas_size: tuple[int, int],
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> tuple[list[PlacedObject], list[str]] | None:
    tray_roi = _tray_roi_pixels(recipe, canvas_size)
    placed: list[PlacedObject] = []
    condition_tags: list[str] = []
    for index, candidate in enumerate(candidates):
        if scenario == "EDGE" and index == 0:
            edge_result = _place_edge_object(
                candidate,
                canvas_size=canvas_size,
                tray_roi=tray_roi,
                rng=rng,
            )
            if edge_result is None:
                return None
            row, tag = edge_result
            placed.append(row)
            condition_tags.append(tag)
            continue
        if scenario == "OCCLUSION" and index == 1 and placed:
            row = _place_forced_overlap(
                candidate,
                target=placed[0],
                placed=placed,
                canvas_size=canvas_size,
                rng=rng,
                recipe=recipe,
            )
            if row is None:
                return None
            placed.append(row)
            condition_tags.append("OBJECT_OCCLUSION_10_30")
            continue
        maximum_overlap = 0.018 if scenario == "DENSE" else 0.0
        row = _place_nonoverlapping(
            candidate,
            placed=placed,
            canvas_size=canvas_size,
            tray_roi=tray_roi,
            maximum_overlap=maximum_overlap,
            rng=rng,
            recipe=recipe,
        )
        if row is None:
            return None
        placed.append(row)
    return placed, condition_tags


def _render_placed_objects(
    background: Image.Image,
    placed: Sequence[PlacedObject],
    *,
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> tuple[Image.Image, dict[str, Any]]:
    canvas = background.copy()
    shadow_enabled = bool(placed and rng.random() < recipe.shadow_probability)
    shadow_angle = float(rng.uniform(0.0, math.tau))
    shadow_offset = float(rng.uniform(4.0, 11.0))
    shadow_blur = float(rng.uniform(5.0, 12.0))
    shadow_opacity = float(rng.uniform(0.12, 0.24))
    offset_x = int(round(math.cos(shadow_angle) * shadow_offset))
    offset_y = int(round(math.sin(shadow_angle) * shadow_offset))
    for row in placed:
        if shadow_enabled:
            shadow = _offset_blurred_mask(
                row.in_frame_mask,
                offset_x=offset_x,
                offset_y=offset_y,
                blur_radius=shadow_blur,
            )
            opacity = shadow.point(lambda value: round(value * shadow_opacity))
            canvas.paste(Image.new("RGB", canvas.size, (70, 65, 57)), (0, 0), opacity)
        _paste_cutout(canvas, row)
    return canvas, {
        "enabled": shadow_enabled,
        "angle_radians": shadow_angle,
        "offset_pixels": shadow_offset,
        "blur_radius": shadow_blur,
        "opacity": shadow_opacity,
    }


def _apply_recapture_effect(
    image: Image.Image,
    subtype: str,
    *,
    rng: np.random.Generator,
) -> tuple[Image.Image, dict[str, Any]]:
    if subtype == "SEVERE_BLUR":
        length = int(rng.integers(11, 22))
        angle = float(rng.uniform(0.0, math.tau))
        return _motion_blur(image, length=length, angle=angle), {
            "motion_blur_length": length,
            "motion_blur_angle_radians": angle,
        }
    if subtype == "OVEREXPOSED":
        brightness = float(rng.uniform(1.50, 1.82))
        contrast = float(rng.uniform(0.58, 0.78))
        result = ImageEnhance.Brightness(image).enhance(brightness)
        return ImageEnhance.Contrast(result).enhance(contrast), {
            "brightness": brightness,
            "contrast": contrast,
        }
    if subtype == "UNDEREXPOSED":
        brightness = float(rng.uniform(0.36, 0.54))
        contrast = float(rng.uniform(0.82, 1.02))
        result = ImageEnhance.Brightness(image).enhance(brightness)
        return ImageEnhance.Contrast(result).enhance(contrast), {
            "brightness": brightness,
            "contrast": contrast,
        }
    return image, {}


def _visible_annotations(
    placed: Sequence[PlacedObject],
    *,
    external_occluder: np.ndarray,
    recipe: GridCompositeRecipe,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    occluders = external_occluder.copy()
    reversed_rows: list[tuple[PlacedObject, np.ndarray]] = []
    for row in reversed(placed):
        visible = row.in_frame_mask & ~occluders
        reversed_rows.append((row, visible))
        occluders |= row.in_frame_mask
    rows = list(reversed(reversed_rows))
    annotations: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for row, visible in rows:
        visible_area = int(np.count_nonzero(visible))
        in_frame_area = int(np.count_nonzero(row.in_frame_mask))
        frame_visibility = in_frame_area / max(1, row.full_area)
        occlusion_fraction = 1.0 - visible_area / max(1, in_frame_area)
        total_visibility = visible_area / max(1, row.full_area)
        if total_visibility < recipe.minimum_visibility_fraction:
            return None
        bbox = _bbox_from_mask(visible)
        if bbox is None:
            return None
        annotation = {
            "category_id": row.candidate.category_id,
            "bbox_xywh": list(bbox),
            "area": visible_area,
            "iscrowd": 0,
            "visible_fraction": total_visibility,
            "frame_visibility_fraction": frame_visibility,
            "occlusion_fraction": occlusion_fraction,
        }
        annotations.append(annotation)
        provenance.append(
            {
                "category_id": row.candidate.category_id,
                "side": row.candidate.side,
                "source_image": row.candidate.asset.source.relative_path,
                "source_image_sha256": row.candidate.asset.source.sha256,
                "source_grid_position": row.candidate.asset.source.grid_position,
                "source_capture_view": row.candidate.asset.source.capture_view,
                "physical_item_id": row.candidate.asset.source.physical_item_id,
                "position": [row.left, row.top],
                "bbox_xywh": list(bbox),
                "visible_fraction": total_visibility,
                "frame_visibility_fraction": frame_visibility,
                "occlusion_fraction": occlusion_fraction,
                "transform": row.candidate.transform,
            }
        )
    return annotations, provenance


def _generate_scene_attempt(
    *,
    scenario: str,
    subtype: str | None,
    category_ids: Sequence[int],
    background: Image.Image,
    assets: dict[tuple[int, str, str], LoadedAsset],
    side_counts: Counter[tuple[int, str]],
    view_counts: Counter[tuple[int, str, str]],
    rng: np.random.Generator,
    recipe: GridCompositeRecipe,
) -> (
    tuple[
        GeneratedScene,
        Counter[tuple[int, str]],
        Counter[tuple[int, str, str]],
    ]
    | None
):
    if subtype == "EMPTY_TRAY":
        image, lighting = _apply_spatial_lighting(background.copy(), rng)
        return (
            GeneratedScene(
                image=image,
                annotations=[],
                object_provenance=[],
                scenario=scenario,
                subtype=subtype,
                condition_tags=[subtype],
                expected_status="IMAGE_RECAPTURE",
                expected_reason_codes=["IMAGE_RECAPTURE_REQUIRED"],
                scene_parameters={"lighting": lighting},
            ),
            side_counts.copy(),
            view_counts.copy(),
        )

    candidates, next_side_counts, next_view_counts = _build_candidates(
        category_ids,
        scenario=scenario,
        background=background,
        assets=assets,
        side_counts=side_counts,
        view_counts=view_counts,
        rng=rng,
        recipe=recipe,
    )
    placement = _place_candidates(
        candidates,
        scenario=scenario,
        canvas_size=background.size,
        rng=rng,
        recipe=recipe,
    )
    if placement is None:
        return None
    placed, condition_tags = placement
    image, shadow = _render_placed_objects(background, placed, rng=rng, recipe=recipe)
    external_mask = np.zeros((background.height, background.width), dtype=bool)
    external_parameters: dict[str, Any] = {}
    if subtype == "SEVERE_OCCLUSION":
        overlay, external_mask, external_parameters = _external_occluder(
            placed,
            canvas_size=background.size,
            rng=rng,
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        condition_tags.append(subtype)

    visible = _visible_annotations(
        placed,
        external_occluder=external_mask,
        recipe=recipe,
    )
    if visible is None:
        return None
    annotations, provenance = visible
    lighting: dict[str, Any] = {}
    recapture_effect: dict[str, Any] = {}
    if scenario == "LIGHTING":
        image, lighting = _apply_spatial_lighting(image, rng)
        condition_tags.append("BOUNDED_LIGHTING_VARIATION")
    if scenario == "RECAPTURE" and subtype is not None:
        image, recapture_effect = _apply_recapture_effect(image, subtype, rng=rng)
        if subtype not in condition_tags:
            condition_tags.append(subtype)
    expected_status = "IMAGE_RECAPTURE" if scenario == "RECAPTURE" else "SEGMENTATION"
    expected_reason_codes = (
        ["IMAGE_RECAPTURE_REQUIRED"] if expected_status == "IMAGE_RECAPTURE" else []
    )
    return (
        GeneratedScene(
            image=image,
            annotations=annotations,
            object_provenance=provenance,
            scenario=scenario,
            subtype=subtype,
            condition_tags=condition_tags,
            expected_status=expected_status,
            expected_reason_codes=expected_reason_codes,
            scene_parameters={
                "shadow": shadow,
                "lighting": lighting,
                "recapture_effect": recapture_effect,
                "external_occluder": external_parameters,
            },
        ),
        next_side_counts,
        next_view_counts,
    )


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="JPEG", quality=quality, optimize=False)
    return stream.getvalue()


def _capture_manifest_payload(grid: CaptureGrid) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "contract": grid.contract,
        "background": {
            "image_path": grid.background_relative_path,
            "sha256": grid.background_sha256,
            "width": grid.width,
            "height": grid.height,
        },
        "assets": [
            {
                "category_id": row.category_id,
                "class_name": row.class_name,
                "side": row.side,
                "grid_position": row.grid_position,
                "capture_view": row.capture_view,
                "extraction_mode": row.extraction_mode,
                "image_path": row.relative_path,
                "image_sha256": row.sha256,
                "physical_item_id": row.physical_item_id,
                "capture_session_id": row.capture_session_id,
            }
            for row in grid.assets
        ],
    }


def _contact_sheet(image_paths: Sequence[Path], output_path: Path) -> None:
    selected = list(image_paths[:12])
    if not selected:
        return
    cell_width = 240
    cell_height = 232
    columns = 4
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (32, 34, 38))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(selected):
        with Image.open(path) as opened:
            preview = ImageOps.contain(
                ImageOps.exif_transpose(opened).convert("RGB"),
                (cell_width - 8, cell_height - 28),
                method=Image.Resampling.LANCZOS,
            )
        left = index % columns * cell_width + (cell_width - preview.width) // 2
        top = index // columns * cell_height + 4
        sheet.paste(preview, (left, top))
        draw.text(
            (index % columns * cell_width + 8, index // columns * cell_height + cell_height - 20),
            path.stem,
            fill=(235, 237, 240),
        )
    sheet.save(output_path, format="JPEG", quality=92)


def _validate_generated_distribution(
    *,
    plan: GridScenePlan,
    scenario_counts: Counter[str],
    class_counts: Counter[int],
    rows: Sequence[dict[str, Any]],
) -> None:
    expected = {
        "NORMAL": plan.normal,
        "DENSE": plan.dense,
        "OCCLUSION": plan.occlusion,
        "EDGE": plan.edge,
        "LIGHTING": plan.lighting,
        "RECAPTURE": plan.recapture,
    }
    if dict(scenario_counts) != {key: value for key, value in expected.items() if value}:
        raise RuntimeError(
            f"generated scenario distribution mismatch: expected={expected}, actual={scenario_counts}"
        )
    if len(rows) != plan.total:
        raise RuntimeError("generated image count does not match the scene plan")
    observed = [class_counts[category_id] for category_id in range(1, 21)]
    if observed and max(observed) - min(observed) > 2:
        raise RuntimeError(f"generated class occurrence imbalance is too large: {observed}")
    if any(row["split"] != "train_synthetic" or int(row["fold"]) != 0 for row in rows):
        raise RuntimeError("synthetic-only output must remain a training-only dataset")


def generate_grid_composite_dataset(
    capture_root: Path,
    output_root: Path,
    *,
    seed: int,
    recipe: GridCompositeRecipe = GridCompositeRecipe(),
    background_image: Path | None = None,
) -> dict[str, Any]:
    """Generate a balanced, training-only detector dataset from the 201-image grid contract."""
    recipe.validate()
    grid = (
        discover_pose_cutouts(capture_root, background_image)
        if background_image is not None
        else discover_capture_grid(capture_root)
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "images"
    image_root.mkdir()
    background, assets = _load_assets(grid, recipe)
    recipe_digest = _recipe_sha256(recipe)
    rng = np.random.default_rng(seed)
    schedule = _scenario_schedule(recipe.plan, rng)
    category_counts: Counter[int] = Counter()
    side_counts: Counter[tuple[int, str]] = Counter()
    view_counts: Counter[tuple[int, str, str]] = Counter()
    source_usage: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    annotation_id = 1

    for image_index, (scenario, subtype) in enumerate(schedule):
        count = _object_count(scenario, subtype, rng, recipe)
        category_ids = _balanced_categories(category_counts, count, rng)
        generated: GeneratedScene | None = None
        next_side_counts = side_counts
        next_view_counts = view_counts
        for _ in range(recipe.scene_attempts):
            attempt = _generate_scene_attempt(
                scenario=scenario,
                subtype=subtype,
                category_ids=category_ids,
                background=background,
                assets=assets,
                side_counts=side_counts,
                view_counts=view_counts,
                rng=rng,
                recipe=recipe,
            )
            if attempt is not None:
                generated, next_side_counts, next_view_counts = attempt
                break
        if generated is None:
            raise RuntimeError(
                f"failed to generate {scenario}/{subtype or 'default'} scene {image_index + 1}"
            )
        side_counts = next_side_counts
        view_counts = next_view_counts
        category_counts.update(row["category_id"] for row in generated.annotations)
        scenario_counts[scenario] += 1
        if subtype is not None:
            subtype_counts[subtype] += 1
        quality = int(rng.integers(recipe.jpeg_quality_min, recipe.jpeg_quality_max + 1))
        file_name = f"grid_composite_{image_index + 1:05d}.jpg"
        image_path = image_root / file_name
        image_bytes = _jpeg_bytes(generated.image, quality)
        image_path.write_bytes(image_bytes)
        image_paths.append(image_path)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        image_id = image_index + 1

        row_annotations: list[dict[str, Any]] = []
        for annotation in generated.annotations:
            output_annotation = {
                "annotation_id": annotation_id,
                **annotation,
            }
            row_annotations.append(output_annotation)
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation["category_id"],
                    "bbox": annotation["bbox_xywh"],
                    "area": annotation["area"],
                    "iscrowd": annotation["iscrowd"],
                    "visible_fraction": annotation["visible_fraction"],
                    "frame_visibility_fraction": annotation["frame_visibility_fraction"],
                    "occlusion_fraction": annotation["occlusion_fraction"],
                }
            )
            annotation_id += 1
        physical_item_ids = sorted({row["physical_item_id"] for row in generated.object_provenance})
        manifest_rows.append(
            {
                "record_type": "detection",
                "source": "bread_grid_position_aware_composite",
                "source_dataset": grid.source_dataset,
                "image_id": image_id,
                "image_path": f"images/{file_name}",
                "image_sha256": image_sha256,
                "capture_session_id": f"synthetic-grid:{seed}:{image_id:05d}",
                "perceptual_group_id": f"synthetic-grid:{seed}:{image_id:05d}",
                "physical_item_ids": physical_item_ids,
                "split": "train_synthetic",
                "fold": 0,
                "width": grid.width,
                "height": grid.height,
                "scenario": scenario,
                "scenario_subtype": subtype,
                "condition_tags": generated.condition_tags,
                "expected_image_status": generated.expected_status,
                "expected_reason_codes": generated.expected_reason_codes,
                "annotations": row_annotations,
                "generation_recipe_sha256": recipe_digest,
            }
        )
        for source in generated.object_provenance:
            source_usage[source["source_image"]] += 1
        provenance_rows.append(
            {
                "image_id": image_id,
                "image_sha256": image_sha256,
                "seed": seed,
                "scenario": scenario,
                "scenario_subtype": subtype,
                "condition_tags": generated.condition_tags,
                "background_image": grid.background_relative_path,
                "background_image_sha256": grid.background_sha256,
                "jpeg_quality": quality,
                "scene_parameters": generated.scene_parameters,
                "objects": generated.object_provenance,
            }
        )
        coco_images.append(
            {
                "id": image_id,
                "file_name": f"images/{file_name}",
                "width": grid.width,
                "height": grid.height,
                "scenario": scenario,
                "scenario_subtype": subtype,
                "expected_image_status": generated.expected_status,
                "expected_reason_codes": generated.expected_reason_codes,
            }
        )

    _validate_generated_distribution(
        plan=recipe.plan,
        scenario_counts=scenario_counts,
        class_counts=category_counts,
        rows=manifest_rows,
    )
    capture_manifest = _capture_manifest_payload(grid)
    capture_manifest_body = json.dumps(capture_manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_body = "".join(_canonical_json(row) + "\n" for row in manifest_rows)
    provenance_body = "".join(_canonical_json(row) + "\n" for row in provenance_rows)
    categories = [
        {
            "id": category_id,
            "name": next(row.class_name for row in grid.assets if row.category_id == category_id),
            "supercategory": "bread",
        }
        for category_id in range(1, 21)
    ]
    coco_payload = {
        "info": {
            "description": "Position-aware 20-class bread grid composites",
            "version": "1.0",
            "training_only": True,
            "independent_validation": False,
            "generation_recipe_sha256": recipe_digest,
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    coco_body = json.dumps(coco_payload, ensure_ascii=False, indent=2) + "\n"
    (output_root / "source-manifest.json").write_text(capture_manifest_body, encoding="utf-8")
    (output_root / "manifest.jsonl").write_text(manifest_body, encoding="utf-8", newline="\n")
    (output_root / "provenance.jsonl").write_text(
        provenance_body,
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "instances.json").write_text(coco_body, encoding="utf-8", newline="\n")
    _contact_sheet(image_paths, output_root / "preview.jpg")
    metadata = {
        "schema_version": "1.0",
        "contract": f"{grid.contract}-to-1500",
        "dataset_version": grid.source_dataset,
        "training_only": True,
        "independent_validation": False,
        "source_image_count": len(grid.assets),
        "background_image_count": 1,
        "generated_image_count": len(manifest_rows),
        "generated_annotation_count": len(coco_annotations),
        "synthetic_image_count": len(manifest_rows),
        "synthetic_annotation_count": len(coco_annotations),
        "synthetic_empty_image_count": sum(not row["annotations"] for row in manifest_rows),
        "seed": seed,
        "recipe": asdict(recipe),
        "recipe_sha256": recipe_digest,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "recapture_subtype_counts": dict(sorted(subtype_counts.items())),
        "class_occurrence_counts": {
            str(category_id): category_counts[category_id] for category_id in range(1, 21)
        },
        "side_occurrence_counts": {
            f"{category_id}:{side}": side_counts[(category_id, side)]
            for category_id in range(1, 21)
            for side in SIDES
        },
        "view_occurrence_counts": {
            f"{category_id}:{side}:{position}": view_counts[(category_id, side, position)]
            for category_id in range(1, 21)
            for side in SIDES
            for position in GRID_POSITIONS
        },
        "source_usage_counts": dict(sorted(source_usage.items())),
        "source_manifest_sha256": hashlib.sha256(capture_manifest_body.encode("utf-8")).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_body.encode("utf-8")).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_body.encode("utf-8")).hexdigest(),
        "instances_sha256": hashlib.sha256(coco_body.encode("utf-8")).hexdigest(),
        "limitations": [
            "synthetic-only training data is not independent real-world performance evidence",
            "2D composition cannot synthesize unseen bread surfaces or true 3D contact deformation",
            "all derivatives retain their source physical-item provenance",
        ],
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 1,500 position-aware detector composites from a 201-image grid"
    )
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--background-image",
        type=Path,
        help="Use four-direction + vertical white-background cutouts with this empty tray",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    metadata = generate_grid_composite_dataset(
        args.capture_root,
        args.output_root,
        seed=args.seed,
        background_image=args.background_image,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
