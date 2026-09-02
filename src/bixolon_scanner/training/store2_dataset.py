from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage

from bixolon_scanner.configuration import load_json_config

from .ten_shot_manifest import inspect_image

SOURCE_DIRECTORY = "single_objects_4"
EXPECTED_CLASS_COUNT = 20
EXPECTED_SHOTS_PER_CLASS = 10
EXPECTED_BACKGROUND_COUNT = 10
CLASS_DIRECTORY_PATTERN = re.compile(r"^bread_(?P<category>\d{2})_(?P<slug>[a-z0-9_]+)$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _labels(root: Path) -> list[dict[str, Any]]:
    annotation = root.parent / "annotations" / "multi_object_instances.json"
    payload = json.loads(annotation.read_text(encoding="utf-8-sig"))
    categories = sorted(payload["categories"], key=lambda row: int(row["id"]))
    if [int(row["id"]) for row in categories] != list(range(1, EXPECTED_CLASS_COUNT + 1)):
        raise ValueError("bread labels must be contiguous from 1 through 20")
    return [
        {
            "category_id": int(row["id"]),
            "class_id": f"bread_{int(row['id']):02d}",
            "class_name": str(row["name"]),
        }
        for row in categories
    ]


def audit_store2_source(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = source_root.resolve()
    if not root.is_dir() or root.name != SOURCE_DIRECTORY:
        raise ValueError(f"source root must be a directory named {SOURCE_DIRECTORY}")
    labels = _labels(root)
    labels_by_id = {int(row["category_id"]): row for row in labels}
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    expected_names = {"background"} | {
        path.name for path in directories if CLASS_DIRECTORY_PATTERN.fullmatch(path.name)
    }
    actual_names = {path.name for path in directories}
    if expected_names != actual_names or "background" not in actual_names:
        raise ValueError("single_objects_4 may contain only 20 bread directories and background")
    class_directories = [
        path for path in directories if CLASS_DIRECTORY_PATTERN.fullmatch(path.name)
    ]
    if len(class_directories) != EXPECTED_CLASS_COUNT:
        raise ValueError("single_objects_4 requires exactly 20 class directories")

    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for directory in class_directories:
        match = CLASS_DIRECTORY_PATTERN.fullmatch(directory.name)
        assert match is not None
        category_id = int(match.group("category"))
        label = labels_by_id.get(category_id)
        if label is None:
            raise ValueError(f"class directory has no label: {directory.name}")
        files = sorted(path for path in directory.iterdir() if path.is_file())
        if len(files) != EXPECTED_SHOTS_PER_CLASS:
            raise ValueError(f"{directory.name} must contain exactly 10 JPEG images")
        for capture_index, path in enumerate(files):
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError(f"unsupported source image: {path.name}")
            inspected = inspect_image(path)
            if inspected.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"source image must decode as RGB: {path.name}")
            if inspected.sha256 in seen_hashes:
                raise ValueError(f"duplicate source image: {path.name}")
            seen_hashes.add(inspected.sha256)
            records.append(
                {
                    "record_type": "classification",
                    "source": "bread_store2_single_original",
                    "source_dataset": SOURCE_DIRECTORY,
                    "image_path": path.relative_to(root).as_posix(),
                    "image_sha256": inspected.sha256,
                    "category_id": category_id,
                    "class_id": label["class_id"],
                    "class_name": label["class_name"],
                    "capture_session_id": f"bread_{category_id:02d}:single_objects_4:session",
                    "physical_item_id": f"bread_{category_id:02d}:store2:item",
                    "source_group": f"bread_{category_id:02d}:store2:item",
                    "capture_index": capture_index,
                    "width": inspected.width,
                    "height": inspected.height,
                }
            )

    if sorted({int(row["category_id"]) for row in records}) != list(
        range(1, EXPECTED_CLASS_COUNT + 1)
    ):
        raise ValueError("single_objects_4 classes must be contiguous")
    counts = Counter(int(row["category_id"]) for row in records)
    if set(counts.values()) != {EXPECTED_SHOTS_PER_CLASS}:
        raise ValueError("single_objects_4 must contain 10 images per class")

    background_records = []
    for capture_index, path in enumerate(sorted((root / "background").iterdir())):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("background must contain JPEG files only")
        inspected = inspect_image(path)
        if inspected.sha256 in seen_hashes:
            raise ValueError(f"duplicate background image: {path.name}")
        seen_hashes.add(inspected.sha256)
        background_records.append(
            {
                "record_type": "background_negative",
                "source": "bread_store2_empty_tray",
                "source_dataset": SOURCE_DIRECTORY,
                "image_path": path.relative_to(root).as_posix(),
                "image_sha256": inspected.sha256,
                "capture_session_id": "background:single_objects_4:session",
                "physical_item_id": None,
                "source_group": f"background:store2:{capture_index:02d}",
                "capture_index": capture_index,
                "width": inspected.width,
                "height": inspected.height,
            }
        )
    if len(background_records) != EXPECTED_BACKGROUND_COUNT:
        raise ValueError("single_objects_4 requires exactly 10 background images")
    all_records = [*records, *background_records]
    digest = hashlib.sha256(
        _canonical_json(
            [
                {
                    "role": row["record_type"],
                    "path": row["image_path"],
                    "sha256": row["image_sha256"],
                }
                for row in all_records
            ]
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": "1.0",
        "dataset_version": f"bread-store2-{digest[:12]}",
        "source_directory": SOURCE_DIRECTORY,
        "source_image_set_sha256": digest,
        "class_count": EXPECTED_CLASS_COUNT,
        "shots_per_class": EXPECTED_SHOTS_PER_CLASS,
        "object_image_count": len(records),
        "background_image_count": len(background_records),
        "labels": labels,
        "background_records": background_records,
        "allowed_source_root": root.as_posix(),
    }
    return records, metadata


@dataclass(frozen=True)
class ForegroundRecipe:
    alignment_width: int = 640
    weak_difference: float = 18.0
    strong_difference: float = 32.0
    minimum_area_ratio: float = 0.0015
    maximum_area_ratio: float = 0.22
    bbox_margin_ratio: float = 0.06
    mask_feather_radius: float = 1.0


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)


def _align_background(source: np.ndarray, background: np.ndarray, width: int) -> np.ndarray:
    import cv2

    height = max(1, round(source.shape[0] * width / source.shape[1]))
    size = (width, height)
    source_gray = cv2.cvtColor(cv2.resize(source, size), cv2.COLOR_RGB2GRAY)
    background_gray = cv2.cvtColor(cv2.resize(background, size), cv2.COLOR_RGB2GRAY)
    source_gray = cv2.GaussianBlur(source_gray, (5, 5), 0)
    background_gray = cv2.GaussianBlur(background_gray, (5, 5), 0)
    mask = np.full(source_gray.shape, 255, dtype=np.uint8)
    mask[round(height * 0.23) : round(height * 0.78), round(width * 0.28) : round(width * 0.72)] = 0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        cv2.findTransformECC(
            source_gray,
            background_gray,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )
    except cv2.error:
        warp = np.eye(2, 3, dtype=np.float32)
    scale = source.shape[1] / width
    warp[0, 2] *= scale
    warp[1, 2] *= scale
    return cv2.warpAffine(
        background,
        warp,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def _reference_background(paths: list[Path]) -> np.ndarray:
    images = [_rgb(path) for path in paths]
    reference = images[0]
    aligned = [reference]
    for image in images[1:]:
        aligned.append(_align_background(reference, image, 640))
    return np.median(np.stack(aligned), axis=0).astype(np.uint8)


def _capture_roi(shape: tuple[int, int], capture_index: int) -> np.ndarray:
    """Return a broad safety region; tray segmentation provides the precise support."""
    height, width = shape
    del capture_index
    roi = np.zeros((height, width), dtype=bool)
    roi[:, round(width * 0.10) : round(width * 0.98)] = True
    return roi


def _tray_roi(background: np.ndarray) -> np.ndarray:
    import cv2

    height, width = background.shape[:2]
    hsv = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
    pale = (hsv[..., 1] <= 85) & (hsv[..., 2] >= 115)
    pale = ndimage.binary_closing(pale, structure=np.ones((15, 15)), iterations=2)
    labels, count = ndimage.label(pale)
    if count == 0:
        raise ValueError("background tray segmentation found no component")
    center = np.asarray([height * 0.5, width * 0.5])
    candidates = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        area = int(component.sum())
        if area < height * width * 0.12:
            continue
        centroid = np.argwhere(component).mean(axis=0)
        distance = float(np.linalg.norm((centroid - center) / [height, width]))
        candidates.append((area / (1.0 + distance * 4.0), component))
    if not candidates:
        raise ValueError("background tray segmentation found no large central component")
    tray = max(candidates, key=lambda row: row[0])[1]
    return ndimage.binary_dilation(tray, iterations=max(16, round(width * 0.035)))


def _foreground(
    source: np.ndarray,
    reference_background: np.ndarray,
    recipe: ForegroundRecipe,
    *,
    capture_index: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, float]]:
    import cv2

    original_height, original_width = source.shape[:2]
    segmentation_width = max(960, recipe.alignment_width)
    if original_width > segmentation_width:
        segmentation_height = max(1, round(original_height * segmentation_width / original_width))
        resized_source = cv2.resize(
            source,
            (segmentation_width, segmentation_height),
            interpolation=cv2.INTER_AREA,
        )
        resized_background = cv2.resize(
            reference_background,
            (segmentation_width, segmentation_height),
            interpolation=cv2.INTER_AREA,
        )
        resized_alpha, resized_bbox, statistics = _foreground(
            resized_source,
            resized_background,
            recipe,
            capture_index=capture_index,
        )
        scale_x = original_width / segmentation_width
        scale_y = original_height / segmentation_height
        left, top, right, bottom = resized_bbox
        bbox = (
            max(0, round(left * scale_x)),
            max(0, round(top * scale_y)),
            min(original_width, round(right * scale_x)),
            min(original_height, round(bottom * scale_y)),
        )
        alpha = cv2.resize(
            resized_alpha,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        statistics["segmentation_width"] = float(segmentation_width)
        for key in ("prompt_x", "prompt_left", "prompt_right"):
            statistics[key] *= scale_x
        for key in ("prompt_y", "prompt_top", "prompt_bottom"):
            statistics[key] *= scale_y
        return alpha, bbox, statistics

    aligned = _align_background(source, reference_background, recipe.alignment_width)
    source_values = source.astype(np.float32)
    background_values = aligned.astype(np.float32)
    height, width = source.shape[:2]
    border = np.zeros((height, width), dtype=bool)
    border[: height // 8] = True
    border[-height // 8 :] = True
    border[:, : width // 8] = True
    border[:, -width // 8 :] = True
    exposure = np.median(source_values[border] - background_values[border], axis=0)
    adjusted = np.clip(background_values + exposure, 0, 255)
    distance = np.max(np.abs(source_values - adjusted), axis=-1)
    roi = _capture_roi((height, width), capture_index) & _tray_roi(aligned)
    residual_values = distance[border & ~roi]
    residual = float(np.percentile(residual_values, 99)) if residual_values.size else 0.0
    weak_threshold = max(recipe.weak_difference, residual + 5.0)
    strong_threshold = max(recipe.strong_difference, weak_threshold + 10.0)
    weak = (distance >= weak_threshold) & roi
    saturation = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    color_seed = (saturation >= 0.08) & (distance >= max(8.0, weak_threshold * 0.4)) & roi
    strong_seed = ndimage.binary_opening(
        color_seed,
        structure=np.ones((3, 3)),
        iterations=1,
    )
    strong_seed = ndimage.binary_closing(strong_seed, structure=np.ones((5, 5)), iterations=1)
    labels, count = ndimage.label(strong_seed)
    candidates = []
    roi_coordinates = np.argwhere(roi)
    center = roi_coordinates.mean(axis=0)
    for component_id in range(1, count + 1):
        component = labels == component_id
        area = int(component.sum())
        area_ratio = area / float(height * width)
        if not recipe.minimum_area_ratio * 0.08 <= area_ratio <= recipe.maximum_area_ratio:
            continue
        coordinates = np.argwhere(component)
        centroid = coordinates.mean(axis=0)
        center_distance = float(np.linalg.norm((centroid - center) / [height, width]))
        mean_saturation = float(saturation[component].mean())
        score = area * (0.35 + mean_saturation) * math.exp(-6.0 * center_distance * center_distance)
        candidates.append((score, component))
    if not candidates:
        raise ValueError("foreground extraction found no valid centered component")
    seed = max(candidates, key=lambda row: row[0])[1]
    seed_y, seed_x = np.nonzero(seed)
    seed_left = int(seed_x.min())
    seed_top = int(seed_y.min())
    seed_right = int(seed_x.max() + 1)
    seed_bottom = int(seed_y.max() + 1)
    seed_extent = max(seed_right - seed_left, seed_bottom - seed_top)
    seed_depth = ndimage.distance_transform_edt(seed)
    prompt_y, prompt_x = np.unravel_index(int(np.argmax(seed_depth)), seed_depth.shape)
    prompt_margin = max(8, round(seed_extent * 0.18))
    prompt_box = (
        max(0, seed_left - prompt_margin),
        max(0, seed_top - prompt_margin),
        min(width, seed_right + prompt_margin),
        min(height, seed_bottom + prompt_margin),
    )
    seed_contours, _hierarchy = cv2.findContours(
        seed.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    hull_points = cv2.convexHull(np.concatenate(seed_contours, axis=0))
    hull = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(hull, hull_points, 1)
    object_support = ndimage.binary_dilation(
        hull.astype(bool),
        iterations=max(6, round(seed_extent * 0.14)),
    )
    expansion = max(32, round(seed_extent * 0.35))
    window = (
        max(0, seed_left - expansion),
        max(0, seed_top - expansion),
        min(width, seed_right + expansion),
        min(height, seed_bottom + expansion),
    )
    window_left, window_top, window_right, window_bottom = window
    grabcut_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[window_top:window_bottom, window_left:window_right] = cv2.GC_PR_BGD
    seed_neighborhood = ndimage.binary_dilation(
        seed,
        iterations=max(12, round(seed_extent * 0.30)),
    )
    probable_foreground = weak & seed_neighborhood & roi
    grabcut_mask[probable_foreground] = cv2.GC_PR_FGD
    grabcut_mask[seed] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    grabcut_foreground = (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)
    grabcut_foreground &= roi & object_support
    final_labels, final_count = ndimage.label(grabcut_foreground)
    selected_labels = {
        component_id
        for component_id in range(1, final_count + 1)
        if np.any(seed & (final_labels == component_id))
    }
    component = np.isin(final_labels, list(selected_labels))
    if not np.any(component):
        component = seed
    component = ndimage.binary_closing(component, structure=np.ones((5, 5)), iterations=1)
    ys, xs = np.nonzero(component)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    margin = max(4, round(max(right - left, bottom - top) * recipe.bbox_margin_ratio * 0.5))
    bbox = (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )
    alpha = ndimage.gaussian_filter(component.astype(np.float32), recipe.mask_feather_radius)
    alpha = np.rint(np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    return (
        alpha,
        bbox,
        {
            "alignment_residual_p99": residual,
            "weak_threshold": weak_threshold,
            "strong_threshold": strong_threshold,
            "foreground_area_ratio": float(component.mean()),
            "capture_position": float(capture_index % 5),
            "prompt_x": float(prompt_x),
            "prompt_y": float(prompt_y),
            "prompt_left": float(prompt_box[0]),
            "prompt_top": float(prompt_box[1]),
            "prompt_right": float(prompt_box[2]),
            "prompt_bottom": float(prompt_box[3]),
            "grabcut_window_area_ratio": float(
                (window_right - window_left) * (window_bottom - window_top) / (height * width)
            ),
        },
    )


class _SamAnnotationRefiner:
    """Use an official SAM checkpoint only to refine source annotations."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import SamModel, SamProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = SamProcessor.from_pretrained(model_name)
        self._model = SamModel.from_pretrained(model_name).to(self._device).eval()
        self.model_name = model_name
        self.revision = str(getattr(self._model.config, "_commit_hash", "unknown"))

    def refine(
        self,
        source: np.ndarray,
        seed_statistics: dict[str, float],
        recipe: ForegroundRecipe,
    ) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, float]]:
        point_x = float(seed_statistics["prompt_x"])
        point_y = float(seed_statistics["prompt_y"])
        height, width = source.shape[:2]
        half_extent = round(min(width, height) * 0.35)
        manual_box = all(
            name in seed_statistics
            for name in (
                "manual_prompt_left",
                "manual_prompt_top",
                "manual_prompt_right",
                "manual_prompt_bottom",
            )
        )
        prompt_box = (
            tuple(
                round(seed_statistics[name])
                for name in (
                    "manual_prompt_left",
                    "manual_prompt_top",
                    "manual_prompt_right",
                    "manual_prompt_bottom",
                )
            )
            if manual_box
            else (
                max(0, round(point_x) - half_extent),
                max(0, round(point_y) - half_extent),
                min(width, round(point_x) + half_extent),
                min(height, round(point_y) + half_extent),
            )
        )
        inputs = self._processor(
            Image.fromarray(source),
            input_points=[[[point_x, point_y]]],
            input_boxes=[[list(prompt_box)]],
            return_tensors="pt",
        )
        original_sizes = inputs.pop("original_sizes")
        reshaped_input_sizes = inputs.pop("reshaped_input_sizes")
        model_inputs = {
            name: value.to(self._device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self._model(**model_inputs)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(), original_sizes, reshaped_input_sizes
        )[0][0]
        scores = outputs.iou_scores.detach().cpu()[0, 0]
        candidates = []
        prompt_row = min(height - 1, max(0, round(point_y)))
        prompt_column = min(width - 1, max(0, round(point_x)))
        for candidate_index, mask_tensor in enumerate(masks):
            mask = mask_tensor.numpy() > 0
            labels, _count = ndimage.label(mask)
            prompt_label = int(labels[prompt_row, prompt_column])
            if prompt_label == 0:
                continue
            component = labels == prompt_label
            area_ratio = float(component.mean())
            if not recipe.minimum_area_ratio * 0.5 <= area_ratio <= recipe.maximum_area_ratio:
                continue
            predicted_iou = float(scores[candidate_index])
            candidates.append((predicted_iou, -candidate_index, component, candidate_index))
        if not candidates:
            raise ValueError("SAM annotation refinement found no valid prompt component")
        predicted_iou, _order, component, candidate_index = max(
            candidates, key=lambda row: (row[0], row[1])
        )
        component = ndimage.binary_closing(component, structure=np.ones((3, 3)), iterations=1)
        ys, xs = np.nonzero(component)
        left, top, right, bottom = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        )
        margin = max(3, round(max(right - left, bottom - top) * 0.01))
        bbox = (
            max(0, left - margin),
            max(0, top - margin),
            min(width, right + margin),
            min(height, bottom + margin),
        )
        alpha = ndimage.gaussian_filter(component.astype(np.float32), recipe.mask_feather_radius)
        alpha = np.rint(np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
        statistics = {
            **seed_statistics,
            "sam_candidate_index": float(candidate_index),
            "sam_predicted_iou": predicted_iou,
            "sam_foreground_area_ratio": float(component.mean()),
            "sam_prompt_box_left": float(prompt_box[0]),
            "sam_prompt_box_top": float(prompt_box[1]),
            "sam_prompt_box_right": float(prompt_box[2]),
            "sam_prompt_box_bottom": float(prompt_box[3]),
        }
        return alpha, bbox, statistics


def _write_contact_sheet(paths: list[Path], output: Path) -> None:
    thumbnails = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((320, 180), Image.Resampling.LANCZOS)
            thumbnails.append(image.copy())
    columns = 5
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * 320, rows * 180), "white")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * 320, (index // columns) * 180))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def prepare_store2_source(
    source_root: Path,
    output_root: Path,
    *,
    recipe: ForegroundRecipe = ForegroundRecipe(),
    annotation_model: str | None = None,
    annotation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    records, metadata = audit_store2_source(source_root)
    override_rows = annotation_overrides.get("overrides", []) if annotation_overrides else []
    if (
        annotation_overrides
        and annotation_overrides.get("source_image_set_sha256")
        != metadata["source_image_set_sha256"]
    ):
        raise ValueError("annotation overrides target a different source image set")
    overrides_by_image_id = {int(row["image_id"]): row for row in override_rows}
    if len(overrides_by_image_id) != len(override_rows):
        raise ValueError("annotation overrides contain duplicate image_id values")
    output_root.mkdir(parents=True)
    background_paths = [source_root / row["image_path"] for row in metadata["background_records"]]
    reference = _reference_background(background_paths)
    Image.fromarray(reference).save(output_root / "reference-background.jpg", quality=95)
    crop_root = output_root / "crops"
    cutout_root = output_root / "cutouts"
    overlay_root = output_root / "overlays"
    crop_root.mkdir()
    cutout_root.mkdir()
    overlay_root.mkdir()
    annotation_refiner = (
        _SamAnnotationRefiner(annotation_model) if annotation_model is not None else None
    )
    classification_rows = []
    detection_rows = []
    provenance_rows = []
    overlay_groups: dict[int, list[Path]] = {}
    for image_id, record in enumerate(records, start=1):
        source_path = source_root / record["image_path"]
        source = _rgb(source_path)
        alpha, bbox, statistics = _foreground(
            source,
            reference,
            recipe,
            capture_index=int(record["capture_index"]),
        )
        override = overrides_by_image_id.get(image_id)
        if override is not None:
            if override["original_image_sha256"] != record["image_sha256"]:
                raise ValueError(f"annotation override SHA-256 mismatch for image_id={image_id}")
            point_x, point_y = (float(value) for value in override["point_xy"])
            if not (0 <= point_x < source.shape[1] and 0 <= point_y < source.shape[0]):
                raise ValueError(f"annotation override point is outside image_id={image_id}")
            statistics = {
                **statistics,
                "prompt_x": point_x,
                "prompt_y": point_y,
                "manual_prompt_override": 1.0,
            }
            if "box_xyxy" in override:
                manual_box = [float(value) for value in override["box_xyxy"]]
                left, top, right, bottom = manual_box
                if not (0 <= left < right <= source.shape[1]):
                    raise ValueError(
                        f"annotation override box x bounds are invalid for image_id={image_id}"
                    )
                if not (0 <= top < bottom <= source.shape[0]):
                    raise ValueError(
                        f"annotation override box y bounds are invalid for image_id={image_id}"
                    )
                if not (left <= point_x < right and top <= point_y < bottom):
                    raise ValueError(
                        f"annotation override point is outside box for image_id={image_id}"
                    )
                statistics.update(
                    {
                        "manual_prompt_left": left,
                        "manual_prompt_top": top,
                        "manual_prompt_right": right,
                        "manual_prompt_bottom": bottom,
                    }
                )
        if annotation_refiner is not None:
            alpha, bbox, statistics = annotation_refiner.refine(source, statistics, recipe)
        left, top, right, bottom = bbox
        class_directory = f"bread_{int(record['category_id']):02d}"
        crop_directory = crop_root / class_directory
        cutout_directory = cutout_root / class_directory
        overlay_directory = overlay_root / class_directory
        crop_directory.mkdir(exist_ok=True)
        cutout_directory.mkdir(exist_ok=True)
        overlay_directory.mkdir(exist_ok=True)
        stem = f"{image_id:04d}"
        crop_path = crop_directory / f"{stem}.png"
        cutout_path = cutout_directory / f"{stem}.png"
        overlay_path = overlay_directory / f"{stem}.jpg"
        Image.fromarray(source[top:bottom, left:right]).save(crop_path)
        rgba = np.dstack((source, alpha))[top:bottom, left:right]
        Image.fromarray(rgba, mode="RGBA").save(cutout_path)
        overlay_array = source.copy()
        foreground = alpha >= 128
        overlay_array[foreground] = np.rint(
            overlay_array[foreground].astype(np.float32) * 0.72
            + np.asarray([0, 255, 80], dtype=np.float32) * 0.28
        ).astype(np.uint8)
        overlay = Image.fromarray(overlay_array).convert("RGB")
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 0, 0), width=8)
        overlay.thumbnail((640, 360), Image.Resampling.LANCZOS)
        overlay.save(overlay_path, quality=90)
        overlay_groups.setdefault(int(record["category_id"]), []).append(overlay_path)
        derived_sha256 = _sha256(crop_path)
        classification_rows.append(
            {
                **record,
                "image_path": crop_path.relative_to(output_root).as_posix(),
                "image_sha256": derived_sha256,
                "original_image_path": record["image_path"],
                "original_image_sha256": record["image_sha256"],
                "perceptual_group_id": f"store2:{record['image_sha256'][:16]}",
                "fold": None,
                "split": "train_support",
                "width": right - left,
                "height": bottom - top,
            }
        )
        detection_rows.append(
            {
                "record_type": "detection",
                "source": "bread_store2_single_original",
                "source_dataset": metadata["dataset_version"],
                "image_id": image_id,
                "image_path": record["image_path"],
                "image_sha256": record["image_sha256"],
                "capture_session_id": record["capture_session_id"],
                "physical_item_ids": [record["physical_item_id"]],
                "split": "train",
                "fold": None,
                "width": int(source.shape[1]),
                "height": int(source.shape[0]),
                "annotations": [
                    {
                        "annotation_id": image_id,
                        "category_id": int(record["category_id"]),
                        "bbox_xywh": [left, top, right - left, bottom - top],
                        "area": (right - left) * (bottom - top),
                        "iscrowd": 0,
                    }
                ],
            }
        )
        provenance_rows.append(
            {
                "image_id": image_id,
                "original_image_path": record["image_path"],
                "original_image_sha256": record["image_sha256"],
                "crop_path": crop_path.relative_to(output_root).as_posix(),
                "crop_sha256": derived_sha256,
                "cutout_path": cutout_path.relative_to(output_root).as_posix(),
                "cutout_sha256": _sha256(cutout_path),
                "bbox_xyxy": list(bbox),
                "statistics": statistics,
            }
        )
    for offset, row in enumerate(metadata["background_records"], start=1):
        detection_rows.append(
            {
                "record_type": "detection",
                "source": "bread_store2_empty_tray",
                "source_dataset": metadata["dataset_version"],
                "image_id": len(records) + offset,
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "capture_session_id": row["capture_session_id"],
                "physical_item_ids": [],
                "split": "train",
                "fold": None,
                "width": row["width"],
                "height": row["height"],
                "annotations": [],
            }
        )
    for category_id, paths in overlay_groups.items():
        _write_contact_sheet(paths, output_root / "qa" / f"bread_{category_id:02d}.jpg")
    classification_body = "".join(_canonical_json(row) + "\n" for row in classification_rows)
    detection_body = "".join(_canonical_json(row) + "\n" for row in detection_rows)
    provenance_body = "".join(_canonical_json(row) + "\n" for row in provenance_rows)
    (output_root / "classifier-manifest.jsonl").write_text(
        classification_body, encoding="utf-8", newline="\n"
    )
    (output_root / "detector-real-manifest.jsonl").write_text(
        detection_body, encoding="utf-8", newline="\n"
    )
    (output_root / "provenance.jsonl").write_text(provenance_body, encoding="utf-8", newline="\n")
    prepared_metadata = {
        **metadata,
        "foreground_recipe": asdict(recipe),
        "classifier_manifest_sha256": hashlib.sha256(classification_body.encode()).hexdigest(),
        "detector_manifest_sha256": hashlib.sha256(detection_body.encode()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_body.encode()).hexdigest(),
        "actual_access_original_sha256": sorted(
            [row["image_sha256"] for row in records]
            + [row["image_sha256"] for row in metadata["background_records"]]
        ),
        "annotation_review_required": True,
        "annotation_assistant": (
            {
                "role": "annotation_only_not_training_or_runtime",
                "model": annotation_refiner.model_name,
                "revision": annotation_refiner.revision,
            }
            if annotation_refiner is not None
            else None
        ),
        "annotation_override_count": len(override_rows),
        "annotation_overrides_sha256": (
            hashlib.sha256(_canonical_json(annotation_overrides).encode()).hexdigest()
            if annotation_overrides is not None
            else None
        ),
        "external_training_images_accessed": False,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(prepared_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return prepared_metadata


def approve_store2_annotations(
    prepared_root: Path,
    *,
    reviewer: str,
    reviewed_class_ids: list[int],
    findings: str,
) -> dict[str, Any]:
    """Bind a completed visual review to the exact prepared annotation payload."""
    if sorted(reviewed_class_ids) != list(range(1, EXPECTED_CLASS_COUNT + 1)):
        raise ValueError("annotation approval requires visual review of all 20 classes")
    metadata_path = prepared_root / "metadata.json"
    provenance_path = prepared_root / "provenance.jsonl"
    detection_path = prepared_root / "detector-real-manifest.jsonl"
    metadata = load_json_config(metadata_path)
    provenance_rows = [
        json.loads(line)
        for line in provenance_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    detection_rows = [
        json.loads(line) for line in detection_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(provenance_rows) != EXPECTED_CLASS_COUNT * EXPECTED_SHOTS_PER_CLASS:
        raise ValueError("annotation approval requires exactly 200 provenance rows")
    if len(detection_rows) != 210:
        raise ValueError("annotation approval requires exactly 210 detector rows")
    detection_by_id = {int(row["image_id"]): row for row in detection_rows}
    for row in provenance_rows:
        image_id = int(row["image_id"])
        cutout_path = prepared_root / row["cutout_path"]
        if _sha256(cutout_path) != row["cutout_sha256"]:
            raise ValueError(f"cutout checksum mismatch for image_id={image_id}")
        with Image.open(cutout_path) as opened:
            cutout = np.asarray(opened.convert("RGBA"))
        alpha = cutout[..., 3] >= 128
        if not np.any(alpha):
            raise ValueError(f"cutout has empty mask for image_id={image_id}")
        left, top, right, bottom = (int(value) for value in row["bbox_xyxy"])
        if cutout.shape[1] != right - left or cutout.shape[0] != bottom - top:
            raise ValueError(f"cutout dimensions disagree with bbox for image_id={image_id}")
        detection = detection_by_id.get(image_id)
        if detection is None or len(detection["annotations"]) != 1:
            raise ValueError(f"detector annotation is missing for image_id={image_id}")
        if detection["annotations"][0]["bbox_xywh"] != [
            left,
            top,
            right - left,
            bottom - top,
        ]:
            raise ValueError(f"detector bbox disagrees with provenance for image_id={image_id}")
    qa_rows = []
    for category_id in reviewed_class_ids:
        path = prepared_root / "qa" / f"bread_{category_id:02d}.jpg"
        if not path.is_file():
            raise ValueError(f"annotation review sheet is missing: {path.name}")
        qa_rows.append(
            {
                "category_id": category_id,
                "path": path.relative_to(prepared_root).as_posix(),
                "sha256": _sha256(path),
                "reviewed_object_count": EXPECTED_SHOTS_PER_CLASS,
                "status": "APPROVED",
            }
        )
    review = {
        "schema_version": "1.0",
        "status": "APPROVED",
        "reviewer": reviewer,
        "review_method": "manual_visual_inspection_of_mask_and_bbox_overlays",
        "review_criteria": [
            "mask contains the complete product silhouette",
            "mask excludes tray, table, furniture, and detached shadow",
            "bbox tightly encloses the accepted mask",
        ],
        "findings": findings,
        "reviewed_class_count": EXPECTED_CLASS_COUNT,
        "reviewed_object_count": EXPECTED_CLASS_COUNT * EXPECTED_SHOTS_PER_CLASS,
        "manual_override_count": int(metadata.get("annotation_override_count", 0)),
        "source_image_set_sha256": metadata["source_image_set_sha256"],
        "prepared_metadata_sha256": _sha256(metadata_path),
        "provenance_sha256": _sha256(provenance_path),
        "detector_manifest_sha256": _sha256(detection_path),
        "qa_sheets": qa_rows,
    }
    review_path = prepared_root / "annotation-review.json"
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return review


def _validated_annotation_review(prepared_root: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    review_path = prepared_root / "annotation-review.json"
    if not review_path.is_file():
        raise ValueError("synthetic generation requires approved annotation-review.json")
    review = load_json_config(review_path)
    if review.get("status") != "APPROVED" or review.get("reviewed_object_count") != 200:
        raise ValueError("annotation review is not approved for all 200 objects")
    if review.get("source_image_set_sha256") != metadata.get("source_image_set_sha256"):
        raise ValueError("annotation review targets a different source image set")
    expected_hashes = {
        "prepared_metadata_sha256": _sha256(prepared_root / "metadata.json"),
        "provenance_sha256": _sha256(prepared_root / "provenance.jsonl"),
        "detector_manifest_sha256": _sha256(prepared_root / "detector-real-manifest.jsonl"),
    }
    for field, expected in expected_hashes.items():
        if review.get(field) != expected:
            raise ValueError(f"annotation review binding mismatch: {field}")
    for row in review.get("qa_sheets", []):
        if _sha256(prepared_root / row["path"]) != row["sha256"]:
            raise ValueError(f"annotation review sheet changed: {row['path']}")
    if len(review.get("qa_sheets", [])) != EXPECTED_CLASS_COUNT:
        raise ValueError("annotation review must bind all 20 QA sheets")
    return review


@dataclass(frozen=True)
class Store2SyntheticRecipe:
    image_count: int = 6000
    image_size: int = 640
    empty_probability: float = 0.08
    single_probability: float = 0.07
    minimum_objects: int = 3
    maximum_objects: int = 7
    minimum_scale: float = 0.10
    maximum_scale: float = 0.44
    hard_scene_probability: float = 0.35
    easy_scene_probability: float = 0.25
    border_probability: float = 0.08
    duplicate_class_probability: float = 0.16
    medium_maximum_overlap: float = 0.24
    hard_maximum_overlap: float = 0.42
    hard_cluster_probability: float = 0.65
    maximum_rotation_degrees: float = 180.0
    side_view_probability: float = 0.0
    side_view_minimum_compression: float = 0.30
    side_view_maximum_compression: float = 0.58
    placement_margin_ratio: float = 0.07
    qa_image_count: int = 100
    seed: int = 20260901


def generate_store2_synthetic_dataset(
    source_root: Path,
    prepared_root: Path,
    output_root: Path,
    *,
    recipe: Store2SyntheticRecipe = Store2SyntheticRecipe(),
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    if recipe.image_count < 1:
        raise ValueError("synthetic image_count must be positive")
    if not 0 <= recipe.empty_probability + recipe.single_probability < 1:
        raise ValueError("empty and single probabilities must sum to less than one")
    if not 0 <= recipe.easy_scene_probability + recipe.hard_scene_probability <= 1:
        raise ValueError("easy and hard scene probabilities must sum to at most one")
    if not 1 <= recipe.minimum_objects <= recipe.maximum_objects:
        raise ValueError("synthetic object count range is invalid")
    for name, value in (
        ("medium_maximum_overlap", recipe.medium_maximum_overlap),
        ("hard_maximum_overlap", recipe.hard_maximum_overlap),
        ("hard_cluster_probability", recipe.hard_cluster_probability),
        ("side_view_probability", recipe.side_view_probability),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"synthetic {name} must be between zero and one")
    if not (0 < recipe.side_view_minimum_compression <= recipe.side_view_maximum_compression <= 1):
        raise ValueError("synthetic side-view compression range is invalid")
    metadata = load_json_config(prepared_root / "metadata.json")
    annotation_review = _validated_annotation_review(prepared_root, metadata)
    records = [
        json.loads(line)
        for line in (prepared_root / "classifier-manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    provenance = [
        json.loads(line)
        for line in (prepared_root / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    provenance_by_original = {row["original_image_sha256"]: row for row in provenance}
    cutouts: list[tuple[Image.Image, dict[str, Any], dict[str, Any]]] = []
    cutouts_by_category: dict[int, list[int]] = {}
    for record in records:
        row = provenance_by_original[record["original_image_sha256"]]
        with Image.open(prepared_root / row["cutout_path"]) as opened:
            cutouts.append((opened.convert("RGBA").copy(), record, row))
        cutouts_by_category.setdefault(int(record["category_id"]), []).append(len(cutouts) - 1)
    backgrounds = []
    for row in metadata["background_records"]:
        with Image.open(source_root / row["image_path"]) as opened:
            backgrounds.append((ImageOps.fit(opened.convert("RGB"), (640, 640)).copy(), row))
    output_root.mkdir(parents=True)
    image_root = output_root / "images"
    image_root.mkdir()
    qa_root = output_root / "qa"
    qa_root.mkdir()
    rng = np.random.default_rng(recipe.seed)
    manifest_rows = []
    provenance_rows = []
    qa_paths = []
    annotation_id = 1
    for index in range(recipe.image_count):
        background_index = int(rng.integers(0, len(backgrounds)))
        background, background_row = backgrounds[background_index]
        canvas = background.copy()
        canvas = ImageEnhance.Brightness(canvas).enhance(float(rng.uniform(0.72, 1.28)))
        canvas = ImageEnhance.Contrast(canvas).enhance(float(rng.uniform(0.82, 1.18)))
        canvas = ImageEnhance.Color(canvas).enhance(float(rng.uniform(0.82, 1.16)))
        count_draw = float(rng.random())
        if count_draw < recipe.empty_probability:
            object_count = 0
            scene_tier = "empty"
        elif count_draw < recipe.empty_probability + recipe.single_probability:
            object_count = 1
            scene_tier = "single"
        else:
            object_count = int(rng.integers(recipe.minimum_objects, recipe.maximum_objects + 1))
            tier_draw = float(rng.random())
            if tier_draw < recipe.easy_scene_probability:
                scene_tier = "easy"
            elif tier_draw < recipe.easy_scene_probability + recipe.hard_scene_probability:
                scene_tier = "hard"
            else:
                scene_tier = "medium"
        selected: list[int] = []
        selected_categories: list[int] = []
        categories = sorted(cutouts_by_category)
        for _ in range(object_count):
            if selected_categories and rng.random() < recipe.duplicate_class_probability:
                category_id = int(rng.choice(selected_categories))
            else:
                available = [value for value in categories if value not in selected_categories]
                category_id = int(rng.choice(available or categories))
            selected_categories.append(category_id)
            selected.append(int(rng.choice(cutouts_by_category[category_id])))
        boxes = []
        annotations = []
        sources = []
        for source_index in selected:
            source, record, source_provenance = cutouts[int(source_index)]
            transformed = source.copy()
            mirrored = bool(rng.random() < 0.5)
            if mirrored:
                transformed = ImageOps.mirror(transformed)
            side_view_compression = 1.0
            if rng.random() < recipe.side_view_probability:
                side_view_compression = float(
                    rng.uniform(
                        recipe.side_view_minimum_compression,
                        recipe.side_view_maximum_compression,
                    )
                )
                transformed = transformed.resize(
                    (
                        transformed.width,
                        max(2, round(transformed.height * side_view_compression)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            rotation_degrees = float(
                rng.uniform(
                    -recipe.maximum_rotation_degrees,
                    recipe.maximum_rotation_degrees,
                )
            )
            transformed = transformed.rotate(
                rotation_degrees,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            visible = transformed.getchannel("A").getbbox()
            if visible is None:
                continue
            transformed = transformed.crop(visible)
            if scene_tier == "hard":
                scale_low = recipe.minimum_scale
                scale_high = min(recipe.maximum_scale, 0.34)
            elif scene_tier == "easy":
                scale_low = max(recipe.minimum_scale, 0.18)
                scale_high = recipe.maximum_scale
            else:
                scale_low = max(recipe.minimum_scale, 0.14)
                scale_high = min(recipe.maximum_scale, 0.39)
            target_scale = float(rng.uniform(scale_low, scale_high))
            if rng.random() < 0.08:
                target_scale = float(rng.uniform(recipe.minimum_scale * 0.8, 0.15))
            elif rng.random() < 0.08:
                target_scale = float(rng.uniform(0.38, recipe.maximum_scale))
            target = round(recipe.image_size * target_scale)
            ratio = target / max(transformed.size)
            transformed = transformed.resize(
                (
                    max(2, round(transformed.width * ratio)),
                    max(2, round(transformed.height * ratio)),
                ),
                Image.Resampling.LANCZOS,
            )
            visible = transformed.getchannel("A").getbbox()
            if visible is None:
                continue
            transformed = transformed.crop(visible)
            alpha = transformed.getchannel("A")
            transformed_rgb = ImageEnhance.Brightness(transformed.convert("RGB")).enhance(
                float(rng.uniform(0.82, 1.18))
            )
            transformed_rgb = ImageEnhance.Contrast(transformed_rgb).enhance(
                float(rng.uniform(0.88, 1.14))
            )
            transformed_rgb = ImageEnhance.Color(transformed_rgb).enhance(
                float(rng.uniform(0.88, 1.12))
            )
            transformed = transformed_rgb.convert("RGBA")
            transformed.putalpha(alpha)
            selected_box = None
            selected_position = None
            maximum_overlap = {
                "empty": 0.0,
                "single": 0.0,
                "easy": 0.05,
                "medium": recipe.medium_maximum_overlap,
                "hard": recipe.hard_maximum_overlap,
            }[scene_tier]
            for _ in range(80):
                border_placement = rng.random() < recipe.border_probability
                if border_placement:
                    edge = int(rng.integers(0, 4))
                    left = int(rng.integers(0, max(1, recipe.image_size - transformed.width + 1)))
                    top = int(rng.integers(0, max(1, recipe.image_size - transformed.height + 1)))
                    clip_x = max(1, round(transformed.width * float(rng.uniform(0.03, 0.16))))
                    clip_y = max(1, round(transformed.height * float(rng.uniform(0.03, 0.16))))
                    if edge == 0:
                        left = -clip_x
                    elif edge == 1:
                        left = recipe.image_size - transformed.width + clip_x
                    elif edge == 2:
                        top = -clip_y
                    else:
                        top = recipe.image_size - transformed.height + clip_y
                elif (
                    scene_tier == "hard"
                    and boxes
                    and rng.random() < recipe.hard_cluster_probability
                ):
                    anchor = boxes[int(rng.integers(0, len(boxes)))]
                    left = round(
                        rng.uniform(
                            anchor[0] - transformed.width * 0.55,
                            anchor[2] - transformed.width * 0.45,
                        )
                    )
                    top = round(
                        rng.uniform(
                            anchor[1] - transformed.height * 0.55,
                            anchor[3] - transformed.height * 0.45,
                        )
                    )
                else:
                    placement_margin = round(recipe.image_size * recipe.placement_margin_ratio)
                    minimum_left = min(
                        placement_margin,
                        max(0, recipe.image_size - transformed.width),
                    )
                    maximum_left = max(
                        minimum_left,
                        recipe.image_size - placement_margin - transformed.width,
                    )
                    minimum_top = min(
                        placement_margin,
                        max(0, recipe.image_size - transformed.height),
                    )
                    maximum_top = max(
                        minimum_top,
                        recipe.image_size - placement_margin - transformed.height,
                    )
                    left = int(rng.integers(minimum_left, maximum_left + 1))
                    top = int(rng.integers(minimum_top, maximum_top + 1))
                right = left + transformed.width
                bottom = top + transformed.height
                box = (
                    max(0, left),
                    max(0, top),
                    min(recipe.image_size, right),
                    min(recipe.image_size, bottom),
                )
                visible_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
                if visible_area < 0.70 * transformed.width * transformed.height:
                    continue
                if all(
                    max(0, min(box[2], other[2]) - max(box[0], other[0]))
                    * max(0, min(box[3], other[3]) - max(box[1], other[1]))
                    <= maximum_overlap
                    * min(
                        (box[2] - box[0]) * (box[3] - box[1]),
                        (other[2] - other[0]) * (other[3] - other[1]),
                    )
                    for other in boxes
                ):
                    selected_box = box
                    selected_position = (left, top)
                    break
            if selected_box is None:
                continue
            assert selected_position is not None
            paste_left, paste_top = selected_position
            left, top, right, bottom = selected_box
            alpha = transformed.getchannel("A")
            if rng.random() < 0.72:
                shadow_opacity = float(rng.uniform(0.12, 0.30))
                shadow_alpha = alpha.filter(
                    ImageFilter.GaussianBlur(float(rng.uniform(2.0, 7.0)))
                ).point(lambda value: round(value * shadow_opacity))
                shadow = Image.new("RGB", transformed.size, (20, 14, 8))
                shadow_offset = (
                    paste_left + int(rng.integers(2, 10)),
                    paste_top + int(rng.integers(3, 13)),
                )
                canvas.paste(shadow, shadow_offset, shadow_alpha)
            canvas.paste(transformed.convert("RGB"), (paste_left, paste_top), alpha)
            boxes.append(selected_box)
            annotations.append(
                {
                    "annotation_id": annotation_id,
                    "category_id": int(record["category_id"]),
                    "bbox_xywh": [left, top, right - left, bottom - top],
                    "area": (right - left) * (bottom - top),
                    "iscrowd": 0,
                }
            )
            sources.append(
                {
                    "original_image_sha256": record["original_image_sha256"],
                    "cutout_sha256": source_provenance["cutout_sha256"],
                    "category_id": int(record["category_id"]),
                    "scale": target_scale,
                    "rotation_degrees": rotation_degrees,
                    "mirrored": mirrored,
                    "side_view_compression": side_view_compression,
                    "placement_xy": [paste_left, paste_top],
                }
            )
            annotation_id += 1
        if object_count and not annotations:
            raise RuntimeError("synthetic frame lost every selected object")
        if rng.random() < 0.20:
            canvas = canvas.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 1.1))))
        path = image_root / f"synthetic_{index + 1:05d}.jpg"
        canvas.save(path, quality=int(rng.integers(84, 98)))
        if index < recipe.qa_image_count:
            qa_image = canvas.copy()
            qa_draw = ImageDraw.Draw(qa_image)
            for box in boxes:
                qa_draw.rectangle(box, outline=(255, 0, 0), width=4)
            qa_path = qa_root / f"synthetic_{index + 1:05d}.jpg"
            qa_image.save(qa_path, quality=90)
            qa_paths.append(qa_path)
        digest = _sha256(path)
        manifest_rows.append(
            {
                "record_type": "detection",
                "source": "bread_store2_source_only_composite",
                "source_dataset": metadata["dataset_version"],
                "image_id": index + 1,
                "image_path": path.relative_to(output_root).as_posix(),
                "image_sha256": digest,
                "capture_session_id": f"synthetic:{index // 3:05d}",
                "physical_item_ids": sorted(
                    {f"bread_{row['category_id']:02d}:store2:item" for row in sources}
                ),
                "split": "train",
                "fold": None,
                "width": recipe.image_size,
                "height": recipe.image_size,
                "annotations": annotations,
            }
        )
        provenance_rows.append(
            {
                "image_id": index + 1,
                "image_sha256": digest,
                "background_original_sha256": background_row["image_sha256"],
                "sources": sources,
                "seed": recipe.seed,
                "scene_tier": scene_tier,
            }
        )
    manifest_body = "".join(_canonical_json(row) + "\n" for row in manifest_rows)
    provenance_body = "".join(_canonical_json(row) + "\n" for row in provenance_rows)
    (output_root / "manifest.jsonl").write_text(manifest_body, encoding="utf-8", newline="\n")
    (output_root / "provenance.jsonl").write_text(provenance_body, encoding="utf-8", newline="\n")
    for sheet_index in range(0, len(qa_paths), 20):
        _write_contact_sheet(
            qa_paths[sheet_index : sheet_index + 20],
            qa_root / f"contact_{sheet_index // 20 + 1:02d}.jpg",
        )
    result = {
        "schema_version": "1.0",
        "dataset_version": metadata["dataset_version"],
        "training_source_policy": "single_objects_4-only",
        "source_original_count": 210,
        "synthetic_image_count": len(manifest_rows),
        "synthetic_annotation_count": sum(len(row["annotations"]) for row in manifest_rows),
        "synthetic_empty_image_count": sum(not row["annotations"] for row in manifest_rows),
        "scene_tier_counts": dict(Counter(row["scene_tier"] for row in provenance_rows)),
        "object_count_distribution": dict(
            Counter(str(len(row["annotations"])) for row in manifest_rows)
        ),
        "category_distribution": dict(
            Counter(
                str(annotation["category_id"])
                for row in manifest_rows
                for annotation in row["annotations"]
            )
        ),
        "recipe": asdict(recipe),
        "manifest_sha256": hashlib.sha256(manifest_body.encode()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_body.encode()).hexdigest(),
        "external_training_images_accessed": False,
        "annotation_review_sha256": _sha256(prepared_root / "annotation-review.json"),
        "annotation_reviewed_object_count": annotation_review["reviewed_object_count"],
    }
    (output_root / "metadata.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for image, _, _ in cutouts:
        image.close()
    for image, _ in backgrounds:
        image.close()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the locked second-store source dataset")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--synthetic-output", type=Path)
    parser.add_argument("--annotation-model")
    parser.add_argument("--annotation-overrides", type=Path)
    args = parser.parse_args(argv)
    overrides = (
        load_json_config(args.annotation_overrides)
        if args.annotation_overrides is not None
        else None
    )
    prepared = prepare_store2_source(
        args.source_root,
        args.output_root,
        annotation_model=args.annotation_model,
        annotation_overrides=overrides,
    )
    result: dict[str, Any] = {"prepared": prepared}
    if args.synthetic_output is not None:
        result["synthetic"] = generate_store2_synthetic_dataset(
            args.source_root, args.output_root, args.synthetic_output
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
