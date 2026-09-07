from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from bixolon_scanner.operations.catalog_activation import fit_diagonal_lda_adapter
from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from bixolon_scanner.training.models import (
    DINO_V3_HUB_REPOSITORY,
    build_dino_classifier,
    require_torch,
)
from bixolon_scanner.training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    prepare_direct_roi_source,
)

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tensor(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (values - MEAN) / STD


def _support_tensor(image: Image.Image, size: int, crop_mode: str) -> np.ndarray:
    if crop_mode == "box_resize":
        return _tensor(image, size)
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb)
    border = np.concatenate((pixels[0], pixels[-1], pixels[1:-1, 0], pixels[1:-1, -1]))
    fill = tuple(int(value) for value in np.median(border, axis=0))
    squared = ImageOps.pad(
        rgb,
        (max(rgb.width, rgb.height),) * 2,
        method=Image.Resampling.BICUBIC,
        color=fill,
    )
    return _tensor(squared, size)


def _color_constancy(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate((rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]), axis=0)
    background = np.median(border, axis=0).clip(min=16.0)
    neutral = float(background.mean())
    channel_gain = np.clip(neutral / background, 0.7, 1.4)
    exposure_gain = float(np.clip(200.0 / neutral, 0.7, 1.5))
    corrected = np.clip(rgb * channel_gain * exposure_gain, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(corrected, mode="RGB")


def _expanded_crop(image: Image.Image, bbox: list[float], margin: float) -> Image.Image:
    x, y, width, height = bbox
    margin_x = width * margin
    margin_y = height * margin
    box = (
        max(0.0, x - margin_x),
        max(0.0, y - margin_y),
        min(float(image.width), x + width + margin_x),
        min(float(image.height), y + height + margin_y),
    )
    return image.crop(box)


def _features(
    model,
    tensors: list[np.ndarray],
    *,
    batch_size: int,
    rotation_ensemble: int,
    exposure_ensemble: int,
) -> np.ndarray:
    torch = require_torch()
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(np.asarray(tensors[start : start + batch_size])).cuda()
            rotations = (
                (0,)
                if rotation_ensemble == 1
                else ((0, 2) if rotation_ensemble == 2 else (0, 1, 2, 3))
            )
            exposure_gains = (
                (1.0,)
                if exposure_ensemble == 1
                else ((0.8, 1.0, 1.2) if exposure_ensemble == 3 else (0.7, 0.85, 1.0, 1.15, 1.3))
            )
            mean = torch.tensor(MEAN[:, 0, 0], device=batch.device)[None, :, None, None]
            std = torch.tensor(STD[:, 0, 0], device=batch.device)[None, :, None, None]
            values = torch.stack(
                [
                    model.extract_features(
                        torch.rot90(
                            (((batch * std + mean) * gain).clamp(0.0, 1.0) - mean) / std,
                            turns,
                            dims=(-2, -1),
                        )
                    ).float()
                    for turns in rotations
                    for gain in exposure_gains
                ],
                dim=0,
            ).mean(dim=0)
            values = torch.nn.functional.normalize(values, dim=-1)
            output.append(values.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _rotation_head_scores(model, head, tensors: list[np.ndarray], *, batch_size: int) -> np.ndarray:
    torch = require_torch()
    output = []
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(np.asarray(tensors[start : start + batch_size])).cuda()
            scores = torch.stack(
                [
                    head(model.extract_features(torch.rot90(batch, turns, dims=(-2, -1)))).float()
                    for turns in range(4)
                ],
                dim=1,
            )
            output.append(scores.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _patch_features(model, tensors: list[np.ndarray], *, batch_size: int) -> np.ndarray:
    torch = require_torch()
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(np.asarray(tensors[start : start + batch_size])).cuda()
            values = model.extract_patch_features(batch).float()
            values = torch.nn.functional.normalize(values, dim=-1)
            output.append(values.cpu().numpy().astype(np.float16))
    return np.concatenate(output, axis=0)


def _local_patch_scores(
    support: np.ndarray,
    support_labels: np.ndarray,
    evaluation: np.ndarray,
    *,
    visible_fraction: float,
    batch_size: int,
) -> np.ndarray:
    torch = require_torch()
    class_banks = [
        torch.from_numpy(support[support_labels == class_id].reshape(-1, support.shape[-1]))
        .cuda()
        .half()
        for class_id in range(20)
    ]
    visible_count = max(1, round(evaluation.shape[1] * visible_fraction))
    output = []
    with torch.inference_mode():
        for start in range(0, len(evaluation), batch_size):
            query = torch.from_numpy(evaluation[start : start + batch_size]).cuda().half()
            class_scores = []
            for bank in class_banks:
                best_per_query_patch = torch.matmul(query, bank.transpose(0, 1)).amax(dim=-1)
                class_scores.append(
                    best_per_query_patch.topk(visible_count, dim=-1).values.mean(dim=-1)
                )
            output.append(torch.stack(class_scores, dim=-1).float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _local_patch_exemplar_scores(
    support: np.ndarray,
    support_labels: np.ndarray,
    evaluation: np.ndarray,
    *,
    visible_fraction: float,
    batch_size: int,
) -> np.ndarray:
    """Match query patches coherently to individual support views before class pooling."""
    torch = require_torch()
    class_banks = [
        torch.from_numpy(support[support_labels == class_id]).cuda().half()
        for class_id in range(20)
    ]
    visible_count = max(1, round(evaluation.shape[1] * visible_fraction))
    output = []
    with torch.inference_mode():
        for start in range(0, len(evaluation), batch_size):
            query = torch.from_numpy(evaluation[start : start + batch_size]).cuda().half()
            class_scores = []
            for bank in class_banks:
                similarities = torch.einsum("bqd,epd->bqep", query, bank)
                query_scores = similarities.amax(dim=-1)
                exemplar_scores = query_scores.topk(visible_count, dim=1).values.mean(dim=1)
                exemplar_count = min(3, exemplar_scores.shape[1])
                class_scores.append(exemplar_scores.topk(exemplar_count, dim=1).values.mean(dim=1))
            output.append(torch.stack(class_scores, dim=-1).float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _local_patch_symmetric_scores(
    support: np.ndarray,
    support_labels: np.ndarray,
    evaluation: np.ndarray,
    *,
    visible_fraction: float,
    batch_size: int,
) -> np.ndarray:
    """Require both query coverage and complete-support structural coverage."""
    torch = require_torch()
    class_banks = [
        torch.from_numpy(support[support_labels == class_id]).cuda().half()
        for class_id in range(20)
    ]
    visible_count = max(1, round(evaluation.shape[1] * visible_fraction))
    output = []
    with torch.inference_mode():
        for start in range(0, len(evaluation), batch_size):
            query = torch.from_numpy(evaluation[start : start + batch_size]).cuda().half()
            class_scores = []
            for bank in class_banks:
                similarities = torch.einsum("bqd,epd->bqep", query, bank)
                query_to_support = similarities.amax(dim=-1)
                query_to_support = query_to_support.topk(visible_count, dim=1).values.mean(dim=1)
                support_to_query = similarities.amax(dim=1).mean(dim=-1)
                symmetric = 0.5 * (query_to_support + support_to_query)
                exemplar_count = min(3, symmetric.shape[1])
                class_scores.append(symmetric.topk(exemplar_count, dim=1).values.mean(dim=1))
            output.append(torch.stack(class_scores, dim=-1).float().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def _handcrafted_features(tensors: list[np.ndarray]) -> np.ndarray:
    import cv2

    output = []
    mean = MEAN[:, 0, 0]
    std = STD[:, 0, 0]
    for tensor in tensors:
        rgb = np.clip((tensor * std[:, None, None] + mean[:, None, None]) * 255.0, 0, 255)
        rgb = rgb.transpose(1, 2, 0).astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        border = np.concatenate((lab[0], lab[-1], lab[1:-1, 0], lab[1:-1, -1]), axis=0)
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab.astype(np.float32) - background, axis=2)
        mask = (distance > 13.0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if component_count > 1:
            selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == selected).astype(np.uint8)
        if int(mask.sum()) < 32:
            mask[:] = 1
        selected_rgb = rgb[mask.astype(bool)].astype(np.float32) / 255.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        selected_hsv = hsv[mask.astype(bool)].astype(np.float32)
        selected_hsv /= np.asarray((180.0, 255.0, 255.0), dtype=np.float32)
        values = []
        for selected_values in (selected_rgb, selected_hsv):
            values.extend(selected_values.mean(axis=0))
            values.extend(selected_values.std(axis=0))
            values.extend(np.quantile(selected_values, (0.1, 0.25, 0.5, 0.75, 0.9), axis=0).ravel())
            for channel in range(3):
                histogram, _ = np.histogram(
                    selected_values[:, channel], bins=16, range=(0.0, 1.0), density=False
                )
                values.extend(histogram / max(1, histogram.sum()))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        area_ratio = float(mask.mean())
        if contour is None:
            values.extend((area_ratio, 1.0, 1.0, *([0.0] * 7)))
        else:
            x, y, width, height = cv2.boundingRect(contour)
            hu = cv2.HuMoments(cv2.moments(contour)).ravel()
            hu = -np.sign(hu) * np.log10(np.abs(hu).clip(min=1e-12))
            values.extend(
                (
                    area_ratio,
                    width / max(height, 1),
                    float(mask.sum()) / max(width * height, 1),
                    *hu,
                )
            )
        output.append(np.asarray(values, dtype=np.float32))
    return np.stack(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose fresh DINOv3 classifier on GT ROIs")
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--additional-support-manifest", type=Path)
    parser.add_argument("--additional-support-root", type=Path)
    parser.add_argument("--support-repeat", type=int, default=1)
    parser.add_argument("--additional-support-repeat", type=int, default=1)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--maximum-evaluation-records",
        type=int,
        help="Deterministically subsample large source diagnostics before tensor loading",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        choices=(
            "dinov3_convnext_tiny",
            "dinov3_convnext_tiny_multiscale",
            "dinov3_vitb16",
        ),
        default="dinov3_convnext_tiny",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scores-output",
        type=Path,
        help="Optional per-object score bundle for source-independent cascade analysis",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        help="Optional normalized support and evaluation feature bundle for adapter research",
    )
    parser.add_argument(
        "--rotation-head-scores-output",
        type=Path,
        help="Optional per-rotation trained-head logits for source-calibrated consistency analysis",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument(
        "--crop-mode", choices=("box_resize", "square_context"), default="box_resize"
    )
    parser.add_argument("--neighbor-mask", action="store_true")
    parser.add_argument("--neighbor-context-scale", type=float, default=0.0)
    parser.add_argument("--color-constancy", action="store_true")
    parser.add_argument("--neighbor-distance-bias", type=float, default=-0.1)
    parser.add_argument("--augmented-views", type=int, default=0)
    parser.add_argument("--handcrafted", action="store_true")
    parser.add_argument(
        "--handcrafted-weight",
        type=float,
        help="Force a handcrafted feature weight selected on separate source validation",
    )
    parser.add_argument("--rotation-ensemble", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--exposure-ensemble", type=int, choices=(1, 3, 5), default=1)
    parser.add_argument("--local-patch-retrieval", action="store_true")
    parser.add_argument("--local-visible-fraction", type=float, default=0.5)
    parser.add_argument(
        "--local-patch-mode",
        choices=("class_bank", "exemplar", "symmetric"),
        default="class_bank",
    )
    parser.add_argument(
        "--decision-head",
        help=(
            "Force a decision head selected on a separate validation set; by default this "
            "diagnostic selects the most accurate head on the evaluated manifest"
        ),
    )
    args = parser.parse_args()
    if (args.additional_support_manifest is None) != (args.additional_support_root is None):
        parser.error(
            "--additional-support-manifest and --additional-support-root must be provided together"
        )
    if args.support_repeat < 1 or args.additional_support_repeat < 1:
        parser.error("support repeat values must be positive")
    if not 0.0 <= args.neighbor_context_scale <= 1.0:
        parser.error("neighbor context scale must be in [0, 1]")
    if args.augmented_views and args.handcrafted:
        parser.error("augmented views and handcrafted fusion are separate diagnostics")
    if args.handcrafted_weight is not None and not args.handcrafted:
        parser.error("--handcrafted-weight requires --handcrafted")
    if args.handcrafted_weight is not None and args.handcrafted_weight <= 0.0:
        parser.error("--handcrafted-weight must be positive")
    support_sources = [(args.support_manifest, args.support_root, args.support_repeat)]
    if args.additional_support_manifest is not None:
        support_sources.append(
            (
                args.additional_support_manifest,
                args.additional_support_root,
                args.additional_support_repeat,
            )
        )
    support_records = [
        (record, root)
        for manifest, root, repeat in support_sources
        for _ in range(repeat)
        for record in _records(manifest)
    ]
    support_tensors = []
    support_labels = []
    augmented_support_tensors = []
    augmented_support_labels = []
    recipe = DirectRoiRecipe(
        output_size=args.image_size,
        canvas_scale_min=0.72,
        canvas_scale_max=0.98,
        rotation_degrees=180.0,
        perspective_fraction=0.04,
        brightness_min=0.8,
        brightness_max=1.2,
        contrast_min=0.85,
        contrast_max=1.15,
        saturation_min=0.85,
        saturation_max=1.15,
        blur_probability=0.15,
        blur_radius_max=0.7,
        jpeg_quality_min=82,
        jpeg_quality_max=96,
        crop_mode="border_connected_composite",
        procedural_gradient=True,
        procedural_shadow=True,
    )
    for support_index, (record, support_root) in enumerate(support_records):
        with Image.open(support_root / record["image_path"]) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            if args.color_constancy:
                oriented = _color_constancy(oriented)
            support_tensors.append(_support_tensor(oriented, args.image_size, args.crop_mode))
            if args.augmented_views:
                cutout = prepare_direct_roi_source(oriented, recipe)
                for view_index in range(args.augmented_views):
                    sample = augment_direct_roi(
                        oriented,
                        source_sha256=str(record["image_sha256"]),
                        category_id=int(record["category_id"]),
                        seed=20260904 + support_index * 1000 + view_index,
                        recipe=recipe,
                        prepared_cutout=cutout,
                    )
                    augmented_support_tensors.append(_tensor(sample.image, args.image_size))
                    augmented_support_labels.append(int(record["category_id"]) - 1)
        support_labels.append(int(record["category_id"]) - 1)

    evaluation_records = _records(args.evaluation_manifest)
    if args.maximum_evaluation_records is not None:
        if args.maximum_evaluation_records < 1:
            parser.error("--maximum-evaluation-records must be positive")
        if len(evaluation_records) > args.maximum_evaluation_records:
            generator = np.random.default_rng(20261118)
            selected = np.sort(
                generator.choice(
                    len(evaluation_records),
                    args.maximum_evaluation_records,
                    replace=False,
                )
            )
            evaluation_records = [evaluation_records[int(index)] for index in selected]
    evaluation_tensors = []
    evaluation_labels = []
    evaluation_ids = []
    for record in evaluation_records:
        with Image.open(args.evaluation_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            detections = []
            for annotation in record["annotations"]:
                x, y, width, height = annotation["bbox_xywh"]
                detections.append(Detection(x, y, x + width, y + height, score=1.0))
            for detection_index, annotation in enumerate(record["annotations"]):
                if args.neighbor_mask:
                    crop_box = classifier_crop_box(
                        detections[detection_index],
                        image.width,
                        image.height,
                        margin_ratio=args.margin,
                        crop_mode=args.crop_mode,
                    )
                    crop = image.crop(crop_box)
                    if args.color_constancy:
                        crop = _color_constancy(crop)
                    tensor = prepare_rgb(
                        crop,
                        (args.image_size, args.image_size),
                        tuple(float(value) for value in MEAN[:, 0, 0]),
                        tuple(float(value) for value in STD[:, 0, 0]),
                        reducing_gap=1.0,
                    )
                    mask = classifier_neighbor_ownership_mask(
                        detections,
                        detection_index,
                        image_width=image.width,
                        image_height=image.height,
                        output_size=args.image_size,
                        margin_ratio=args.margin,
                        distance_bias=args.neighbor_distance_bias,
                        shared_scale=False,
                    )
                    masked_tensor = apply_classifier_background_masks(tensor[None], mask[None])[0]
                    evaluation_tensors.append(
                        args.neighbor_context_scale * tensor
                        + (1.0 - args.neighbor_context_scale) * masked_tensor
                    )
                else:
                    crop = _expanded_crop(image, annotation["bbox_xywh"], args.margin)
                    if args.color_constancy:
                        crop = _color_constancy(crop)
                    evaluation_tensors.append(_tensor(crop, args.image_size))
                evaluation_labels.append(int(annotation["category_id"]) - 1)
                evaluation_ids.append(
                    {
                        "image_id": int(record["image_id"]),
                        "annotation_id": int(annotation["annotation_id"]),
                    }
                )

    classifier_head = None
    classifier_output_count = 20
    classifier_checkpoint_state = None
    if args.classifier_checkpoint is not None:
        classifier_checkpoint_state = require_torch().load(
            args.classifier_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        classifier_weight = classifier_checkpoint_state.get("classifier.weight")
        if classifier_weight is None or classifier_weight.ndim != 2:
            raise ValueError("classifier checkpoint is missing classifier.weight")
        classifier_output_count = int(classifier_weight.shape[0])
        if classifier_output_count not in (2, 20, 21):
            raise ValueError(
                "classifier checkpoint must contain 2 quality outputs or 20 identity outputs "
                "with optional quality"
            )
    if args.variant.startswith("dinov3_convnext_tiny"):
        base_model = build_dino_classifier(
            "dinov3_convnext_tiny",
            classifier_output_count,
            weights_path=args.weights,
            feature_l2_normalize=True,
            classifier_head_kind=("cosine" if args.classifier_checkpoint is not None else "linear"),
        )
        if args.classifier_checkpoint is not None:
            base_model.load_state_dict(
                classifier_checkpoint_state,
                strict=True,
            )
            if args.variant == "dinov3_convnext_tiny":
                classifier_head = base_model.classifier
        if args.variant == "dinov3_convnext_tiny":
            model = base_model.cuda()
        else:
            torch = require_torch()

            class _ConvNextMultiscaleWrapper(torch.nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone

                def extract_features(self, pixel_values):
                    values = pixel_values
                    pooled = []
                    for stage_index in range(4):
                        values = self.backbone.downsample_layers[stage_index](values)
                        values = self.backbone.stages[stage_index](values)
                        stage = values.mean(dim=(-2, -1))
                        if stage_index == 3:
                            stage = self.backbone.norm(stage)
                        pooled.append(torch.nn.functional.normalize(stage, dim=-1))
                    return torch.nn.functional.normalize(torch.cat(pooled, dim=-1), dim=-1)

            model = _ConvNextMultiscaleWrapper(base_model.backbone).cuda()
    elif args.classifier_checkpoint is not None:
        base_model = build_dino_classifier(
            "dinov3_vitb16",
            classifier_output_count,
            weights_path=args.weights,
            feature_l2_normalize=True,
            classifier_head_kind="cosine",
        )
        base_model.load_state_dict(
            classifier_checkpoint_state,
            strict=True,
        )
        classifier_head = base_model.classifier
        model = base_model.cuda()
    else:
        torch = require_torch()
        backbone = torch.hub.load(
            DINO_V3_HUB_REPOSITORY,
            "dinov3_vitb16",
            source="github",
            trust_repo=True,
            verbose=False,
            pretrained=False,
        )
        backbone.load_state_dict(
            torch.load(args.weights, map_location="cpu", weights_only=True),
            strict=True,
        )

        class _VitB16Wrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def extract_features(self, pixel_values):
                return self.wrapped.forward_features(pixel_values, masks=None)["x_norm_clstoken"]

            def extract_patch_features(self, pixel_values):
                return self.wrapped.forward_features(pixel_values, masks=None)["x_norm_patchtokens"]

        model = _VitB16Wrapper(backbone).cuda()
    support_features = _features(
        model,
        support_tensors,
        batch_size=args.batch_size,
        rotation_ensemble=args.rotation_ensemble,
        exposure_ensemble=args.exposure_ensemble,
    )
    if augmented_support_tensors:
        augmented_support_features = _features(
            model,
            augmented_support_tensors,
            batch_size=args.batch_size,
            rotation_ensemble=args.rotation_ensemble,
            exposure_ensemble=args.exposure_ensemble,
        )
    else:
        augmented_support_features = np.empty((0, support_features.shape[1]), dtype=np.float32)
    evaluation_features = _features(
        model,
        evaluation_tensors,
        batch_size=args.batch_size,
        rotation_ensemble=args.rotation_ensemble,
        exposure_ensemble=args.exposure_ensemble,
    )
    support_labels_array = np.asarray(support_labels, dtype=np.int64)
    labels = np.asarray(evaluation_labels, dtype=np.int64)
    local_patch_scores = None
    if args.local_patch_retrieval:
        if args.variant != "dinov3_vitb16":
            parser.error("local patch retrieval requires dinov3_vitb16")
        support_patch_features = _patch_features(model, support_tensors, batch_size=args.batch_size)
        evaluation_patch_features = _patch_features(
            model, evaluation_tensors, batch_size=args.batch_size
        )
        if args.local_patch_mode == "exemplar":
            local_patch_scores = _local_patch_exemplar_scores(
                support_patch_features,
                support_labels_array,
                evaluation_patch_features,
                visible_fraction=args.local_visible_fraction,
                batch_size=min(args.batch_size, 16),
            )
        elif args.local_patch_mode == "symmetric":
            local_patch_scores = _local_patch_symmetric_scores(
                support_patch_features,
                support_labels_array,
                evaluation_patch_features,
                visible_fraction=args.local_visible_fraction,
                batch_size=min(args.batch_size, 16),
            )
        else:
            local_patch_scores = _local_patch_scores(
                support_patch_features,
                support_labels_array,
                evaluation_patch_features,
                visible_fraction=args.local_visible_fraction,
                batch_size=min(args.batch_size, 32),
            )
    handcrafted_sweep = None
    selected_handcrafted_weight = 0.0
    if args.handcrafted:
        support_handcrafted = _handcrafted_features(support_tensors)
        evaluation_handcrafted = _handcrafted_features(evaluation_tensors)
        center = support_handcrafted.mean(axis=0, keepdims=True)
        scale = support_handcrafted.std(axis=0, keepdims=True).clip(min=1e-4)
        support_handcrafted = (support_handcrafted - center) / scale
        evaluation_handcrafted = (evaluation_handcrafted - center) / scale
        support_handcrafted /= np.linalg.norm(support_handcrafted, axis=1, keepdims=True).clip(
            min=1e-12
        )
        evaluation_handcrafted /= np.linalg.norm(
            evaluation_handcrafted, axis=1, keepdims=True
        ).clip(min=1e-12)
        original_support = support_features
        original_evaluation = evaluation_features
        handcrafted_sweep = []
        best_accuracy = -1.0
        handcrafted_weights = (
            (args.handcrafted_weight,)
            if args.handcrafted_weight is not None
            else (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
        )
        for weight in handcrafted_weights:
            candidate_support = np.concatenate(
                (original_support, support_handcrafted * weight), axis=1
            )
            candidate_evaluation = np.concatenate(
                (original_evaluation, evaluation_handcrafted * weight), axis=1
            )
            candidate_support /= np.linalg.norm(candidate_support, axis=1, keepdims=True).clip(
                min=1e-12
            )
            candidate_evaluation /= np.linalg.norm(
                candidate_evaluation, axis=1, keepdims=True
            ).clip(min=1e-12)
            candidate_prototypes = np.stack(
                [
                    candidate_support[support_labels_array == class_id].mean(axis=0)
                    for class_id in range(20)
                ]
            )
            candidate_prototypes /= np.linalg.norm(
                candidate_prototypes, axis=1, keepdims=True
            ).clip(min=1e-12)
            candidate_scores = candidate_evaluation @ candidate_prototypes.T
            accuracy = float((candidate_scores.argmax(axis=1) == labels).mean())
            handcrafted_sweep.append({"weight": weight, "prototype_accuracy": accuracy})
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                selected_handcrafted_weight = weight
                support_features = candidate_support
                evaluation_features = candidate_evaluation
    prototypes = np.stack(
        [support_features[support_labels_array == class_id].mean(axis=0) for class_id in range(20)]
    )
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
    prototype_scores = evaluation_features @ prototypes.T
    exemplar_scores = evaluation_features @ support_features.T
    exemplar_class_scores = np.stack(
        [
            np.sort(exemplar_scores[:, support_labels_array == class_id], axis=1)[:, -3:].mean(
                axis=1
            )
            for class_id in range(20)
        ],
        axis=1,
    )
    hybrid_scores = 0.5 * prototype_scores + 0.5 * exemplar_class_scores
    if not len(augmented_support_features):
        augmented_support_features = np.empty((0, support_features.shape[1]), dtype=np.float32)
    ridge_features = np.concatenate([support_features, augmented_support_features], axis=0)
    ridge_labels = np.concatenate(
        [support_labels_array, np.asarray(augmented_support_labels, dtype=np.int64)]
    )
    design = np.concatenate(
        [ridge_features, np.ones((len(ridge_features), 1), dtype=np.float32)],
        axis=1,
    )
    targets = np.eye(20, dtype=np.float32)[ridge_labels]
    if design.shape[0] < design.shape[1]:
        centered_features = ridge_features - ridge_features.mean(axis=0, keepdims=True)
        centered_targets = targets - targets.mean(axis=0, keepdims=True)
        coefficients_without_bias = centered_features.T @ np.linalg.solve(
            centered_features @ centered_features.T
            + np.eye(len(centered_features), dtype=np.float32) * 0.01,
            centered_targets,
        )
        bias = targets.mean(axis=0) - ridge_features.mean(axis=0) @ coefficients_without_bias
        coefficients = np.concatenate([coefficients_without_bias, bias[None]], axis=0)
    else:
        regularization = np.eye(design.shape[1], dtype=np.float32) * 0.01
        regularization[-1, -1] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + regularization,
            design.T @ targets,
        )
    ridge_scores = (evaluation_features @ coefficients[:-1] + coefficients[-1]).astype(np.float32)
    lda_weight, lda_bias = fit_diagonal_lda_adapter(
        ridge_features,
        ridge_labels,
        class_count=20,
    )
    lda_scores = evaluation_features @ lda_weight + lda_bias
    hybrid_predictions = hybrid_scores.argmax(axis=1)
    ridge_predictions = ridge_scores.argmax(axis=1)
    hybrid_accuracy = float((hybrid_predictions == labels).mean())
    ridge_accuracy = float((ridge_predictions == labels).mean())
    lda_accuracy = float((lda_scores.argmax(axis=1) == labels).mean())
    classifier_head_accuracy = None
    classifier_head_scores = None
    classifier_quality_scores = None
    classifier_ensemble_sweep = None
    if classifier_head is not None and not args.handcrafted:
        torch = require_torch()
        with torch.inference_mode():
            classifier_head_outputs = (
                classifier_head(torch.from_numpy(evaluation_features).cuda()).float().cpu().numpy()
            )
        if classifier_head_outputs.shape[1] == 2:
            shifted_quality_outputs = classifier_head_outputs - classifier_head_outputs.max(
                axis=1, keepdims=True
            )
            classifier_probabilities = np.exp(shifted_quality_outputs)
            classifier_probabilities /= classifier_probabilities.sum(axis=1, keepdims=True)
            classifier_quality_scores = classifier_probabilities[:, 1]
        elif classifier_head_outputs.shape[1] == 21:
            shifted_quality_outputs = classifier_head_outputs - classifier_head_outputs.max(
                axis=1, keepdims=True
            )
            classifier_probabilities = np.exp(shifted_quality_outputs)
            classifier_probabilities /= classifier_probabilities.sum(axis=1, keepdims=True)
            classifier_quality_scores = classifier_probabilities[:, 20]
            classifier_head_scores = classifier_head_outputs[:, :20]
        else:
            classifier_head_scores = classifier_head_outputs
        if classifier_head_scores is not None:
            classifier_head_accuracy = float(
                (classifier_head_scores.argmax(axis=1) == labels).mean()
            )
    candidates = [
        (hybrid_accuracy, "prototype_knn_hybrid", hybrid_scores),
        (ridge_accuracy, "ridge_adapter", ridge_scores),
        (lda_accuracy, "diagonal_lda_adapter", lda_scores),
    ]
    local_patch_accuracy = None
    local_patch_ensemble_sweep = None
    if local_patch_scores is not None:
        local_patch_accuracy = float((local_patch_scores.argmax(axis=1) == labels).mean())
        local_patch_name = f"local_patch_{args.local_patch_mode}"
        candidates.append((local_patch_accuracy, local_patch_name, local_patch_scores))
        local_patch_ensemble_sweep = []
        for local_weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9):
            ensemble_scores = (
                1.0 - local_weight
            ) * hybrid_scores + local_weight * local_patch_scores
            accuracy = float((ensemble_scores.argmax(axis=1) == labels).mean())
            local_patch_ensemble_sweep.append({"local_weight": local_weight, "accuracy": accuracy})
            candidates.append((accuracy, f"global_local_patch_w{local_weight:g}", ensemble_scores))
    if classifier_head_scores is not None and classifier_head_accuracy is not None:
        candidates.append((classifier_head_accuracy, "trained_cosine_head", classifier_head_scores))
        classifier_ensemble_sweep = []
        shifted_head = classifier_head_scores - classifier_head_scores.max(axis=1, keepdims=True)
        head_probabilities = np.exp(shifted_head)
        head_probabilities /= head_probabilities.sum(axis=1, keepdims=True)
        for retrieval_temperature in (8.0, 16.0, 24.0, 32.0):
            shifted_retrieval = hybrid_scores * retrieval_temperature
            shifted_retrieval -= shifted_retrieval.max(axis=1, keepdims=True)
            retrieval_probabilities = np.exp(shifted_retrieval)
            retrieval_probabilities /= retrieval_probabilities.sum(axis=1, keepdims=True)
            for head_weight in (0.25, 0.4, 0.5, 0.6, 0.75):
                ensemble_scores = (
                    head_weight * head_probabilities + (1.0 - head_weight) * retrieval_probabilities
                )
                ensemble_accuracy = float((ensemble_scores.argmax(axis=1) == labels).mean())
                name = f"head_retrieval_ensemble_t{retrieval_temperature:g}_w{head_weight:g}"
                classifier_ensemble_sweep.append(
                    {
                        "retrieval_temperature": retrieval_temperature,
                        "head_weight": head_weight,
                        "accuracy": ensemble_accuracy,
                    }
                )
                candidates.append((ensemble_accuracy, name, ensemble_scores))
    if args.decision_head is None:
        _, decision_head, scores = max(candidates, key=lambda item: item[0])
        decision_head_selection = "evaluated_manifest_diagnostic"
    else:
        matches = [item for item in candidates if item[1] == args.decision_head]
        if not matches:
            available = ", ".join(item[1] for item in candidates)
            parser.error(f"unknown --decision-head {args.decision_head!r}; available: {available}")
        _, decision_head, scores = matches[0]
        decision_head_selection = "forced_from_separate_validation"
    predictions = scores.argmax(axis=1)
    order = np.argsort(scores, axis=1)[:, ::-1]
    top1 = scores[np.arange(len(labels)), order[:, 0]]
    margin = top1 - scores[np.arange(len(labels)), order[:, 1]]
    correct = predictions == labels
    per_class = {}
    for class_id in range(20):
        selected = labels == class_id
        per_class[str(class_id + 1)] = {
            "count": int(selected.sum()),
            "correct": int(correct[selected].sum()),
            "accuracy": float(correct[selected].mean()) if selected.any() else None,
        }
    errors = []
    for index in np.flatnonzero(~correct):
        errors.append(
            {
                **evaluation_ids[int(index)],
                "expected": int(labels[index] + 1),
                "predicted": int(predictions[index] + 1),
                "score": float(top1[index]),
                "margin": float(margin[index]),
                "top3": [int(value + 1) for value in order[index, :3]],
            }
        )
    report = {
        "schema_version": "1.0",
        "mode": "evaluation_gt_roi_diagnostic_only",
        "development_evaluation_used_for_head_selection": args.decision_head is None,
        "decision_head_selection": decision_head_selection,
        "support_count": len(support_records),
        "support_repeat": args.support_repeat,
        "additional_support_repeat": args.additional_support_repeat,
        "augmented_support_count": len(augmented_support_labels),
        "augmentation_recipe": asdict(recipe) if args.augmented_views else None,
        "handcrafted_feature_fusion": args.handcrafted,
        "handcrafted_weight_sweep": handcrafted_sweep,
        "selected_handcrafted_weight": selected_handcrafted_weight,
        "evaluation_object_count": len(labels),
        "method": f"official_{args.variant}_prototype_knn_hybrid",
        "selected_decision_head": decision_head,
        "prototype_knn_hybrid_accuracy": hybrid_accuracy,
        "ridge_adapter_accuracy": ridge_accuracy,
        "diagonal_lda_adapter_accuracy": lda_accuracy,
        "trained_classifier_head_accuracy": classifier_head_accuracy,
        "classifier_ensemble_sweep": classifier_ensemble_sweep,
        "local_patch_retrieval_accuracy": local_patch_accuracy,
        "local_patch_mode": args.local_patch_mode,
        "local_patch_ensemble_sweep": local_patch_ensemble_sweep,
        "local_visible_fraction": args.local_visible_fraction,
        "image_size": args.image_size,
        "crop_margin": args.margin,
        "crop_mode": args.crop_mode,
        "neighbor_mask": args.neighbor_mask,
        "neighbor_context_scale": args.neighbor_context_scale,
        "color_constancy": args.color_constancy,
        "neighbor_distance_bias": args.neighbor_distance_bias,
        "internal_rotation_ensemble": args.rotation_ensemble,
        "internal_exposure_ensemble": args.exposure_ensemble,
        "correct_count": int(correct.sum()),
        "error_count": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "correct_score_min": float(top1[correct].min()) if correct.any() else None,
        "correct_margin_min": float(margin[correct].min()) if correct.any() else None,
        "per_class": per_class,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_name("ridge-adapter.npz"),
        weight=coefficients[:-1].astype(np.float32),
        bias=coefficients[-1].astype(np.float32),
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.scores_output is not None:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.scores_output,
            labels=labels,
            image_ids=np.asarray([row["image_id"] for row in evaluation_ids], dtype=np.int64),
            annotation_ids=np.asarray(
                [row["annotation_id"] for row in evaluation_ids], dtype=np.int64
            ),
            prototype_scores=prototype_scores.astype(np.float32),
            hybrid_scores=hybrid_scores.astype(np.float32),
            ridge_scores=ridge_scores.astype(np.float32),
            lda_scores=lda_scores.astype(np.float32),
            classifier_head_scores=(
                np.empty((0, 20), dtype=np.float32)
                if classifier_head_scores is None
                else classifier_head_scores.astype(np.float32)
            ),
            selected_scores=scores.astype(np.float32),
            selected_decision_head=np.asarray(decision_head),
            local_patch_scores=(
                np.empty((0, 20), dtype=np.float32)
                if local_patch_scores is None
                else local_patch_scores.astype(np.float32)
            ),
            quality_scores=(
                np.empty((0,), dtype=np.float32)
                if classifier_quality_scores is None
                else classifier_quality_scores.astype(np.float32)
            ),
        )
    if args.features_output is not None:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.features_output,
            support_features=support_features.astype(np.float32),
            support_labels=support_labels_array,
            augmented_support_features=augmented_support_features.astype(np.float32),
            augmented_support_labels=np.asarray(augmented_support_labels, dtype=np.int64),
            evaluation_features=evaluation_features.astype(np.float32),
            evaluation_labels=labels,
            image_ids=np.asarray([row["image_id"] for row in evaluation_ids], dtype=np.int64),
            annotation_ids=np.asarray(
                [row["annotation_id"] for row in evaluation_ids], dtype=np.int64
            ),
        )
    if args.rotation_head_scores_output is not None:
        if classifier_head is None:
            parser.error("rotation head scores require --classifier-checkpoint")
        rotation_scores = _rotation_head_scores(
            model,
            classifier_head,
            evaluation_tensors,
            batch_size=args.batch_size,
        )
        args.rotation_head_scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.rotation_head_scores_output,
            labels=labels,
            image_ids=np.asarray([row["image_id"] for row in evaluation_ids], dtype=np.int64),
            annotation_ids=np.asarray(
                [row["annotation_id"] for row in evaluation_ids], dtype=np.int64
            ),
            rotation_head_scores=rotation_scores,
        )
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))


if __name__ == "__main__":
    main()
