from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from bixolon_scanner.pipeline.ports import Detection
from bixolon_scanner.runtime.onnx import prepare_rgb
from bixolon_scanner.runtime.preprocessing import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
)
from bixolon_scanner.training.class_conditional_quality import joint_partial_quality_label
from bixolon_scanner.training.margin_objective import normalized_margin_loss
from bixolon_scanner.training.models import (
    DINO_V3_HUB_REPOSITORY,
    build_dino_classifier,
    require_torch,
    set_frozen_backbone,
)
from bixolon_scanner.training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    prepare_direct_roi_source,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _AugmentedSupportDataset:
    def __init__(
        self,
        manifest: Path,
        root: Path,
        *,
        views: int,
        image_size: int,
        detail_image_size: int | None = None,
        expected_count: int | None = None,
        balanced_samples_per_class: int = 0,
        side_view_probability: float = 0.12,
        side_view_minimum_compression: float = 0.70,
        side_view_maximum_compression: float = 0.90,
        complete_quality_label: bool = False,
    ) -> None:
        self.records = [
            json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
        ]
        if expected_count is not None and len(self.records) != expected_count:
            raise ValueError(
                f"classifier training requires exactly {expected_count} support images"
            )
        self.views = views
        self.image_size = image_size
        self.detail_image_size = detail_image_size
        self.complete_quality_label = complete_quality_label
        self.recipe = DirectRoiRecipe(
            output_size=image_size,
            canvas_scale_min=0.78,
            canvas_scale_max=1.0,
            rotation_degrees=180.0,
            perspective_fraction=0.05,
            side_view_probability=side_view_probability,
            side_view_minimum_compression=side_view_minimum_compression,
            side_view_maximum_compression=side_view_maximum_compression,
            brightness_min=0.65,
            brightness_max=1.25,
            contrast_min=0.75,
            contrast_max=1.25,
            saturation_min=0.75,
            saturation_max=1.2,
            blur_probability=0.2,
            blur_radius_max=0.9,
            jpeg_quality_min=76,
            jpeg_quality_max=96,
            crop_mode="border_connected_composite",
            procedural_gradient=True,
            procedural_shadow=True,
        )
        self.images = []
        self.cutouts = []
        for record in self.records:
            with Image.open(root / record["image_path"]) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                self.images.append(image.copy())
                self.cutouts.append(prepare_direct_roi_source(image, self.recipe))
        self.presentations: list[tuple[int, int]] | None = None
        if balanced_samples_per_class:
            by_class: dict[int, list[int]] = {}
            for record_index, record in enumerate(self.records):
                by_class.setdefault(int(record["category_id"]), []).append(record_index)
            if sorted(by_class) != list(range(1, 21)):
                raise ValueError("balanced classifier support must contain all 20 classes")
            self.presentations = []
            for category_id in range(1, 21):
                indexes = by_class[category_id]
                for presentation_index in range(balanced_samples_per_class):
                    source_index = indexes[presentation_index % len(indexes)]
                    view_index = presentation_index // len(indexes)
                    self.presentations.append((source_index, view_index))

    def __len__(self) -> int:
        if self.presentations is not None:
            return len(self.presentations)
        return len(self.records) * self.views

    def __getitem__(self, index: int):
        torch = require_torch()
        if self.presentations is None:
            source_index = index % len(self.records)
            view_index = index // len(self.records)
        else:
            source_index, view_index = self.presentations[index]
        record = self.records[source_index]
        if view_index == 0:
            image = self.images[source_index]
        else:
            image = augment_direct_roi(
                self.images[source_index],
                source_sha256=str(record["image_sha256"]),
                category_id=int(record["category_id"]),
                seed=20260905 + source_index * 1000 + view_index,
                recipe=self.recipe,
                prepared_cutout=self.cutouts[source_index],
            ).image
        values = prepare_rgb(
            image,
            (self.image_size, self.image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        label = 0 if self.complete_quality_label else int(record["category_id"]) - 1
        if self.detail_image_size is None:
            return torch.from_numpy(values), label
        detail_values = prepare_rgb(
            image,
            (self.detail_image_size, self.detail_image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        return torch.from_numpy(values), torch.from_numpy(detail_values), label


class _SyntheticSceneRoiDataset:
    """Sample labeled, occlusion-bearing ROIs from source-only composite scenes."""

    def __init__(
        self,
        manifest: Path,
        root: Path,
        *,
        samples_per_epoch: int,
        image_size: int,
        detail_image_size: int | None,
        seed: int,
        neighbor_mask: bool = False,
        neighbor_distance_bias: float = -0.1,
        neighbor_context_scale: float = 0.0,
        minimum_visible_fraction: float = 0.0,
        maximum_visible_fraction: float = 1.0,
        partial_quality_threshold: float | None = None,
        identity_minimum_visible_fraction: float = 0.0,
        binary_quality_head: bool = False,
        minimum_margin: float = 0.0,
        maximum_margin: float = 0.08,
        forced_label: int | None = None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.detail_image_size = detail_image_size
        self.seed = seed
        self.neighbor_mask = neighbor_mask
        self.neighbor_distance_bias = neighbor_distance_bias
        self.neighbor_context_scale = neighbor_context_scale
        self.minimum_margin = minimum_margin
        self.maximum_margin = maximum_margin
        self.samples = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            image_path = str(record["image_path"])
            annotations = record.get("annotations", [])
            for annotation_index, annotation in enumerate(annotations):
                visible_fraction = float(annotation.get("visible_fraction", 1.0))
                if forced_label is not None:
                    label = forced_label
                elif partial_quality_threshold is None:
                    if not minimum_visible_fraction <= visible_fraction <= maximum_visible_fraction:
                        continue
                    label = int(annotation["category_id"]) - 1
                else:
                    if binary_quality_head:
                        if visible_fraction <= partial_quality_threshold:
                            label = 1
                        elif visible_fraction >= identity_minimum_visible_fraction:
                            label = 0
                        else:
                            label = None
                    else:
                        label = joint_partial_quality_label(
                            int(annotation["category_id"]),
                            visible_fraction,
                            partial_quality_threshold=partial_quality_threshold,
                            identity_minimum_visible_fraction=(identity_minimum_visible_fraction),
                        )
                    if label is None:
                        continue
                self.samples.append(
                    (
                        image_path,
                        annotations,
                        annotation_index,
                        label,
                    )
                )
        if not self.samples:
            raise ValueError("synthetic scene manifest has no labeled ROIs")
        self.samples_per_epoch = samples_per_epoch or len(self.samples)
        self.presentation_indexes = None
        if partial_quality_threshold is not None:
            by_label: dict[int, list[int]] = {}
            for sample_index, sample in enumerate(self.samples):
                by_label.setdefault(int(sample[3]), []).append(sample_index)
            label_count = 2 if binary_quality_head else 21
            if sorted(by_label) != list(range(label_count)):
                raise ValueError(f"quality training requires all {label_count} labels")
            self.presentation_indexes = []
            for presentation_index in range(self.samples_per_epoch):
                label = (presentation_index + seed) % label_count
                label_indexes = by_label[label]
                within_label = ((presentation_index // label_count) * 104729 + seed) % len(
                    label_indexes
                )
                self.presentation_indexes.append(label_indexes[within_label])

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        torch = require_torch()
        if self.presentation_indexes is None:
            sample_index = (index * 104729 + self.seed) % len(self.samples)
        else:
            sample_index = self.presentation_indexes[index]
        image_path, annotations, annotation_index, label = self.samples[sample_index]
        generator = random.Random(self.seed + index * 65537)
        with Image.open(self.root / image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            detections = []
            for annotation in annotations:
                x, y, width, height = annotation["bbox_xywh"]
                detections.append(Detection(x, y, x + width, y + height, score=1.0))
            margin = generator.uniform(self.minimum_margin, self.maximum_margin)
            crop = image.crop(
                classifier_crop_box(
                    detections[annotation_index],
                    image.width,
                    image.height,
                    margin_ratio=margin,
                    crop_mode="box_resize",
                )
            )
            masks = {}
            if self.neighbor_mask:
                for output_size in {self.image_size, self.detail_image_size} - {None}:
                    masks[output_size] = classifier_neighbor_ownership_mask(
                        detections,
                        annotation_index,
                        image_width=image.width,
                        image_height=image.height,
                        output_size=output_size,
                        margin_ratio=margin,
                        distance_bias=self.neighbor_distance_bias,
                        shared_scale=False,
                    )
        turns = generator.randrange(4)
        if turns:
            crop = crop.rotate(90 * turns, expand=True)
        crop = ImageEnhance.Brightness(crop).enhance(generator.uniform(0.85, 1.15))
        crop = ImageEnhance.Contrast(crop).enhance(generator.uniform(0.9, 1.1))
        crop = ImageEnhance.Color(crop).enhance(generator.uniform(0.9, 1.1))
        values = prepare_rgb(
            crop,
            (self.image_size, self.image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        if self.neighbor_mask:
            mask = (
                np.rot90(masks[self.image_size], turns).copy() if turns else masks[self.image_size]
            )
            masked_values = apply_classifier_background_masks(values[None], mask[None])[0]
            values = (
                self.neighbor_context_scale * values
                + (1.0 - self.neighbor_context_scale) * masked_values
            )
        if self.detail_image_size is None:
            return torch.from_numpy(values), label
        detail_values = prepare_rgb(
            crop,
            (self.detail_image_size, self.detail_image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        if self.neighbor_mask:
            detail_mask = masks[self.detail_image_size]
            if turns:
                detail_mask = np.rot90(detail_mask, turns).copy()
            masked_detail_values = apply_classifier_background_masks(
                detail_values[None], detail_mask[None]
            )[0]
            detail_values = (
                self.neighbor_context_scale * detail_values
                + (1.0 - self.neighbor_context_scale) * masked_detail_values
            )
        return torch.from_numpy(values), torch.from_numpy(detail_values), label


class _RealSceneTruncatedRoiDataset:
    """Create tight partial-object views from source-camera multi-object ROIs."""

    def __init__(
        self,
        manifest: Path,
        root: Path,
        *,
        samples_per_epoch: int,
        image_size: int,
        detail_image_size: int | None,
        seed: int,
        minimum_retained_fraction: float,
        maximum_retained_fraction: float,
        minimum_margin: float,
        maximum_margin: float,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.detail_image_size = detail_image_size
        self.seed = seed
        self.minimum_retained_fraction = minimum_retained_fraction
        self.maximum_retained_fraction = maximum_retained_fraction
        self.minimum_margin = minimum_margin
        self.maximum_margin = maximum_margin
        self.samples: list[tuple[str, list[dict], int]] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            annotations = record.get("annotations", [])
            for annotation_index in range(len(annotations)):
                self.samples.append((str(record["image_path"]), annotations, annotation_index))
        if not self.samples:
            raise ValueError("real partial scene manifest has no ROIs")
        self.samples_per_epoch = samples_per_epoch or len(self.samples)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        torch = require_torch()
        sample_index = (index * 104729 + self.seed) % len(self.samples)
        image_path, annotations, annotation_index = self.samples[sample_index]
        generator = random.Random(self.seed + index * 65537)
        annotation = annotations[annotation_index]
        x, y, width, height = annotation["bbox_xywh"]
        detection = Detection(x, y, x + width, y + height, score=1.0)
        margin = generator.uniform(self.minimum_margin, self.maximum_margin)
        with Image.open(self.root / image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            crop = image.crop(
                classifier_crop_box(
                    detection,
                    image.width,
                    image.height,
                    margin_ratio=margin,
                    crop_mode="box_resize",
                )
            )
        retained_fraction = generator.uniform(
            self.minimum_retained_fraction, self.maximum_retained_fraction
        )
        if generator.random() < 0.5:
            retained = max(2, round(crop.width * retained_fraction))
            maximum_start = max(0, crop.width - retained)
            if generator.random() < 0.8:
                start = 0 if generator.random() < 0.5 else maximum_start
            else:
                start = generator.randint(0, maximum_start)
            crop = crop.crop((start, 0, start + retained, crop.height))
        else:
            retained = max(2, round(crop.height * retained_fraction))
            maximum_start = max(0, crop.height - retained)
            if generator.random() < 0.8:
                start = 0 if generator.random() < 0.5 else maximum_start
            else:
                start = generator.randint(0, maximum_start)
            crop = crop.crop((0, start, crop.width, start + retained))
        turns = generator.randrange(4)
        if turns:
            crop = crop.rotate(90 * turns, expand=True)
        crop = ImageEnhance.Brightness(crop).enhance(generator.uniform(0.85, 1.15))
        crop = ImageEnhance.Contrast(crop).enhance(generator.uniform(0.9, 1.1))
        crop = ImageEnhance.Color(crop).enhance(generator.uniform(0.9, 1.1))
        values = prepare_rgb(
            crop,
            (self.image_size, self.image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        if self.detail_image_size is None:
            return torch.from_numpy(values), 1
        detail_values = prepare_rgb(
            crop,
            (self.detail_image_size, self.detail_image_size),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
        return torch.from_numpy(values), torch.from_numpy(detail_values), 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-epoch source-only DINOv3 classifier fine-tuning"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-support-count", type=int, default=200)
    parser.add_argument("--support-samples-per-class", type=int, default=0)
    parser.add_argument("--additional-manifest", type=Path)
    parser.add_argument("--additional-root", type=Path)
    parser.add_argument("--additional-views", type=int, default=4)
    parser.add_argument("--additional-samples-per-class", type=int, default=0)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--scene-samples-per-epoch", type=int, default=0)
    parser.add_argument("--scene-neighbor-mask", action="store_true")
    parser.add_argument("--scene-neighbor-distance-bias", type=float, default=-0.1)
    parser.add_argument("--scene-neighbor-context-scale", type=float, default=0.0)
    parser.add_argument("--scene-minimum-margin", type=float, default=0.0)
    parser.add_argument("--scene-maximum-margin", type=float, default=0.08)
    parser.add_argument("--scene-minimum-visible-fraction", type=float, default=0.0)
    parser.add_argument("--scene-maximum-visible-fraction", type=float, default=1.0)
    parser.add_argument("--scene-partial-quality-threshold", type=float)
    parser.add_argument("--scene-identity-minimum-visible-fraction", type=float, default=0.0)
    parser.add_argument("--identity-scene-manifest", type=Path)
    parser.add_argument("--identity-scene-root", type=Path)
    parser.add_argument("--identity-scene-samples-per-epoch", type=int, default=0)
    parser.add_argument("--complete-scene-manifest", type=Path)
    parser.add_argument("--complete-scene-root", type=Path)
    parser.add_argument("--complete-scene-samples-per-epoch", type=int, default=0)
    parser.add_argument("--real-partial-scene-manifest", type=Path)
    parser.add_argument("--real-partial-scene-root", type=Path)
    parser.add_argument("--real-partial-scene-samples-per-epoch", type=int, default=0)
    parser.add_argument("--real-partial-minimum-retained-fraction", type=float, default=0.18)
    parser.add_argument("--real-partial-maximum-retained-fraction", type=float, default=0.55)
    parser.add_argument(
        "--binary-quality-head",
        action="store_true",
        help="Train a balanced complete-versus-partial quality head without identity logits",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument(
        "--variant",
        choices=("dinov3_convnext_tiny", "dinov3_vitb16"),
        default="dinov3_convnext_tiny",
    )
    parser.add_argument("--teacher-vit-weights", type=Path)
    parser.add_argument("--teacher-vit-weights-sha256")
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--teacher-logit-weight", type=float, default=0.0)
    parser.add_argument("--teacher-temperature", type=float, default=0.05)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--initial-identity-checkpoint",
        type=Path,
        help=(
            "Initialize a 21-output identity-plus-quality model from a 20-output "
            "identity checkpoint while retaining the newly initialized quality row"
        ),
    )
    parser.add_argument(
        "--freeze-initial-identity-rows",
        action="store_true",
        help="Restore the 20 imported identity rows after every optimizer step",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--detail-image-size", type=int)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--normalized-margin-weight", type=float, default=0.0)
    parser.add_argument("--normalized-margin-target", type=float, default=0.9)
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=0.00001)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--side-view-probability", type=float, default=0.12)
    parser.add_argument("--side-view-minimum-compression", type=float, default=0.70)
    parser.add_argument("--side-view-maximum-compression", type=float, default=0.90)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    if args.normalized_margin_weight < 0 or not 0 < args.normalized_margin_target <= 1:
        parser.error("margin loss requires nonnegative weight and target inside (0, 1]")

    if args.additional_samples_per_class < 0:
        parser.error("--additional-samples-per-class cannot be negative")
    if args.expected_support_count < 1:
        parser.error("--expected-support-count must be positive")
    if args.support_samples_per_class < 0:
        parser.error("--support-samples-per-class cannot be negative")
    if not (
        0.0 <= args.scene_minimum_visible_fraction <= args.scene_maximum_visible_fraction <= 1.0
    ):
        parser.error("scene visible-fraction range must satisfy 0 <= minimum <= maximum <= 1")
    if not 0.0 <= args.scene_minimum_margin <= args.scene_maximum_margin <= 1.0:
        parser.error("scene margin range must satisfy 0 <= minimum <= maximum <= 1")
    if not 0.0 <= args.scene_neighbor_context_scale <= 1.0:
        parser.error("scene neighbor context scale must be in [0, 1]")
    if args.scene_partial_quality_threshold is not None:
        if not (
            0.0
            <= args.scene_partial_quality_threshold
            < args.scene_identity_minimum_visible_fraction
            <= 1.0
        ):
            parser.error("scene quality threshold must be below the identity visibility threshold")
        if args.initial_checkpoint is not None:
            parser.error("quality training cannot load an identity initial checkpoint directly")
    elif args.initial_identity_checkpoint is not None:
        parser.error("--initial-identity-checkpoint requires 21-class quality training")
    if args.initial_checkpoint is not None and args.initial_identity_checkpoint is not None:
        parser.error("choose only one initial checkpoint mode")
    if args.freeze_initial_identity_rows and args.initial_identity_checkpoint is None:
        parser.error("--freeze-initial-identity-rows requires --initial-identity-checkpoint")
    if args.binary_quality_head and args.scene_partial_quality_threshold is None:
        parser.error("--binary-quality-head requires --scene-partial-quality-threshold")
    if args.binary_quality_head and args.freeze_initial_identity_rows:
        parser.error("binary quality training has no identity rows to freeze")
    if args.binary_quality_head and args.identity_scene_manifest is not None:
        parser.error("identity scene examples cannot train a binary quality head")
    if not (
        0.0
        < args.real_partial_minimum_retained_fraction
        <= args.real_partial_maximum_retained_fraction
        < 1.0
    ):
        parser.error("real partial retained-fraction range must be inside (0, 1)")

    actual_sha256 = _sha256(args.weights)
    if actual_sha256 != args.weights_sha256.lower():
        raise ValueError("official DINOv3 weight checksum mismatch")
    if (args.teacher_vit_weights is None) != (args.teacher_vit_weights_sha256 is None):
        parser.error(
            "--teacher-vit-weights and --teacher-vit-weights-sha256 must be provided together"
        )
    if args.teacher_vit_weights is not None:
        teacher_sha256 = _sha256(args.teacher_vit_weights)
        if teacher_sha256 != args.teacher_vit_weights_sha256.lower():
            raise ValueError("official DINOv3 teacher weight checksum mismatch")
    else:
        teacher_sha256 = None
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch = require_torch()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    primary_dataset = _AugmentedSupportDataset(
        args.manifest,
        args.root,
        views=args.views,
        image_size=args.image_size,
        detail_image_size=args.detail_image_size,
        expected_count=args.expected_support_count,
        balanced_samples_per_class=args.support_samples_per_class,
        side_view_probability=args.side_view_probability,
        side_view_minimum_compression=args.side_view_minimum_compression,
        side_view_maximum_compression=args.side_view_maximum_compression,
        complete_quality_label=args.binary_quality_head,
    )
    additional_args = (args.additional_manifest, args.additional_root)
    if any(value is not None for value in additional_args) and any(
        value is None for value in additional_args
    ):
        raise ValueError("additional manifest and root must be provided together")
    datasets = [primary_dataset]
    additional_dataset = None
    if args.additional_manifest is not None:
        additional_dataset = _AugmentedSupportDataset(
            args.additional_manifest,
            args.additional_root,
            views=args.additional_views,
            balanced_samples_per_class=args.additional_samples_per_class,
            image_size=args.image_size,
            detail_image_size=args.detail_image_size,
            side_view_probability=args.side_view_probability,
            side_view_minimum_compression=args.side_view_minimum_compression,
            side_view_maximum_compression=args.side_view_maximum_compression,
            complete_quality_label=args.binary_quality_head,
        )
        datasets.append(additional_dataset)
    scene_args = (args.scene_manifest, args.scene_root)
    if any(value is not None for value in scene_args) and any(
        value is None for value in scene_args
    ):
        raise ValueError("scene manifest and root must be provided together")
    scene_dataset = None
    if args.scene_manifest is not None:
        scene_dataset = _SyntheticSceneRoiDataset(
            args.scene_manifest,
            args.scene_root,
            samples_per_epoch=args.scene_samples_per_epoch,
            image_size=args.image_size,
            detail_image_size=args.detail_image_size,
            seed=args.seed + 100_000,
            neighbor_mask=args.scene_neighbor_mask,
            neighbor_distance_bias=args.scene_neighbor_distance_bias,
            neighbor_context_scale=args.scene_neighbor_context_scale,
            minimum_visible_fraction=args.scene_minimum_visible_fraction,
            maximum_visible_fraction=args.scene_maximum_visible_fraction,
            partial_quality_threshold=args.scene_partial_quality_threshold,
            identity_minimum_visible_fraction=(args.scene_identity_minimum_visible_fraction),
            binary_quality_head=args.binary_quality_head,
            minimum_margin=args.scene_minimum_margin,
            maximum_margin=args.scene_maximum_margin,
        )
        datasets.append(scene_dataset)
    identity_scene_args = (args.identity_scene_manifest, args.identity_scene_root)
    if any(value is not None for value in identity_scene_args) and any(
        value is None for value in identity_scene_args
    ):
        raise ValueError("identity scene manifest and root must be provided together")
    identity_scene_dataset = None
    if args.identity_scene_manifest is not None:
        identity_scene_dataset = _SyntheticSceneRoiDataset(
            args.identity_scene_manifest,
            args.identity_scene_root,
            samples_per_epoch=args.identity_scene_samples_per_epoch,
            image_size=args.image_size,
            detail_image_size=args.detail_image_size,
            seed=args.seed + 150_000,
            neighbor_mask=args.scene_neighbor_mask,
            neighbor_distance_bias=args.scene_neighbor_distance_bias,
            neighbor_context_scale=args.scene_neighbor_context_scale,
            minimum_margin=args.scene_minimum_margin,
            maximum_margin=args.scene_maximum_margin,
        )
        datasets.append(identity_scene_dataset)
    complete_scene_args = (args.complete_scene_manifest, args.complete_scene_root)
    if any(value is not None for value in complete_scene_args) and any(
        value is None for value in complete_scene_args
    ):
        raise ValueError("complete scene manifest and root must be provided together")
    complete_scene_dataset = None
    if args.complete_scene_manifest is not None:
        if not args.binary_quality_head:
            parser.error("complete scene quality examples require --binary-quality-head")
        complete_scene_dataset = _SyntheticSceneRoiDataset(
            args.complete_scene_manifest,
            args.complete_scene_root,
            samples_per_epoch=args.complete_scene_samples_per_epoch,
            image_size=args.image_size,
            detail_image_size=args.detail_image_size,
            seed=args.seed + 200_000,
            neighbor_mask=args.scene_neighbor_mask,
            neighbor_distance_bias=args.scene_neighbor_distance_bias,
            neighbor_context_scale=args.scene_neighbor_context_scale,
            minimum_margin=args.scene_minimum_margin,
            maximum_margin=args.scene_maximum_margin,
            forced_label=0,
        )
        datasets.append(complete_scene_dataset)
    real_partial_args = (
        args.real_partial_scene_manifest,
        args.real_partial_scene_root,
    )
    if any(value is not None for value in real_partial_args) and any(
        value is None for value in real_partial_args
    ):
        raise ValueError("real partial scene manifest and root must be provided together")
    real_partial_scene_dataset = None
    if args.real_partial_scene_manifest is not None:
        if not args.binary_quality_head:
            parser.error("real partial scene examples require --binary-quality-head")
        real_partial_scene_dataset = _RealSceneTruncatedRoiDataset(
            args.real_partial_scene_manifest,
            args.real_partial_scene_root,
            samples_per_epoch=args.real_partial_scene_samples_per_epoch,
            image_size=args.image_size,
            detail_image_size=args.detail_image_size,
            seed=args.seed + 300_000,
            minimum_retained_fraction=args.real_partial_minimum_retained_fraction,
            maximum_retained_fraction=args.real_partial_maximum_retained_fraction,
            minimum_margin=args.scene_minimum_margin,
            maximum_margin=args.scene_maximum_margin,
        )
        datasets.append(real_partial_scene_dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    class_count = (
        2
        if args.binary_quality_head
        else (21 if args.scene_partial_quality_threshold is not None else 20)
    )
    model = build_dino_classifier(
        args.variant,
        class_count,
        weights_path=args.weights,
        feature_l2_normalize=True,
        classifier_head_kind="cosine",
        cosine_scale=16.0,
    ).cuda()
    initial_checkpoint_sha256 = None
    initial_identity_checkpoint_sha256 = None
    frozen_identity_rows = None
    if args.initial_checkpoint is not None:
        initial_checkpoint_sha256 = _sha256(args.initial_checkpoint)
        model.load_state_dict(
            torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
    elif args.initial_identity_checkpoint is not None:
        initial_identity_checkpoint_sha256 = _sha256(args.initial_identity_checkpoint)
        identity_state = torch.load(
            args.initial_identity_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        target_state = model.state_dict()
        identity_weight = identity_state.get("classifier.weight")
        target_weight = target_state.get("classifier.weight")
        if identity_weight is None or target_weight is None:
            raise ValueError("identity checkpoint classifier shape is incompatible")
        expected_target_rows = 2 if args.binary_quality_head else 21
        if tuple(identity_weight.shape) != (20, target_weight.shape[1]) or tuple(
            target_weight.shape
        ) != (expected_target_rows, identity_weight.shape[1]):
            raise ValueError("identity checkpoint classifier shape is incompatible")
        for key, value in identity_state.items():
            if key == "classifier.weight":
                continue
            if key not in target_state or target_state[key].shape != value.shape:
                raise ValueError(f"identity checkpoint tensor is incompatible: {key}")
            target_state[key] = value
        if not args.binary_quality_head:
            target_state["classifier.weight"][:20] = identity_weight
        model.load_state_dict(target_state, strict=True)
        if args.freeze_initial_identity_rows:
            frozen_identity_rows = identity_weight.cuda().clone()
    teacher = None
    teacher_prototypes = None
    if args.teacher_vit_weights is not None:
        teacher = torch.hub.load(
            DINO_V3_HUB_REPOSITORY,
            "dinov3_vitb16",
            source="github",
            trust_repo=True,
            verbose=False,
            pretrained=False,
        )
        teacher.load_state_dict(
            torch.load(args.teacher_vit_weights, map_location="cpu", weights_only=True),
            strict=True,
        )
        teacher.requires_grad_(False).eval().cuda()
        if args.teacher_logit_weight:
            clean_tensors = torch.stack(
                [primary_dataset[index][0] for index in range(len(primary_dataset.records))]
            )
            clean_features = []
            with torch.inference_mode():
                for start in range(0, len(clean_tensors), 64):
                    clean_batch = clean_tensors[start : start + 64].cuda(non_blocking=True)
                    clean_features.append(
                        teacher.forward_features(clean_batch, masks=None)["x_norm_clstoken"]
                    )
            clean_features = torch.nn.functional.normalize(torch.cat(clean_features), dim=-1)
            clean_labels = torch.tensor(
                [int(record["category_id"]) - 1 for record in primary_dataset.records],
                device="cuda",
            )
            teacher_prototypes = torch.stack(
                [
                    clean_features[clean_labels == class_index].mean(dim=0)
                    for class_index in range(20)
                ]
            )
            teacher_prototypes = torch.nn.functional.normalize(teacher_prototypes, dim=-1)
    if args.freeze_backbone:
        set_frozen_backbone(model)
    else:
        if args.variant != "dinov3_convnext_tiny":
            parser.error("ViT-B/16 training requires --freeze-backbone")
        set_frozen_backbone(model, unfreeze_last_stages=1)
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.backbone_learning_rate * 0.02,
    )
    scaler = torch.amp.GradScaler("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        else:
            model.backbone.stages[0].eval()
            model.backbone.stages[1].eval()
            model.backbone.stages[2].eval()
        loss_sum = 0.0
        correct = 0
        count = 0
        epoch_started = time.perf_counter()
        for batch in loader:
            if args.detail_image_size is None:
                images, labels = batch
                detail_images = None
            else:
                images, detail_images, labels = batch
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            if detail_images is not None:
                detail_images = detail_images.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                features = model.extract_features(images)
                logits = model.classifier(features)
                loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
                if args.normalized_margin_weight:
                    loss = loss + args.normalized_margin_weight * normalized_margin_loss(
                        logits, labels, target_margin=args.normalized_margin_target
                    )
                if teacher is not None:
                    with torch.no_grad():
                        teacher_features = teacher.forward_features(images, masks=None)[
                            "x_norm_clstoken"
                        ]
                        teacher_features = torch.nn.functional.normalize(teacher_features, dim=-1)
                    loss = loss + args.distillation_weight * (
                        1.0
                        - torch.nn.functional.cosine_similarity(
                            features, teacher_features, dim=-1
                        ).mean()
                    )
                    if teacher_prototypes is not None:
                        teacher_probabilities = torch.softmax(
                            teacher_features
                            @ teacher_prototypes.transpose(0, 1)
                            / args.teacher_temperature,
                            dim=-1,
                        )
                        teacher_logit_loss = (
                            -(teacher_probabilities * torch.log_softmax(logits, dim=-1))
                            .sum(dim=-1)
                            .mean()
                        )
                        loss = loss + args.teacher_logit_weight * teacher_logit_loss
                if detail_images is not None:
                    detail_features = model.extract_features(detail_images)
                    detail_logits = model.classifier(detail_features)
                    detail_loss = torch.nn.functional.cross_entropy(
                        detail_logits, labels, label_smoothing=0.05
                    )
                    if args.normalized_margin_weight:
                        detail_loss = (
                            detail_loss
                            + args.normalized_margin_weight
                            * normalized_margin_loss(
                                detail_logits, labels, target_margin=args.normalized_margin_target
                            )
                        )
                    consistency_loss = (
                        1.0
                        - torch.nn.functional.cosine_similarity(
                            features, detail_features, dim=-1
                        ).mean()
                    )
                    loss = 0.5 * (loss + detail_loss) + args.consistency_weight * consistency_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite classifier loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            if frozen_identity_rows is not None:
                with torch.no_grad():
                    model.classifier.weight[:20].copy_(frozen_identity_rows)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            count += len(labels)
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": loss_sum / count,
            "training_accuracy": correct / count,
            "backbone_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            "duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    checkpoint = args.output_dir / "last.pt"
    torch.save(model.state_dict(), checkpoint)
    report = {
        "schema_version": "1.0",
        "experiment": "store2_source_only_dinov3_classifier_finetune",
        "selection_policy": "fixed_last_epoch_without_evaluation",
        "development_evaluation_accessed": False,
        "initial_classifier_checkpoint": (
            None if args.initial_checkpoint is None else str(args.initial_checkpoint)
        ),
        "initial_classifier_checkpoint_sha256": initial_checkpoint_sha256,
        "initial_identity_checkpoint": (
            None
            if args.initial_identity_checkpoint is None
            else str(args.initial_identity_checkpoint)
        ),
        "initial_identity_checkpoint_sha256": initial_identity_checkpoint_sha256,
        "official_backbone_weights_sha256": actual_sha256,
        "official_teacher_weights_sha256": teacher_sha256,
        "fresh_components": [
            "balanced binary quality head" if args.binary_quality_head else "cosine classifier head"
        ],
        "settings": vars(args) | {"output_dir": str(args.output_dir), "weights": str(args.weights)},
        "augmentation_recipe": primary_dataset.recipe.__dict__,
        "sample_counts": {
            "single_object_presentations": len(primary_dataset),
            "multi_object_crop_presentations": (
                0 if additional_dataset is None else len(additional_dataset)
            ),
            "synthetic_scene_roi_presentations": (
                0 if scene_dataset is None else len(scene_dataset)
            ),
            "synthetic_scene_roi_population": (
                0 if scene_dataset is None else len(scene_dataset.samples)
            ),
            "real_identity_scene_roi_presentations": (
                0 if identity_scene_dataset is None else len(identity_scene_dataset)
            ),
            "real_complete_scene_roi_presentations": (
                0 if complete_scene_dataset is None else len(complete_scene_dataset)
            ),
            "real_truncated_scene_roi_presentations": (
                0 if real_partial_scene_dataset is None else len(real_partial_scene_dataset)
            ),
        },
        "history": history,
        "duration_seconds": time.perf_counter() - started,
        "checkpoint": checkpoint.name,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
