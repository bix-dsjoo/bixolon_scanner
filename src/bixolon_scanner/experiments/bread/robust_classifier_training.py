from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...pipeline.ports import Detection
from ...runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from ...training.data import read_manifest
from ...training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ...training.models import require_torch
from ...training.synthetic_roi import (
    ClutterRoiRecipe,
    DirectRoiRecipe,
    augment_clutter_roi,
    clutter_roi_recipe_sha256,
    prepare_direct_roi_source,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_single_objects_records(manifest: Path) -> list[dict[str, Any]]:
    records = [
        row
        for row in read_manifest(manifest)
        if row["record_type"] == "classification" and row["split"] == "development"
    ]
    if not records:
        raise ValueError("classifier manifest contains no development records")
    sources = {Path(str(row["image_path"])).parts[0] for row in records}
    if sources != {"single_objects"}:
        raise ValueError(f"classifier source must be single_objects only, got {sorted(sources)}")
    if len({str(row["image_sha256"]) for row in records}) != len(records):
        raise ValueError("classifier manifest contains duplicate source images")
    if set(int(row["fold"]) for row in records) != {0, 1, 2}:
        raise ValueError("classifier manifest must contain exactly folds 0, 1, and 2")
    return records


def hard_clutter_recipe() -> ClutterRoiRecipe:
    return ClutterRoiRecipe(
        output_size=224,
        target_scale_min=0.50,
        target_scale_max=0.88,
        distractor_count_min=1,
        distractor_count_max=4,
        distractor_scale_min=0.30,
        distractor_scale_max=0.85,
        maximum_rotation_degrees=45.0,
        foreground_distractor_probability=0.65,
        maximum_target_occlusion=0.28,
        placement_attempts=64,
        background_min=185,
        background_max=235,
        jpeg_quality_min=78,
        jpeg_quality_max=96,
    )


def moderate_clutter_recipe() -> ClutterRoiRecipe:
    """Match ordinary detector-boundary clutter without dominating the target."""
    return ClutterRoiRecipe(
        output_size=224,
        target_scale_min=0.62,
        target_scale_max=0.92,
        distractor_count_min=1,
        distractor_count_max=3,
        distractor_scale_min=0.22,
        distractor_scale_max=0.65,
        maximum_rotation_degrees=35.0,
        foreground_distractor_probability=0.45,
        maximum_target_occlusion=0.18,
        placement_attempts=64,
        background_min=200,
        background_max=245,
        jpeg_quality_min=82,
        jpeg_quality_max=96,
    )


def mild_clutter_recipe() -> ClutterRoiRecipe:
    """Keep the target dominant while exposing ordinary detector-boundary neighbors."""
    return ClutterRoiRecipe(
        output_size=224,
        target_scale_min=0.82,
        target_scale_max=0.98,
        distractor_count_min=1,
        distractor_count_max=2,
        distractor_scale_min=0.25,
        distractor_scale_max=0.55,
        maximum_rotation_degrees=40.0,
        foreground_distractor_probability=0.10,
        maximum_target_occlusion=0.05,
        placement_attempts=40,
        background_min=230,
        background_max=255,
        jpeg_quality_min=82,
        jpeg_quality_max=96,
    )


def prepare_clutter_tensor(
    sample,
    *,
    apply_neighbor_mask: bool,
    margin_ratio: float = 0.05,
    distance_bias: float = 0.0,
) -> np.ndarray:
    """Apply the production crop/mask contract to a synthetic single-object scene."""
    if not apply_neighbor_mask:
        return prepare_rgb(
            sample.image,
            (224, 224),
            MEAN,
            STD,
            reducing_gap=1.0,
        )
    detections = [Detection(*sample.bbox_xyxy, score=1.0)]
    for distractor in sample.provenance["distractors"]:
        detections.append(Detection(*distractor["bbox_xyxy"], score=1.0))
    crop_box = classifier_crop_box(
        detections[0],
        sample.image.width,
        sample.image.height,
        margin_ratio=margin_ratio,
        crop_mode="box_resize",
    )
    tensor = prepare_rgb(
        sample.image.crop(crop_box),
        (224, 224),
        MEAN,
        STD,
        reducing_gap=1.0,
    )
    mask = classifier_neighbor_ownership_mask(
        detections,
        0,
        image_width=sample.image.width,
        image_height=sample.image.height,
        output_size=224,
        margin_ratio=margin_ratio,
        distance_bias=distance_bias,
        shared_scale=False,
    )
    return apply_classifier_background_masks(tensor[None], mask[None])[0]


def _prepare_tensor_cache(
    *,
    dataset_root: Path,
    manifest: Path,
    cache_path: Path,
    views_per_source: int,
    seed: int,
    recipe: ClutterRoiRecipe,
    apply_neighbor_mask: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    records = load_single_objects_records(manifest)
    expected_count = len(records) * views_per_source
    labels_path = cache_path.with_suffix(".labels.npy")
    folds_path = cache_path.with_suffix(".folds.npy")
    support_path = cache_path.with_suffix(".support.npy")
    metadata_path = cache_path.with_suffix(".json")
    expected_metadata = {
        "manifest_sha256": _sha256(manifest),
        "recipe": asdict(recipe),
        "recipe_sha256": clutter_roi_recipe_sha256(recipe),
        "seed": seed,
        "source_count": len(records),
        "views_per_source": views_per_source,
        "apply_neighbor_mask": apply_neighbor_mask,
        "neighbor_mask_margin_ratio": 0.05,
        "neighbor_mask_distance_bias": 0.0,
    }
    cache_files = (cache_path, labels_path, folds_path, support_path, metadata_path)
    if all(path.is_file() for path in cache_files):
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual_metadata != expected_metadata:
            raise ValueError("hard-clutter cache metadata does not match the requested recipe")
        tensors = np.load(cache_path, mmap_mode="r")
        labels = np.load(labels_path)
        folds = np.load(folds_path)
        support = np.load(support_path, mmap_mode="r")
        expected_shape = (expected_count, 3, 224, 224)
        if tensors.shape != expected_shape or support.shape != (len(records), 3, 224, 224):
            raise ValueError("hard-clutter tensor cache has an unexpected shape")
        return tensors, labels, folds, support, records

    source_recipe = DirectRoiRecipe(
        crop_mode="border_connected_composite",
        border_color_distance=42,
        mask_feather_radius=0.8,
    )
    images: list[Image.Image] = []
    cutouts: list[Image.Image] = []
    for record in records:
        with Image.open(dataset_root / str(record["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
        images.append(image)
        cutouts.append(prepare_direct_roi_source(image, source_recipe))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float16,
        shape=(expected_count, 3, 224, 224),
    )
    labels = np.empty(expected_count, dtype=np.int64)
    folds = np.empty(expected_count, dtype=np.int64)
    for source_index, record in enumerate(records):
        distractors = [
            (cutouts[index], str(other["image_sha256"]), int(other["category_id"]))
            for index, other in enumerate(records)
            if int(other["category_id"]) != int(record["category_id"])
        ]
        for view_index in range(views_per_source):
            tensor_index = source_index * views_per_source + view_index
            sample = augment_clutter_roi(
                cutouts[source_index],
                target_sha256=str(record["image_sha256"]),
                target_category_id=int(record["category_id"]),
                distractors=distractors,
                seed=seed + source_index * 1_000_003 + view_index,
                recipe=recipe,
            )
            tensors[tensor_index] = prepare_clutter_tensor(
                sample,
                apply_neighbor_mask=apply_neighbor_mask,
            ).astype(np.float16)
            labels[tensor_index] = int(record["category_id"]) - 1
            folds[tensor_index] = int(record["fold"])
        if (source_index + 1) % 20 == 0:
            print(json.dumps({"prepared_sources": source_index + 1}), flush=True)
    tensors.flush()
    support = np.asarray(
        [prepare_rgb(image, (224, 224), MEAN, STD, reducing_gap=1.0) for image in images],
        dtype=np.float16,
    )
    np.save(labels_path, labels)
    np.save(folds_path, folds)
    np.save(support_path, support)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8")
    for image in images:
        image.close()
    for cutout in cutouts:
        cutout.close()
    return (
        np.load(cache_path, mmap_mode="r"),
        labels,
        folds,
        np.load(support_path, mmap_mode="r"),
        records,
    )


def top3_margin_loss(torch, logits, labels, *, margin: float):
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    negatives = logits.clone()
    negatives.scatter_(1, labels[:, None], float("-inf"))
    third_negative = torch.topk(negatives, k=3, dim=1).values[:, 2]
    return torch.nn.functional.softplus(third_negative - true_logits + margin).mean()


def _build_model(torch, checkpoint: dict[str, Any], device):
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    return model.to(device)


def _configure_trainable(model, scope: str) -> list[Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope == "last_stage":
        for parameter in model.backbone.stages[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.norm.parameters():
            parameter.requires_grad = True
    elif scope != "head":
        raise ValueError(f"unsupported trainable scope: {scope}")
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _predict(model, tensors, *, torch, device, batch_size: int) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            parts.append(model(batch).float().cpu().numpy())
    return np.concatenate(parts)


def _metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    ranking = np.argsort(-logits, axis=1, kind="stable")
    top1 = ranking[:, 0]
    return {
        "sample_count": len(targets),
        "top1_error_count": int(np.count_nonzero(top1 != targets)),
        "top1_accuracy": float(np.mean(top1 == targets)),
        "top3_miss_count": int(
            np.count_nonzero(~np.any(ranking[:, :3] == targets[:, None], axis=1))
        ),
        "top3_accuracy": float(np.mean(np.any(ranking[:, :3] == targets[:, None], axis=1))),
    }


def _train_epochs(
    model,
    *,
    tensors,
    labels: np.ndarray,
    selected: np.ndarray,
    support,
    support_labels: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    torch,
    device,
) -> list[dict[str, float]]:
    trainable = _configure_trainable(model, args.trainable_scope)
    reference = [parameter.detach().clone() for parameter in trainable]
    backbone_parameters = (
        {
            id(parameter)
            for module in (model.backbone.stages[-1], model.backbone.norm)
            for parameter in module.parameters()
        }
        if args.trainable_scope == "last_stage"
        else set()
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in trainable if id(p) in backbone_parameters],
                "lr": args.backbone_learning_rate,
            },
            {
                "params": [p for p in trainable if id(p) not in backbone_parameters],
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = selected[torch.randperm(len(selected), generator=generator).numpy()]
        totals = np.zeros(4, dtype=np.float64)
        for step, start in enumerate(range(0, len(order), args.batch_size)):
            indices = order[start : start + args.batch_size]
            pixels = torch.from_numpy(np.array(tensors[indices], dtype=np.float32, copy=True)).to(
                device
            )
            targets = torch.from_numpy(labels[indices]).to(device)
            support_start = (step * args.support_batch) % len(support)
            support_indices = np.arange(support_start, support_start + args.support_batch) % len(
                support
            )
            support_pixels = torch.from_numpy(
                np.array(support[support_indices], dtype=np.float32, copy=True)
            ).to(device)
            clean_targets = torch.from_numpy(support_labels[support_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixels)
            clutter_loss = torch.nn.functional.cross_entropy(logits, targets)
            rank_loss = top3_margin_loss(torch, logits, targets, margin=args.top3_margin)
            clean_loss = torch.nn.functional.cross_entropy(model(support_pixels), clean_targets)
            l2_loss = sum(
                (parameter - initial).square().mean()
                for parameter, initial in zip(trainable, reference)
            )
            loss = (
                clutter_loss
                + args.top3_weight * rank_loss
                + args.support_weight * clean_loss
                + args.l2_weight * l2_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            totals += np.asarray(
                [loss.item(), clutter_loss.item(), rank_loss.item(), clean_loss.item()]
            )
        row = {
            "epoch": float(epoch),
            "loss": float(totals[0] / max(1, math.ceil(len(order) / args.batch_size))),
            "clutter_loss": float(totals[1] / max(1, math.ceil(len(order) / args.batch_size))),
            "top3_margin_loss": float(totals[2] / max(1, math.ceil(len(order) / args.batch_size))),
            "support_loss": float(totals[3] / max(1, math.ceil(len(order) / args.batch_size))),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.validation_fold not in {0, 1, 2}:
        raise ValueError("validation_fold must be 0, 1, or 2")
    torch = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    recipes = {
        "mild": mild_clutter_recipe,
        "moderate": moderate_clutter_recipe,
        "hard": hard_clutter_recipe,
    }
    recipe = recipes[args.recipe_profile]()
    tensors, labels, folds, support, records = _prepare_tensor_cache(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        cache_path=args.tensor_cache,
        views_per_source=args.views_per_source,
        seed=args.seed,
        recipe=recipe,
        apply_neighbor_mask=args.neighbor_mask,
    )
    source_labels = np.asarray([int(row["category_id"]) - 1 for row in records], dtype=np.int64)
    source_folds = np.asarray([int(row["fold"]) for row in records], dtype=np.int64)
    train_groups = {
        str(row["perceptual_group_id"])
        for row in records
        if int(row["fold"]) != args.validation_fold
    }
    validation_groups = {
        str(row["perceptual_group_id"])
        for row in records
        if int(row["fold"]) == args.validation_fold
    }
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"group-aware split overlap: {sorted(overlap)[:3]}")

    calibration_model = _build_model(torch, checkpoint, device)
    train_indices = np.flatnonzero(folds != args.validation_fold)
    validation_indices = np.flatnonzero(folds == args.validation_fold)
    calibration_history = []
    best = None
    best_epoch = 0
    for epoch in range(1, args.max_epochs + 1):
        train_history = _train_epochs(
            calibration_model,
            tensors=tensors,
            labels=labels,
            selected=train_indices,
            support=support[source_folds != args.validation_fold],
            support_labels=source_labels[source_folds != args.validation_fold],
            epochs=1,
            args=args,
            torch=torch,
            device=device,
        )
        validation_logits = _predict(
            calibration_model,
            tensors[validation_indices],
            torch=torch,
            device=device,
            batch_size=args.batch_size,
        )
        validation_metrics = _metrics(validation_logits, labels[validation_indices])
        row = {"epoch": epoch, "training": train_history[-1], "validation": validation_metrics}
        calibration_history.append(row)
        print(json.dumps(row), flush=True)
        key = (
            -validation_metrics["top3_miss_count"],
            -validation_metrics["top1_error_count"],
            -epoch,
        )
        if best is None or key > best:
            best = key
            best_epoch = epoch

    final_model = _build_model(torch, checkpoint, device)
    final_history = _train_epochs(
        final_model,
        tensors=tensors,
        labels=labels,
        selected=np.arange(len(tensors)),
        support=support,
        support_labels=source_labels,
        epochs=best_epoch,
        args=args,
        torch=torch,
        device=device,
    )
    evaluation_tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    evaluation_rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    evaluation_targets = np.asarray([int(row["target"]) for row in evaluation_rows], dtype=np.int64)
    evaluation_logits = _predict(
        final_model,
        evaluation_tensors,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
    )
    args.output_logits.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_logits, base=evaluation_logits, targets=evaluation_targets)
    output_checkpoint = {
        **checkpoint,
        "model_state_dict": copy.deepcopy(final_model.state_dict()),
        "robust_classifier_training": {
            "source_dataset": "single_objects",
            "mixed_support_sources": False,
            "manifest_sha256": _sha256(args.manifest),
            "recipe_sha256": clutter_roi_recipe_sha256(recipe),
            "views_per_source": args.views_per_source,
            "recipe_profile": args.recipe_profile,
            "neighbor_mask": args.neighbor_mask,
            "trainable_scope": args.trainable_scope,
            "selected_epoch_from_source_validation": best_epoch,
            "validation_fold": args.validation_fold,
            "development_evaluation_used_for_training_or_selection": False,
        },
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output_checkpoint)
    evaluation_by_fold = {
        str(fold): _metrics(
            evaluation_logits[
                np.asarray([int(row["fold"]) == fold for row in evaluation_rows], dtype=bool)
            ],
            evaluation_targets[
                np.asarray([int(row["fold"]) == fold for row in evaluation_rows], dtype=bool)
            ],
        )
        for fold in (0, 1, 2)
    }
    report = {
        "schema_version": "1.0",
        "status": "candidate",
        "training_source": "single_objects",
        "mixed_support_sources": False,
        "source_count": len(records),
        "derived_training_count": len(tensors),
        "recipe": asdict(recipe),
        "recipe_sha256": clutter_roi_recipe_sha256(recipe),
        "neighbor_mask": args.neighbor_mask,
        "group_aware_calibration": {
            "validation_fold": args.validation_fold,
            "train_group_count": len(train_groups),
            "validation_group_count": len(validation_groups),
            "group_overlap_count": len(overlap),
            "calibration_history": calibration_history,
            "selected_epoch": best_epoch,
        },
        "final_training": {
            "all_allowed_source_rows": True,
            "history": final_history,
        },
        "development_evaluation_used_for_training_or_selection": False,
        "development_evaluation": {
            "overall": _metrics(evaluation_logits, evaluation_targets),
            "by_fold": evaluation_by_fold,
        },
        "checkpoint_sha256": None,
    }
    report["checkpoint_sha256"] = _sha256(args.output_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a single_objects-only classifier against hard boundary clutter"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--tensor-cache", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-per-source", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--support-batch", type=int, default=48)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--recipe-profile", choices=("mild", "moderate", "hard"), default="moderate"
    )
    parser.add_argument("--neighbor-mask", action="store_true")
    parser.add_argument("--trainable-scope", choices=("head", "last_stage"), default="head")
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
    parser.add_argument("--head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--top3-margin", type=float, default=1.0)
    parser.add_argument("--top3-weight", type=float, default=0.75)
    parser.add_argument("--support-weight", type=float, default=0.35)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
