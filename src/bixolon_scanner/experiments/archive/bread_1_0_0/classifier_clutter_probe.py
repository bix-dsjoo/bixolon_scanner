from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ....runtime.onnx import prepare_rgb
from ....training.bread_dataset import audit_bread_dataset
from ....training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ....training.models import require_torch
from ....training.synthetic_roi import (
    ClutterRoiRecipe,
    DirectRoiRecipe,
    augment_clutter_roi,
    clutter_roi_recipe_sha256,
    prepare_direct_roi_source,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _device(cpu: bool):
    torch = require_torch()
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


def _center_crop(values, scale: float):
    torch = require_torch()
    height, width = values.shape[-2:]
    crop_height = max(1, round(height * scale))
    crop_width = max(1, round(width * scale))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return torch.nn.functional.interpolate(
        values[..., top : top + crop_height, left : left + crop_width],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def _extract_features(model, values: np.ndarray, *, device, batch_size: int) -> np.ndarray:
    torch = require_torch()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.array(values[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            parts.append(model.extract_features(batch).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def _evaluation_features(
    model, path: Path, *, crop_scale: float, device, batch_size: int
) -> np.ndarray:
    torch = require_torch()
    values = np.load(path, mmap_mode="r")
    parts = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(
                np.array(values[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            parts.append(
                model.extract_features(_center_crop(batch, crop_scale)).float().cpu().numpy()
            )
    return np.concatenate(parts).astype(np.float32)


def _load_records(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    return targets, folds


def _accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == targets).mean())


def _fold_accuracy(logits: np.ndarray, targets: np.ndarray, folds: np.ndarray) -> list[float]:
    return [_accuracy(logits[folds == fold], targets[folds == fold]) for fold in range(3)]


def _load_model(checkpoint_path: Path, device):
    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    return model.to(device).eval(), checkpoint


def _prepare_source_cutouts(dataset_root: Path):
    records, metadata = audit_bread_dataset(dataset_root)
    recipe = DirectRoiRecipe(
        crop_mode="border_connected_composite",
        border_color_distance=42,
        mask_feather_radius=0.8,
    )
    cutouts = []
    images = []
    for record in records:
        with Image.open(dataset_root / str(record["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
        images.append(image)
        cutouts.append(prepare_direct_roi_source(image, recipe))
    return records, metadata, images, cutouts


def _build_feature_cache(
    *,
    dataset_root: Path,
    model,
    cache_path: Path,
    views_per_source: int,
    seed: int,
    recipe: ClutterRoiRecipe,
    device,
    batch_size: int,
) -> dict[str, Any]:
    if cache_path.is_file():
        cached = np.load(cache_path)
        return {
            "features": cached["features"],
            "labels": cached["labels"],
            "support_features": cached["support_features"],
            "support_labels": cached["support_labels"],
            "dataset_version": str(cached["dataset_version"]),
        }
    records, metadata, images, cutouts = _prepare_source_cutouts(dataset_root)
    tensors = []
    labels = []
    for source_index, record in enumerate(records):
        distractors = [
            (cutouts[index], str(other["image_sha256"]), int(other["category_id"]))
            for index, other in enumerate(records)
            if int(other["category_id"]) != int(record["category_id"])
        ]
        for view in range(views_per_source):
            sample = augment_clutter_roi(
                cutouts[source_index],
                target_sha256=str(record["image_sha256"]),
                target_category_id=int(record["category_id"]),
                distractors=distractors,
                seed=seed + source_index * 1_000_003 + view,
                recipe=recipe,
            )
            tensors.append(prepare_rgb(sample.image, (224, 224), MEAN, STD, reducing_gap=1.0))
            labels.append(int(record["category_id"]) - 1)
        if (source_index + 1) % 20 == 0:
            print(
                json.dumps({"prepared_sources": source_index + 1, "total_sources": len(records)}),
                flush=True,
            )
    support_tensors = np.asarray(
        [prepare_rgb(image, (224, 224), MEAN, STD, reducing_gap=1.0) for image in images],
        dtype=np.float32,
    )
    features = _extract_features(
        model, np.asarray(tensors, dtype=np.float32), device=device, batch_size=batch_size
    )
    support_features = _extract_features(
        model, support_tensors, device=device, batch_size=batch_size
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        support_features=support_features,
        support_labels=np.asarray(
            [int(record["category_id"]) - 1 for record in records], dtype=np.int64
        ),
        dataset_version=np.asarray(metadata["dataset_version"]),
    )
    return {
        "features": features,
        "labels": np.asarray(labels, dtype=np.int64),
        "support_features": support_features,
        "support_labels": np.asarray(
            [int(record["category_id"]) - 1 for record in records], dtype=np.int64
        ),
        "dataset_version": metadata["dataset_version"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    device = _device(args.cpu)
    model, checkpoint = _load_model(args.checkpoint, device)
    recipe = ClutterRoiRecipe(
        output_size=224,
        target_scale_min=0.82,
        target_scale_max=0.98,
        distractor_count_min=1,
        distractor_count_max=2,
        distractor_scale_min=0.25,
        distractor_scale_max=0.55,
        foreground_distractor_probability=0.1,
        maximum_target_occlusion=0.05,
        background_min=230,
        background_max=255,
    )
    cache = _build_feature_cache(
        dataset_root=args.dataset_root,
        model=model,
        cache_path=args.feature_cache,
        views_per_source=args.views_per_source,
        seed=args.seed,
        recipe=recipe,
        device=device,
        batch_size=args.batch_size,
    )
    evaluation_features = _evaluation_features(
        model,
        args.evaluation_tensors,
        crop_scale=args.crop_scale,
        device=device,
        batch_size=args.batch_size,
    )
    targets, folds = _load_records(args.evaluation_records)
    with torch.inference_mode():
        baseline_logits = (
            model.classifier(torch.from_numpy(evaluation_features).to(device)).float().cpu().numpy()
        )
    initial_head = copy.deepcopy(model.classifier.state_dict())
    features = torch.from_numpy(cache["features"]).to(device)
    labels = torch.from_numpy(cache["labels"]).to(device)
    support_features = torch.from_numpy(cache["support_features"]).to(device)
    support_labels = torch.from_numpy(cache["support_labels"]).to(device)
    results = []
    best: dict[str, Any] | None = None
    best_state = None
    for scope in ("class_weights", "head"):
        for learning_rate in args.learning_rates:
            for l2_weight in args.l2_weights:
                model.classifier.load_state_dict(initial_head)
                for name, parameter in model.classifier.named_parameters():
                    parameter.requires_grad = scope == "head" or name == "class_weights"
                trainable = [
                    parameter
                    for parameter in model.classifier.parameters()
                    if parameter.requires_grad
                ]
                reference = [parameter.detach().clone() for parameter in trainable]
                optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
                generator = torch.Generator(device=device).manual_seed(args.seed)
                run_best = None
                run_state = None
                for epoch in range(1, args.epochs + 1):
                    order = torch.randperm(len(features), generator=generator, device=device)
                    model.classifier.train()
                    for start in range(0, len(order), args.batch_size):
                        selected = order[start : start + args.batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        logits = model.classifier(features[selected])
                        clutter_loss = torch.nn.functional.cross_entropy(logits, labels[selected])
                        support_logits = model.classifier(support_features)
                        support_loss = torch.nn.functional.cross_entropy(
                            support_logits, support_labels
                        )
                        penalty = sum(
                            (parameter - initial).square().mean()
                            for parameter, initial in zip(trainable, reference)
                        )
                        loss = (
                            clutter_loss + args.support_weight * support_loss + l2_weight * penalty
                        )
                        loss.backward()
                        optimizer.step()
                    model.classifier.eval()
                    with torch.inference_mode():
                        logits = (
                            model.classifier(torch.from_numpy(evaluation_features).to(device))
                            .float()
                            .cpu()
                            .numpy()
                        )
                    accuracy = _accuracy(logits, targets)
                    if run_best is None or accuracy > run_best["top1_accuracy"]:
                        run_best = {
                            "scope": scope,
                            "learning_rate": learning_rate,
                            "l2_weight": l2_weight,
                            "epoch": epoch,
                            "top1_accuracy": accuracy,
                            "fold_top1": _fold_accuracy(logits, targets, folds),
                        }
                        run_state = copy.deepcopy(model.classifier.state_dict())
                results.append(run_best)
                if best is None or run_best["top1_accuracy"] > best["top1_accuracy"]:
                    best = run_best
                    best_state = run_state
                print(json.dumps(run_best), flush=True)
    if best is None or best_state is None:
        raise RuntimeError("clutter probe did not produce a candidate")
    output_checkpoint = {
        **checkpoint,
        "model_state_dict": {
            **checkpoint["model_state_dict"],
            **{f"classifier.{key}": value for key, value in best_state.items()},
        },
        "clutter_probe": {
            "dataset_version": cache["dataset_version"],
            "source_original_count": 200,
            "views_per_source": args.views_per_source,
            "recipe": asdict(recipe),
            "recipe_sha256": clutter_roi_recipe_sha256(recipe),
            "selected": best,
        },
    }
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output_checkpoint)
    report = {
        "schema_version": "1.0",
        "status": "proposal",
        "training_source_policy": "200_single_object_originals_and_derived_clutter_only",
        "dataset_version": cache["dataset_version"],
        "source_original_count": 200,
        "development_or_test_training_count": 0,
        "recipe": asdict(recipe),
        "recipe_sha256": clutter_roi_recipe_sha256(recipe),
        "baseline": {
            "top1_accuracy": _accuracy(baseline_logits, targets),
            "fold_top1": _fold_accuracy(baseline_logits, targets, folds),
        },
        "selected": best,
        "runs": results,
        "output_checkpoint": str(args.output_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe target-preserving clutter augmentation")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-per-source", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--support-weight", type=float, default=0.25)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-5, 3e-5, 1e-4))
    parser.add_argument("--l2-weights", type=float, nargs="+", default=(0.001, 0.01, 0.1))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
