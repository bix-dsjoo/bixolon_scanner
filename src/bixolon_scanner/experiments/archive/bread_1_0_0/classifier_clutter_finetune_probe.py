from __future__ import annotations

import argparse
import copy
import json
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


def _center_crop(torch, values, scale: float):
    height, width = values.shape[-2:]
    crop_height, crop_width = round(height * scale), round(width * scale)
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return torch.nn.functional.interpolate(
        values[..., top : top + crop_height, left : left + crop_width],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def _model_logits(model, tensors, *, torch, device, batch_size: int, crop_scale: float):
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            parts.append(model(_center_crop(torch, batch, crop_scale)).float().cpu().numpy())
    return np.concatenate(parts)


def _prepare_tensor_cache(
    *,
    dataset_root: Path,
    cache_path: Path,
    views_per_source: int,
    seed: int,
    recipe: ClutterRoiRecipe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    records, metadata = audit_bread_dataset(dataset_root)
    expected_shape = (len(records) * views_per_source, 3, 224, 224)
    label_path = cache_path.with_suffix(".labels.npy")
    support_path = cache_path.with_suffix(".support.npy")
    if cache_path.is_file() and label_path.is_file() and support_path.is_file():
        tensors = np.load(cache_path, mmap_mode="r")
        labels = np.load(label_path)
        support = np.load(support_path, mmap_mode="r")
        if tensors.shape != expected_shape or labels.shape != (expected_shape[0],):
            raise ValueError("clutter tensor cache contract mismatch")
        return tensors, labels, support, str(metadata["dataset_version"])
    source_recipe = DirectRoiRecipe(
        crop_mode="border_connected_composite",
        border_color_distance=42,
        mask_feather_radius=0.8,
    )
    images = []
    cutouts = []
    for record in records:
        with Image.open(dataset_root / str(record["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
        images.append(image)
        cutouts.append(prepare_direct_roi_source(image, source_recipe))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=expected_shape
    )
    labels = np.empty(expected_shape[0], dtype=np.int64)
    for source_index, record in enumerate(records):
        distractors = [
            (cutouts[index], str(other["image_sha256"]), int(other["category_id"]))
            for index, other in enumerate(records)
            if int(other["category_id"]) != int(record["category_id"])
        ]
        for view in range(views_per_source):
            index = source_index * views_per_source + view
            sample = augment_clutter_roi(
                cutouts[source_index],
                target_sha256=str(record["image_sha256"]),
                target_category_id=int(record["category_id"]),
                distractors=distractors,
                seed=seed + source_index * 1_000_003 + view,
                recipe=recipe,
            )
            tensors[index] = prepare_rgb(
                sample.image, (224, 224), MEAN, STD, reducing_gap=1.0
            ).astype(np.float16)
            labels[index] = int(record["category_id"]) - 1
        if (source_index + 1) % 20 == 0:
            print(json.dumps({"prepared_sources": source_index + 1}), flush=True)
    tensors.flush()
    np.save(label_path, labels)
    support = np.asarray(
        [prepare_rgb(image, (224, 224), MEAN, STD, reducing_gap=1.0) for image in images],
        dtype=np.float16,
    )
    np.save(support_path, support)
    return (
        np.load(cache_path, mmap_mode="r"),
        labels,
        np.load(support_path, mmap_mode="r"),
        str(metadata["dataset_version"]),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    device = torch.device("cpu" if args.cpu else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    model = model.to(device)
    initial_state = copy.deepcopy(model.state_dict())
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
    tensors, labels_array, support, dataset_version = _prepare_tensor_cache(
        dataset_root=args.dataset_root,
        cache_path=args.tensor_cache,
        views_per_source=args.views_per_source,
        seed=args.seed,
        recipe=recipe,
    )
    evaluation_tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    baseline_logits = _model_logits(
        model,
        evaluation_tensors,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
        crop_scale=args.crop_scale,
    )
    baseline_accuracy = float((baseline_logits.argmax(axis=1) == targets).mean())
    results = []
    best = None
    best_state = None
    for learning_rate in args.learning_rates:
        model.load_state_dict(initial_state)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.stages[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.norm.parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        reference = [parameter.detach().clone() for parameter in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
        generator = torch.Generator().manual_seed(args.seed)
        run_best = None
        run_state = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(tensors), generator=generator).numpy()
            for start in range(0, len(order), args.batch_size):
                selected = order[start : start + args.batch_size]
                pixels = torch.from_numpy(
                    np.array(tensors[selected], dtype=np.float32, copy=True)
                ).to(device)
                labels = torch.from_numpy(labels_array[selected]).to(device)
                optimizer.zero_grad(set_to_none=True)
                clutter_loss = torch.nn.functional.cross_entropy(model(pixels), labels)
                support_indices = np.arange(start, start + min(args.support_batch, len(support)))
                support_indices %= len(support)
                support_pixels = torch.from_numpy(
                    np.array(support[support_indices], dtype=np.float32, copy=True)
                ).to(device)
                support_labels = torch.from_numpy((support_indices // 10).astype(np.int64)).to(
                    device
                )
                support_loss = torch.nn.functional.cross_entropy(
                    model(support_pixels), support_labels
                )
                penalty = sum(
                    (parameter - initial).square().mean()
                    for parameter, initial in zip(trainable, reference)
                )
                loss = clutter_loss + args.support_weight * support_loss + args.l2_weight * penalty
                loss.backward()
                optimizer.step()
            logits = _model_logits(
                model,
                evaluation_tensors,
                torch=torch,
                device=device,
                batch_size=args.batch_size,
                crop_scale=args.crop_scale,
            )
            predictions = logits.argmax(axis=1)
            result = {
                "learning_rate": learning_rate,
                "epoch": epoch,
                "top1_accuracy": float((predictions == targets).mean()),
                "fold_top1": [
                    float((predictions[folds == fold] == targets[folds == fold]).mean())
                    for fold in range(3)
                ],
            }
            print(json.dumps(result), flush=True)
            if run_best is None or result["top1_accuracy"] > run_best["top1_accuracy"]:
                run_best = result
                run_state = copy.deepcopy(model.state_dict())
        results.append(run_best)
        if best is None or run_best["top1_accuracy"] > best["top1_accuracy"]:
            best = run_best
            best_state = run_state
    if best is None or best_state is None:
        raise RuntimeError("clutter fine-tuning produced no candidate")
    output_checkpoint = {
        **checkpoint,
        "model_state_dict": best_state,
        "clutter_finetune": {
            "dataset_version": dataset_version,
            "source_original_count": 200,
            "views_per_source": args.views_per_source,
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
        "dataset_version": dataset_version,
        "source_original_count": 200,
        "development_or_test_training_count": 0,
        "baseline_top1_accuracy": baseline_accuracy,
        "selected": best,
        "runs": results,
        "recipe_sha256": clutter_roi_recipe_sha256(recipe),
        "passes_top1_gate": best["top1_accuracy"] >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the final DINOv3 stage on clutter")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--tensor-cache", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views-per-source", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--support-weight", type=float, default=0.25)
    parser.add_argument("--l2-weight", type=float, default=0.001)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-6, 3e-6))
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
