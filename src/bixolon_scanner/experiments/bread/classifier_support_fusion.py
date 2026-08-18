from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...runtime.onnx import prepare_rgb
from ...training.data import read_manifest
from ...training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ...training.models import require_torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VIEW_NAMES = (
    "base",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
    "rot15",
    "rot-15",
    "rot30",
    "rot-30",
)


def class_support_scores(
    evaluation_features: np.ndarray,
    support_features: np.ndarray,
    support_labels: np.ndarray,
    *,
    class_count: int,
    top_k: int,
) -> np.ndarray:
    """Return per-class mean top-k cosine similarity to allowed support rows."""
    if evaluation_features.ndim != 2 or support_features.ndim != 2:
        raise ValueError("evaluation and support features must be matrices")
    if evaluation_features.shape[1] != support_features.shape[1]:
        raise ValueError("evaluation and support feature widths differ")
    if support_labels.shape != (len(support_features),):
        raise ValueError("support labels are not aligned with support features")
    if class_count < 2 or top_k < 1:
        raise ValueError("class_count and top_k must be positive")
    evaluation = evaluation_features.astype(np.float64)
    support = support_features.astype(np.float64)
    evaluation /= np.maximum(np.linalg.norm(evaluation, axis=1, keepdims=True), 1e-12)
    support /= np.maximum(np.linalg.norm(support, axis=1, keepdims=True), 1e-12)
    similarities = evaluation @ support.T
    scores = np.empty((len(evaluation), class_count), dtype=np.float32)
    for class_index in range(class_count):
        class_scores = similarities[:, support_labels == class_index]
        if class_scores.shape[1] < top_k:
            raise ValueError(f"class {class_index} has fewer than {top_k} support rows")
        selected = np.partition(class_scores, -top_k, axis=1)[:, -top_k:]
        scores[:, class_index] = selected.mean(axis=1)
    return scores


def _center_crop(torch, values, scale: float):
    if scale == 1.0:
        return values
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


def _view(torch, values, name: str):
    if name == "base":
        return values
    if name == "hflip":
        return torch.flip(values, dims=(-1,))
    if name == "vflip":
        return torch.flip(values, dims=(-2,))
    if name == "rot90":
        return torch.rot90(values, 1, dims=(-2, -1))
    if name == "rot180":
        return torch.rot90(values, 2, dims=(-2, -1))
    if name == "rot270":
        return torch.rot90(values, 3, dims=(-2, -1))
    from torchvision.transforms.functional import InterpolationMode, rotate

    return rotate(
        values,
        float(name.removeprefix("rot")),
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0.0,
    )


def _extract(model, values):
    output = model.backbone.forward_features(values)
    global_features = output["x_norm_clstoken"]
    patch_features = output["x_norm_patchtokens"]
    logits = model.classifier(global_features)
    return global_features, patch_features, logits


def _patch_support_scores(
    evaluation_patches,
    support_patches,
    support_labels: np.ndarray,
    *,
    torch,
    class_count: int,
) -> dict[str, np.ndarray]:
    evaluation = torch.nn.functional.normalize(evaluation_patches, p=2.0, dim=-1)
    support = torch.nn.functional.normalize(support_patches, p=2.0, dim=-1)
    best_by_class = []
    for class_index in range(class_count):
        selected = torch.from_numpy(support_labels == class_index).to(support.device)
        class_support = support[selected].flatten(0, 1)
        if class_support.shape[0] == 0:
            raise ValueError(f"class {class_index} has no support patches")
        similarities = torch.einsum("bqd,kd->bqk", evaluation, class_support)
        best_by_class.append(similarities.max(dim=2).values)
    best = torch.stack(best_by_class, dim=2)
    patch_count = best.shape[1]
    return {
        "patchmean": best.mean(dim=1).float().cpu().numpy(),
        "patchhalf": torch.topk(best, k=max(1, patch_count // 2), dim=1)
        .values.mean(dim=1)
        .float()
        .cpu()
        .numpy(),
        "patchquarter": torch.topk(best, k=max(1, patch_count // 4), dim=1)
        .values.mean(dim=1)
        .float()
        .cpu()
        .numpy(),
        "patchmax": best.max(dim=1).values.float().cpu().numpy(),
    }


def _support_tensors(
    manifest: Path, dataset_root: Path, input_size: int
) -> tuple[np.ndarray, np.ndarray]:
    records = [
        row
        for row in read_manifest(manifest)
        if row.get("record_type") == "classification" and row.get("split") == "development"
    ]
    if not records:
        raise ValueError("classifier manifest has no development support rows")
    tensors = []
    labels = []
    for row in records:
        source_path = (dataset_root / str(row["image_path"])).resolve()
        source_path.relative_to(dataset_root.resolve())
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            tensors.append(
                prepare_rgb(
                    image,
                    (input_size, input_size),
                    MEAN,
                    STD,
                    reducing_gap=1.0,
                )
            )
        labels.append(int(row["category_id"]) - 1)
    return np.stack(tensors).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, axis=1, kind="stable")
    return {
        "top1_accuracy": float((order[:, 0] == targets).mean()),
        "top1_error_count": int(np.count_nonzero(order[:, 0] != targets)),
        "top3_accuracy": float(np.any(order[:, :3] == targets[:, None], axis=1).mean()),
        "top3_miss_count": int(np.count_nonzero(~np.any(order[:, :3] == targets[:, None], axis=1))),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository=args.hub_repository,
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    device = torch.device("cpu" if args.cpu else "cuda")
    model = model.to(device).eval()
    support_tensors, support_labels = _support_tensors(
        args.classifier_manifest, args.dataset_root, args.input_size
    )
    support_features = []
    support_patches = []
    with torch.inference_mode():
        for start in range(0, len(support_tensors), args.batch_size):
            batch = torch.from_numpy(support_tensors[start : start + args.batch_size]).to(device)
            features, patches, _ = _extract(model, batch)
            support_features.append(features.float().cpu().numpy())
            support_patches.append(patches.float().cpu())
    support_features_array = np.concatenate(support_features).astype(np.float32)
    support_patches_tensor = torch.cat(support_patches).to(device)

    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    if len(tensors) != len(targets):
        raise ValueError("evaluation tensors and records are not aligned")
    outputs: dict[str, Any] = {"targets": targets}
    with torch.inference_mode():
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + args.batch_size], dtype=np.float32, copy=True)
            ).to(device)
            batch = _center_crop(torch, batch, args.crop_scale)
            for name in VIEW_NAMES:
                features, patches, head_logits = _extract(model, _view(torch, batch, name))
                outputs.setdefault(f"head_{name}", []).append(head_logits.float().cpu().numpy())
                feature_values = features.float().cpu().numpy()
                for top_k in args.support_top_k:
                    outputs.setdefault(f"support{top_k}_{name}", []).append(
                        class_support_scores(
                            feature_values,
                            support_features_array,
                            support_labels,
                            class_count=args.class_count,
                            top_k=top_k,
                        )
                    )
                for prefix, values in _patch_support_scores(
                    patches,
                    support_patches_tensor,
                    support_labels,
                    torch=torch,
                    class_count=args.class_count,
                ).items():
                    outputs.setdefault(f"{prefix}_{name}", []).append(values)
    final_outputs = {
        name: value if isinstance(value, np.ndarray) else np.concatenate(value).astype(np.float32)
        for name, value in outputs.items()
    }
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.logits_output, **final_outputs)
    families = {}
    for prefix in (
        "head",
        *(f"support{value}" for value in args.support_top_k),
        "patchmean",
        "patchhalf",
        "patchquarter",
        "patchmax",
    ):
        mean_scores = np.mean([final_outputs[f"{prefix}_{name}"] for name in VIEW_NAMES], axis=0)
        families[prefix] = _metrics(mean_scores, targets)
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_classifier_support_fusion_features",
        "source_dataset": args.source_dataset,
        "mixed_support_sources": False,
        "support_count": len(support_labels),
        "sample_count": len(targets),
        "view_count": len(VIEW_NAMES),
        "families": families,
        "limitations": [
            "This artifact is a grouped-development probe, not independent test evidence.",
            "Fusion policy selection and Worker parity are evaluated separately.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract head and support-similarity logits")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    parser.add_argument(
        "--source-dataset", choices=("single_objects", "single_objects_2"), required=True
    )
    parser.add_argument(
        "--hub-repository",
        default="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--class-count", type=int, default=20)
    parser.add_argument("--support-top-k", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
