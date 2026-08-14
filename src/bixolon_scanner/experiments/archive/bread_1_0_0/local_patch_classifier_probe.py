from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ....evaluation.classifier_detector_fusion import cross_validated_selection, fusion_candidates
from ....training.models import build_dino_classifier, require_torch


def aggregate_patch_matches(query_to_support: np.ndarray) -> dict[str, np.ndarray]:
    """Aggregate [samples, query_patches, classes] best-patch similarities."""
    if query_to_support.ndim != 3:
        raise ValueError("patch similarities must have [samples, patches, classes] shape")
    patch_count = query_to_support.shape[1]
    return {
        "mean_all": query_to_support.mean(axis=1),
        "top_half": np.sort(query_to_support, axis=1)[:, -max(1, patch_count // 2) :].mean(axis=1),
        "top_quarter": np.sort(query_to_support, axis=1)[:, -max(1, patch_count // 4) :].mean(
            axis=1
        ),
        "maximum": query_to_support.max(axis=1),
    }


def _center_crop(tensors: Any, scale: float, torch: Any) -> Any:
    if scale == 1.0:
        return tensors
    height, width = tensors.shape[-2:]
    crop_height = max(1, round(height * scale))
    crop_width = max(1, round(width * scale))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = tensors[..., top : top + crop_height, left : left + crop_width]
    return torch.nn.functional.interpolate(
        cropped,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def _normalized_patches(values: Any, norm: Any, torch: Any) -> Any:
    normalized = torch.nn.functional.layer_norm(
        values,
        (values.shape[-1],),
        weight=norm["weight"],
        bias=norm["bias"],
        eps=norm["epsilon"],
    )
    return torch.nn.functional.normalize(normalized, p=2.0, dim=-1, eps=1e-12)


def extract_patch_scores(
    *,
    tensors: np.ndarray,
    support_patches: np.ndarray,
    support_labels: np.ndarray,
    norm_values: dict[str, np.ndarray | float],
    model: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    crop_scale: float,
) -> dict[str, np.ndarray]:
    norm = {
        "weight": torch.from_numpy(np.asarray(norm_values["weight"], dtype=np.float32)).to(device),
        "bias": torch.from_numpy(np.asarray(norm_values["bias"], dtype=np.float32)).to(device),
        "epsilon": float(norm_values["epsilon"]),
    }
    support = torch.from_numpy(np.array(support_patches, dtype=np.float32, copy=True)).to(device)
    support = _normalized_patches(support, norm, torch)
    support_by_class = [
        support[torch.from_numpy(support_labels == label).to(device)].flatten(0, 1)
        for label in range(20)
    ]
    aggregate_parts: dict[str, list[np.ndarray]] = {}
    symmetric_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            pixels = torch.from_numpy(
                np.asarray(tensors[start : start + batch_size], dtype=np.float32)
            ).to(device)
            pixels = _center_crop(pixels, crop_scale, torch)
            output = model.backbone.forward_features(pixels)["x_prenorm"]
            query = _normalized_patches(output, norm, torch)
            best_by_class = []
            reverse_by_class = []
            for class_support in support_by_class:
                similarities = torch.einsum("bqd,kd->bqk", query, class_support)
                best_by_class.append(similarities.max(dim=2).values)
                reverse = similarities.max(dim=1).values
                reverse_by_class.append(
                    torch.topk(
                        reverse,
                        k=min(query.shape[1], reverse.shape[1]),
                        dim=1,
                    ).values.mean(dim=1)
                )
            best = torch.stack(best_by_class, dim=2).cpu().numpy()
            aggregates = aggregate_patch_matches(best)
            for name, values in aggregates.items():
                aggregate_parts.setdefault(name, []).append(values.astype(np.float32))
            symmetric = 0.5 * (
                aggregates["mean_all"] + torch.stack(reverse_by_class, dim=1).cpu().numpy()
            )
            symmetric_parts.append(symmetric.astype(np.float32))
    result = {name: np.concatenate(parts) for name, parts in aggregate_parts.items()}
    result["symmetric_chamfer"] = np.concatenate(symmetric_parts)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line
    ]
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    dino_logits = np.load(args.classifier_logits)
    tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    support_patches = np.load(args.support_patches, mmap_mode="r")
    cache = np.load(args.training_features)
    norm_cache = np.load(args.patch_norm)
    torch = require_torch()
    device = torch.device("cpu" if args.cpu else "cuda")
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        20,
        weights_path=args.weights,
        hub_repository=args.hub_repository,
        feature_l2_normalize=True,
    ).to(device)
    model.eval()
    scores = extract_patch_scores(
        tensors=tensors,
        support_patches=support_patches,
        support_labels=cache["support_labels"],
        norm_values={
            "weight": norm_cache["weight"],
            "bias": norm_cache["bias"],
            "epsilon": float(norm_cache["epsilon"]),
        },
        model=model,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
        crop_scale=args.crop_scale,
    )
    all_candidates = {}
    patch_metrics = {}
    for method, patch_scores in scores.items():
        predictions = patch_scores.argmax(axis=1)
        correct = int(np.count_nonzero(predictions == targets))
        patch_metrics[method] = {
            "correct": correct,
            "accuracy": correct / len(targets),
        }
        for name, values in fusion_candidates(dino_logits, patch_scores).items():
            all_candidates[f"{method}:{name}"] = values
    cross_validated = cross_validated_selection(all_candidates, targets, folds)
    best_in_sample_name = max(
        sorted(all_candidates),
        key=lambda name: int(np.count_nonzero(all_candidates[name] == targets)),
    )
    best_in_sample_correct = int(np.count_nonzero(all_candidates[best_in_sample_name] == targets))
    report = {
        "evaluation": "local_patch_classifier_diagnostic_only",
        "promotion_status": "diagnostic_only",
        "sample_count": len(targets),
        "training_original_count": len(cache["support_labels"]),
        "support_patch_shape": list(support_patches.shape),
        "crop_scale": args.crop_scale,
        "patch_methods": patch_metrics,
        "candidate_count": len(all_candidates),
        "best_in_sample_policy": {
            "name": best_in_sample_name,
            "correct": best_in_sample_correct,
            "accuracy": best_in_sample_correct / len(targets),
        },
        "cross_validated_policy": cross_validated,
        "passes_top1_gate": cross_validated["top1_accuracy"] >= 0.99,
        "limitations": [
            "Fusion policy is selected on grouped development folds and is not a locked test result.",
            "Patch support comes only from the canonical 200 training originals.",
            "No ONNX export, provider parity or end-to-end latency evidence is produced unless the gate passes.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe DINOv3 local patch matching")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--classifier-logits", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--support-patches", type=Path, required=True)
    parser.add_argument("--patch-norm", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hub-repository",
        default="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
    )
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
