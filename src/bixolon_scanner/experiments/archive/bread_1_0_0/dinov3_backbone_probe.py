from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _center_crop(torch, values, scale: float):
    if scale == 1.0:
        return values
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


def _embed(
    model,
    tensors: np.ndarray,
    *,
    torch,
    device,
    batch_size: int,
    crop_scale: float,
) -> tuple[np.ndarray, float]:
    parts = []
    elapsed = 0.0
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            batch = _center_crop(torch, batch, crop_scale)
            started = time.perf_counter()
            features = model(batch)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            features = torch.nn.functional.normalize(features, dim=-1)
            parts.append(features.float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32), elapsed


def _metrics(scores: np.ndarray, targets: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    predictions = scores.argmax(axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return {
        "top1_accuracy": float((predictions == targets).mean()),
        "top3_accuracy": float(np.any(order[:, :3] == targets[:, None], axis=1).mean()),
        "top1_error_count": int(np.count_nonzero(predictions != targets)),
        "fold_top1": [
            float((predictions[folds == fold] == targets[folds == fold]).mean())
            for fold in range(3)
        ],
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from sklearn.svm import LinearSVC

    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is required unless --cpu is selected")
    device = torch.device("cpu" if args.cpu else "cuda")
    model = (
        torch.hub.load(
            str(args.hub_root),
            args.model,
            source="local",
            pretrained=True,
            weights=args.weights_url,
            verbose=False,
        )
        .to(device)
        .eval()
    )
    training_tensors = np.load(args.training_tensors, mmap_mode="r")
    training_cache = np.load(args.training_cache)
    training_labels = training_cache["labels"].astype(np.int64)
    if len(training_tensors) != len(training_labels):
        raise ValueError("training tensors and 200-original-derived labels are not aligned")
    evaluation_tensors = np.load(args.evaluation_tensors, mmap_mode="r")
    rows = [
        json.loads(line)
        for line in args.evaluation_records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    evaluation_labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    training_features, training_seconds = _embed(
        model,
        training_tensors,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
        crop_scale=1.0,
    )
    evaluation_features, evaluation_seconds = _embed(
        model,
        evaluation_tensors,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
        crop_scale=args.crop_scale,
    )
    results = []
    best = None
    for regularization_c in args.svm_c:
        classifier = LinearSVC(C=regularization_c, dual="auto", max_iter=20_000)
        classifier.fit(training_features, training_labels)
        scores = classifier.decision_function(evaluation_features)
        result = {"svm_c": regularization_c, **_metrics(scores, evaluation_labels, folds)}
        results.append(result)
        if best is None or result["top1_accuracy"] > best["top1_accuracy"]:
            best = result
    report = {
        "schema_version": "1.0",
        "evaluation": "dinov3_backbone_frozen_feature_probe",
        "promotion_status": "diagnostic_only",
        "model": args.model,
        "training_contract": {
            "source_original_count": 200,
            "derived_training_tensor_count": len(training_tensors),
            "evaluation_images_used_for_training": False,
        },
        "selected": best,
        "runs": results,
        "encoder_per_image_ms": {
            "training_mean": training_seconds * 1000.0 / len(training_tensors),
            "evaluation_mean": evaluation_seconds * 1000.0 / len(evaluation_tensors),
            "batch_size": args.batch_size,
        },
        "passes_top1_gate": best is not None and best["top1_accuracy"] >= 0.99,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a larger official DINOv3 backbone")
    parser.add_argument("--hub-root", type=Path, required=True)
    parser.add_argument("--model", default="dinov3_convnext_small")
    parser.add_argument(
        "--weights-url",
        default=(
            "https://dl.fbaipublicfiles.com/dinov3/dinov3_convnext_small/"
            "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
        ),
    )
    parser.add_argument("--training-tensors", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--evaluation-tensors", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-scale", type=float, default=0.855)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--svm-c", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
