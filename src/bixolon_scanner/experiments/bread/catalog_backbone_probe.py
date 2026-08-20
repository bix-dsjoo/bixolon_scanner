from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ...contracts.catalog import sha256_file
from ...runtime.onnx import prepare_rgb
from .classifier_200_only import (
    fit_small_sample_head,
    nested_oof_fit,
    select_finite_oof_policy,
)


def _load_records(manifest: Path, dataset_root: Path) -> tuple[list[dict], list[Image.Image]]:
    records = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(records) != 200:
        raise ValueError("the Catalog backbone probe requires exactly 20 classes x 10 images")
    if sorted({int(row["category_id"]) for row in records}) != list(range(1, 21)):
        raise ValueError("the Catalog backbone probe requires categories 1 through 20")
    images = []
    for row in records:
        path = (dataset_root / str(row["image_path"])).resolve()
        if dataset_root.resolve() not in path.parents or sha256_file(path) != row["image_sha256"]:
            raise ValueError("Catalog source image identity does not match the locked manifest")
        with Image.open(path) as source:
            images.append(ImageOps.exif_transpose(source).convert("RGB").copy())
    return records, images


def _extract_features(
    images: list[Image.Image], model_dir: Path, *, family: str, batch_size: int
) -> np.ndarray:
    import torch
    from transformers import AutoModel

    model = (
        AutoModel.from_pretrained(model_dir.resolve().as_posix(), local_files_only=True)
        .cuda()
        .eval()
    )
    features = []
    mean = (0.5, 0.5, 0.5) if family == "siglip" else (0.485, 0.456, 0.406)
    std = (0.5, 0.5, 0.5) if family == "siglip" else (0.229, 0.224, 0.225)
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            tensors = np.stack(
                [
                    prepare_rgb(
                        image,
                        (224, 224),
                        mean,
                        std,
                        reducing_gap=1.0,
                    )
                    for image in images[start : start + batch_size]
                ]
            )
            pixel_values = torch.from_numpy(tensors).cuda(non_blocking=True)
            if family == "siglip":
                output = model.vision_model(pixel_values=pixel_values).pooler_output
            else:
                output = model(pixel_values=pixel_values).last_hidden_state[:, 0]
            features.append(output.float().cpu().numpy())
    return np.concatenate(features).astype(np.float32)


def run(args: argparse.Namespace) -> dict:
    records, images = _load_records(args.manifest, args.dataset_root)
    try:
        features = _extract_features(
            images, args.model_dir, family=args.family, batch_size=args.batch_size
        )
    finally:
        for image in images:
            image.close()
    labels = np.asarray([int(row["category_id"]) - 1 for row in records], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in records], dtype=np.int64)
    cache = {
        "train_features": features,
        "train_labels": labels,
        "train_folds": folds,
        "validation_features": features,
        "validation_labels": labels,
        "validation_folds": folds,
    }
    candidates = [
        {"kind": "regularized_linear_ridge", "alpha": alpha} for alpha in (0.01, 0.1, 1.0, 10.0)
    ]
    oof_logits, folds_report, selected = nested_oof_fit(
        cache, head_candidates=candidates, class_count=20
    )
    ranking = np.argsort(-oof_logits, axis=1, kind="stable")
    top1 = ranking[:, 0] == labels
    top3 = np.any(ranking[:, :3] == labels[:, None], axis=1)
    policy = select_finite_oof_policy(oof_logits, labels)
    adapter_sha256 = None
    if args.adapter_checkpoint is not None:
        import torch

        weight, bias = fit_small_sample_head(features, labels, candidate=selected, class_count=20)
        args.adapter_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "2.0",
                "backbone_kind": args.family,
                "manifest_sha256": sha256_file(args.manifest),
                "selected_head": selected,
                "approval_threshold": policy["approval_threshold"],
                "top3_safety_threshold": policy["top3_safety_threshold"],
                "model_state_dict": {
                    "classifier.weight": torch.from_numpy(weight.T.copy()),
                    "classifier.bias": torch.from_numpy(bias.copy()),
                },
            },
            args.adapter_checkpoint,
        )
        adapter_sha256 = sha256_file(args.adapter_checkpoint)
    weights = args.model_dir / "model.safetensors"
    report = {
        "schema_version": "2.0",
        "candidate_id": f"catalog-{args.family}-original-10shot-probe",
        "lifecycle": "rejected" if policy["approved_rate"] < 0.9 else "active",
        "evidence_role": "support_leave-group-out_development_probe",
        "promotion_evidence": False,
        "limitations": [
            "same 10-shot source supplies fold-separated fit and validation",
            "original views only; no ROI-domain evaluation or ONNX latency measurement",
            *(
                ["cached checkpoint identifies itself as SigLIP rather than SigLIP 2"]
                if args.family == "siglip"
                else []
            ),
        ],
        "source": {
            "manifest_sha256": sha256_file(args.manifest),
            "image_count": len(records),
            "model_weights_sha256": sha256_file(weights),
        },
        "feature_shape": list(features.shape),
        "adapter_checkpoint_sha256": adapter_sha256,
        "selected_head": selected,
        "folds": folds_report,
        "oof": {
            "sample_count": len(labels),
            "top1_correct_count": int(np.count_nonzero(top1)),
            "top1_accuracy": float(np.mean(top1)),
            "top3_miss_count": int(np.count_nonzero(~top3)),
            "top3_accuracy": float(np.mean(top3)),
            "selective_policy": policy,
        },
        "decision": (
            "stop_before_runtime_export"
            if policy["approved_rate"] < 0.9
            else "continue_to_augmented_roi_and_latency_evaluation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe a frozen Catalog backbone")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--family", choices=("siglip", "dinov2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
