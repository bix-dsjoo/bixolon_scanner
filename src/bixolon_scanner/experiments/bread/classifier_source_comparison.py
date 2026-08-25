from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...evaluation.detected_roi_dataset import crop_tensor
from ...pipeline.ports import Detection
from ...runtime.onnx import (
    apply_classifier_background_masks,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from ...training.data import read_manifest
from ...training.models import build_dino_classifier, require_torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prototype_scores(
    support_features: np.ndarray,
    support_targets: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    support = np.asarray(support_features, dtype=np.float64)
    evaluation = np.asarray(evaluation_features, dtype=np.float64)
    targets = np.asarray(support_targets, dtype=np.int64)
    support /= np.linalg.norm(support, axis=1, keepdims=True).clip(min=1e-12)
    evaluation /= np.linalg.norm(evaluation, axis=1, keepdims=True).clip(min=1e-12)
    prototypes = np.stack(
        [support[targets == class_id].mean(axis=0) for class_id in range(class_count)]
    )
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
    return evaluation @ prototypes.T


def ranking_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, axis=1, kind="stable")
    labels = np.asarray(targets, dtype=np.int64)
    top1_errors = int(np.count_nonzero(order[:, 0] != labels))
    top3_misses = int(np.count_nonzero(~np.any(order[:, :3] == labels[:, None], axis=1)))
    return {
        "sample_count": len(labels),
        "top1_error_count": top1_errors,
        "top1_accuracy": 1.0 - top1_errors / len(labels),
        "top3_miss_count": top3_misses,
        "top3_accuracy": 1.0 - top3_misses / len(labels),
    }


def select_source(results: list[dict[str, Any]]) -> str:
    if {row["source"] for row in results} != {"single_objects", "single_objects_3"}:
        raise ValueError("source comparison requires exactly single_objects and single_objects_3")
    return min(
        results,
        key=lambda row: (
            int(row["metrics"]["top1_error_count"]),
            int(row["metrics"]["top3_miss_count"]),
            row["source"],
        ),
    )["source"]


def _extract(model, tensors: list[np.ndarray], *, torch, device, batch_size: int) -> np.ndarray:
    parts = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.asarray(tensors[start : start + batch_size], dtype=np.float32)
            ).to(device)
            parts.append(model.extract_features(batch).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def _support_features(
    model,
    rows: list[dict[str, Any]],
    dataset_root: Path,
    *,
    torch,
    device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = []
    for row in rows:
        with Image.open(dataset_root / str(row["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensors.append(prepare_rgb(image, (224, 224), MEAN, STD, reducing_gap=1.0))
    return (
        _extract(model, tensors, torch=torch, device=device, batch_size=batch_size),
        np.asarray([int(row["category_id"]) - 1 for row in rows], dtype=np.int64),
    )


def _evaluation_features(
    model,
    rows: list[dict[str, Any]],
    dataset_root: Path,
    *,
    torch,
    device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts = []
    targets = []
    for row_index, row in enumerate(rows):
        detections = [
            Detection(x, y, x + width, y + height, 1.0)
            for x, y, width, height in (
                annotation["bbox_xywh"] for annotation in row["annotations"]
            )
        ]
        if not detections:
            continue
        with Image.open(dataset_root / str(row["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensors = []
            for index, detection in enumerate(detections):
                tensor = crop_tensor(
                    image,
                    detection,
                    crop_margin_ratio=0.05,
                    input_size=224,
                )
                mask = classifier_neighbor_ownership_mask(
                    detections,
                    index,
                    image_width=image.width,
                    image_height=image.height,
                    output_size=224,
                    margin_ratio=0.05,
                    distance_bias=0.04,
                    shared_scale=False,
                )
                tensors.append(apply_classifier_background_masks(tensor[None], mask[None])[0])
        feature_parts.append(
            _extract(model, tensors, torch=torch, device=device, batch_size=batch_size)
        )
        targets.extend(int(annotation["category_id"]) - 1 for annotation in row["annotations"])
        if (row_index + 1) % 25 == 0:
            print(json.dumps({"evaluated_images": row_index + 1}), flush=True)
    return np.concatenate(feature_parts), np.asarray(targets, dtype=np.int64)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    source_manifests = {
        "single_objects": args.single_objects_manifest,
        "single_objects_3": args.single_objects_3_manifest,
    }
    source_rows = {name: read_manifest(path) for name, path in source_manifests.items()}
    for name, rows in source_rows.items():
        roots = {Path(str(row["image_path"])).parts[0] for row in rows}
        if roots != {name}:
            raise ValueError(f"{name} comparison manifest mixes sources: {sorted(roots)}")
    evaluation_rows = read_manifest(args.evaluation_manifest)
    source_sha256 = {str(row["image_sha256"]) for rows in source_rows.values() for row in rows}
    evaluation_sha256 = {str(row["image_sha256"]) for row in evaluation_rows}
    overlap = source_sha256 & evaluation_sha256
    if overlap:
        raise ValueError("classifier source and multi-object evaluation images overlap")

    torch = require_torch()
    device = torch.device("cpu" if args.cpu else "cuda")
    model = (
        build_dino_classifier(
            "dinov3_convnext_tiny",
            20,
            weights_path=args.weights,
        )
        .to(device)
        .eval()
    )
    evaluation_features, targets = _evaluation_features(
        model,
        evaluation_rows,
        args.dataset_root,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
    )
    results = []
    score_payload = {"targets": targets, "evaluation_features": evaluation_features}
    for source, rows in source_rows.items():
        support, support_targets = _support_features(
            model,
            rows,
            args.dataset_root,
            torch=torch,
            device=device,
            batch_size=args.batch_size,
        )
        scores = prototype_scores(
            support,
            support_targets,
            evaluation_features,
            class_count=20,
        )
        score_payload[f"scores_{source}"] = scores.astype(np.float32)
        results.append(
            {
                "source": source,
                "support_image_count": len(rows),
                "metrics": ranking_metrics(scores, targets),
            }
        )
    selected = select_source(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **score_payload)
    report = {
        "schema_version": "1.0",
        "comparison": "frozen-dinov3-convnext-tiny-cosine-prototype",
        "identical_conditions": True,
        "mixed_classifier_sources": False,
        "evaluation_image_count": len(evaluation_rows),
        "evaluation_object_count": len(targets),
        "source_evaluation_exact_sha256_overlap_count": len(overlap),
        "multi_object_product_labels_used_for_training": False,
        "multi_object_product_labels_used_for_calibration": False,
        "multi_object_product_labels_used_for_comparative_metrics": True,
        "threshold_selected": False,
        "results": results,
        "selected_source": selected,
        "independent_test_claimed": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the two allowed Classifier sources")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--single-objects-manifest", type=Path, required=True)
    parser.add_argument("--single-objects-3-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    compare(parser.parse_args(argv))


if __name__ == "__main__":
    main()
