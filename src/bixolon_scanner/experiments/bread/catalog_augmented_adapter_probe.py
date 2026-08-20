from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...operations.catalog_activation import fit_ridge_adapter
from ...runtime.catalog import OnnxEmbedder, l2_normalize
from ...training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    direct_roi_recipe_sha256,
    prepare_direct_roi_source,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _embed_batches(
    embedder: OnnxEmbedder, images: list[Image.Image], *, batch_size: int
) -> np.ndarray:
    parts = []
    for start in range(0, len(images), batch_size):
        parts.append(embedder.embed_images_raw(images[start : start + batch_size]))
    return l2_normalize(np.concatenate(parts))


def _metrics(logits: np.ndarray, targets: np.ndarray, *, allowed_errors: int) -> dict:
    order = np.argsort(-logits, axis=1, kind="stable")
    correct = order[:, 0] == targets
    top3 = np.any(order[:, :3] == targets[:, None], axis=1)
    sorted_logits = np.take_along_axis(logits, order, axis=1)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]) / np.linalg.norm(logits, axis=1).clip(
        min=1e-12
    )
    approved_count = 0
    approved_errors = 0
    threshold = None
    for score in sorted(set(float(value) for value in margin), reverse=True):
        group = margin == score
        next_errors = approved_errors + int(np.count_nonzero(group & ~correct))
        if next_errors > allowed_errors:
            break
        approved_count += int(np.count_nonzero(group))
        approved_errors = next_errors
        threshold = score
    return {
        "sample_count": len(targets),
        "top1_correct_count": int(np.count_nonzero(correct)),
        "top1_accuracy": float(np.mean(correct)),
        "top3_correct_count": int(np.count_nonzero(top3)),
        "top3_accuracy": float(np.mean(top3)),
        "safe_approved_count": approved_count,
        "safe_approved_rate_over_matched": approved_count / len(targets),
        "safe_approved_error_count": approved_errors,
        "approval_threshold": threshold,
    }


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime_package_v2(args.runtime)
    embedder = OnnxEmbedder(runtime, args.provider, args.cuda_dll_dir)
    records = sorted(
        _jsonl(args.support_manifest),
        key=lambda row: (str(row["class_id"]), str(row["image_sha256"])),
    )
    class_ids = sorted({str(row["class_id"]) for row in records})
    class_index = {class_id: index for index, class_id in enumerate(class_ids)}
    recipe = DirectRoiRecipe(
        output_size=224,
        canvas_scale_min=args.canvas_scale_min,
        canvas_scale_max=args.canvas_scale_max,
        rotation_degrees=args.rotation_degrees,
        perspective_fraction=args.perspective_fraction,
        brightness_min=args.brightness_min,
        brightness_max=args.brightness_max,
        contrast_min=args.contrast_min,
        contrast_max=args.contrast_max,
        saturation_min=args.saturation_min,
        saturation_max=args.saturation_max,
        blur_probability=args.blur_probability,
        blur_radius_max=args.blur_radius_max,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        crop_mode=args.crop_mode,
        procedural_gradient=args.procedural_gradient,
        procedural_shadow=args.procedural_shadow,
    )
    feature_path = args.output_dir / "support-features.npz"
    if feature_path.is_file():
        with np.load(feature_path, allow_pickle=False) as cache:
            features = cache["features"].copy()
            labels = cache["labels"].copy()
            view_counts = cache["view_counts"].copy()
    else:
        images = []
        labels_list = []
        view_counts_list = []
        for record_index, row in enumerate(records):
            with Image.open(args.dataset_root / row["image_path"]) as source:
                image = source.convert("RGB")
                images.append(image.copy())
                labels_list.append(class_index[str(row["class_id"])])
                view_counts_list.append(0)
                prepared = prepare_direct_roi_source(image, recipe)
                for view in range(1, args.views_per_source + 1):
                    sample = augment_direct_roi(
                        image,
                        source_sha256=str(row["image_sha256"]),
                        category_id=int(row["category_id"]),
                        seed=args.seed + record_index * 10_000 + view,
                        recipe=recipe,
                        prepared_cutout=prepared,
                    )
                    images.append(sample.image)
                    labels_list.append(class_index[str(row["class_id"])])
                    view_counts_list.append(view)
        try:
            features = _embed_batches(embedder, images, batch_size=args.batch_size)
        finally:
            for image in images:
                image.close()
        labels = np.asarray(labels_list, dtype=np.int64)
        view_counts = np.asarray(view_counts_list, dtype=np.int64)
        np.savez_compressed(feature_path, features=features, labels=labels, view_counts=view_counts)
    with np.load(args.query_features, allow_pickle=False) as cache:
        queries = l2_normalize(cache["features"].copy())
        target_ids = cache["target_ids"].copy()
    targets = np.asarray([class_index[str(value)] for value in target_ids], dtype=np.int64)
    allowed_errors = int(np.floor(args.maximum_error_rate * args.ground_truth_count))
    candidates = []
    for maximum_view in range(0, args.views_per_source + 1):
        selected_views = view_counts <= maximum_view
        for alpha in args.alphas:
            weight, bias = fit_ridge_adapter(
                features[selected_views],
                labels[selected_views],
                alpha=alpha,
                class_count=len(class_ids),
            )
            candidates.append(
                {
                    "derived_views_per_source": maximum_view,
                    "training_feature_count": int(np.count_nonzero(selected_views)),
                    "alpha": alpha,
                    **_metrics(queries @ weight + bias, targets, allowed_errors=allowed_errors),
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row["safe_approved_count"],
            row["top1_correct_count"],
            row["top3_correct_count"],
            -row["derived_views_per_source"],
            -row["alpha"],
        ),
    )
    report = {
        "schema_version": "2.0",
        "candidate_id": "dinov2-catalog-10shot-deterministic-derived-view-ridge",
        "evidence_role": "development_probe",
        "promotion_evidence": False,
        "fitting_contract": {
            "original_support_count": len(records),
            "shots_per_class": len(records) // len(class_ids),
            "development_roi_pixels_used_for_fitting": False,
            "query_features_used_for_candidate_selection_only": True,
            "recipe_sha256": direct_roi_recipe_sha256(recipe),
            "seed": args.seed,
        },
        "selected": selected,
        "candidates": candidates,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selected, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe a derived-view 10-shot Catalog adapter")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--views-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--canvas-scale-min", type=float, default=0.72)
    parser.add_argument("--canvas-scale-max", type=float, default=0.98)
    parser.add_argument("--rotation-degrees", type=float, default=180.0)
    parser.add_argument("--perspective-fraction", type=float, default=0.04)
    parser.add_argument("--brightness-min", type=float, default=0.8)
    parser.add_argument("--brightness-max", type=float, default=1.2)
    parser.add_argument("--contrast-min", type=float, default=0.85)
    parser.add_argument("--contrast-max", type=float, default=1.15)
    parser.add_argument("--saturation-min", type=float, default=0.85)
    parser.add_argument("--saturation-max", type=float, default=1.15)
    parser.add_argument("--blur-probability", type=float, default=0.15)
    parser.add_argument("--blur-radius-max", type=float, default=0.7)
    parser.add_argument("--jpeg-quality-min", type=int, default=82)
    parser.add_argument("--jpeg-quality-max", type=int, default=96)
    parser.add_argument(
        "--crop-mode",
        choices=("white_alpha_composite", "padded_letterbox", "border_connected_composite"),
        default="border_connected_composite",
    )
    parser.add_argument("--procedural-gradient", action="store_true")
    parser.add_argument("--procedural-shadow", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--maximum-error-rate", type=float, default=0.001)
    parser.add_argument("--ground-truth-count", type=int, default=1410)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
