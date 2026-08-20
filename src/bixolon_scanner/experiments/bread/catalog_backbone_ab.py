from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...contracts.catalog import sha256_file
from ...contracts.runtime_package_v2 import load_runtime_package_v2
from ...operations.catalog_activation import fit_ridge_adapter
from ...pipeline.ports import Detection
from ...runtime.catalog import l2_normalize
from ...runtime.imaging import decode_image, image_original_size
from ...runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from ...training.models import build_dino_classifier, require_torch, set_frozen_backbone
from ...training.synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    direct_roi_recipe_sha256,
    prepare_direct_roi_source,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKBONES = ("dinov3_convnext_tiny", "dinov3_vitb16")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _latency(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "sample_count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(values),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _normalized_margin(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-logits, axis=1, kind="stable")
    sorted_logits = np.take_along_axis(logits, order, axis=1)
    margin = (sorted_logits[:, 0] - sorted_logits[:, 1]) / np.linalg.norm(logits, axis=1).clip(
        min=1e-12
    )
    return order, margin


def _inverse_entropy(logits: np.ndarray) -> np.ndarray:
    order = np.argsort(-logits, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.arange(logits.shape[1], dtype=order.dtype)[None],
        axis=1,
    )
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    ranking_logits = 1.0 / (ranks + 1.0) + probabilities * 1e-3
    ranking_logits -= ranking_logits.max(axis=1, keepdims=True)
    ranking_scores = np.exp(ranking_logits)
    ranking_scores /= ranking_scores.sum(axis=1, keepdims=True)
    return np.sum(ranking_scores * np.log(ranking_scores.clip(1e-12)), axis=1)


def _metrics(logits: np.ndarray, targets: np.ndarray, *, allowed_errors: int) -> dict[str, Any]:
    order, margin = _normalized_margin(logits)
    correct = order[:, 0] == targets
    top3 = np.any(order[:, :3] == targets[:, None], axis=1)
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


def _select_top3_safety_threshold(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    approval_threshold: float,
) -> float | None:
    order, margin = _normalized_margin(logits)
    top3 = np.any(order[:, :3] == targets[:, None], axis=1)
    scores = _inverse_entropy(logits)
    misses = scores[(margin < approval_threshold) & ~top3]
    if not len(misses):
        return None
    return float(
        np.nextafter(
            np.float32(misses.max()),
            np.float32(np.inf),
            dtype=np.float32,
        )
    )


def _policy_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    approval_threshold: float,
    top3_safety_threshold: float | None,
    all_ground_truth_count: int,
) -> dict[str, Any]:
    order, margin = _normalized_margin(logits)
    correct = order[:, 0] == targets
    top3 = np.any(order[:, :3] == targets[:, None], axis=1)
    approved = margin >= approval_threshold
    nonapproved = ~approved
    unsafe = np.zeros(len(targets), dtype=bool)
    if top3_safety_threshold is not None:
        unsafe = nonapproved & (_inverse_entropy(logits) < top3_safety_threshold)
    unknown = nonapproved & ~unsafe
    segment_recapture = nonapproved & unsafe
    approved_errors = approved & ~correct
    candidate_out = unknown & ~top3
    return {
        "matched_count": len(targets),
        "all_ground_truth_count": all_ground_truth_count,
        "approved_count": int(np.count_nonzero(approved)),
        "approved_rate_over_segmentation": float(np.mean(approved)),
        "approved_rate_over_all_ground_truth": float(
            np.count_nonzero(approved) / all_ground_truth_count
        ),
        "approved_error_count": int(np.count_nonzero(approved_errors)),
        "approved_misrecognition_rate_over_all_ground_truth": float(
            np.count_nonzero(approved_errors) / all_ground_truth_count
        ),
        "unknown_count": int(np.count_nonzero(unknown)),
        "unknown_rate_over_segmentation": float(np.mean(unknown)),
        "unknown_top3_candidate_out_count": int(np.count_nonzero(candidate_out)),
        "unknown_top3_candidate_out_rate_over_all_ground_truth": float(
            np.count_nonzero(candidate_out) / all_ground_truth_count
        ),
        "segment_recapture_count": int(np.count_nonzero(segment_recapture)),
        "segment_recapture_rate_over_segmentation": float(np.mean(segment_recapture)),
        "top1_accuracy": float(np.mean(correct)),
        "top3_accuracy": float(np.mean(top3)),
    }


def _recipe(args: argparse.Namespace) -> DirectRoiRecipe:
    return DirectRoiRecipe(
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


class _Extractor:
    def __init__(self, args: argparse.Namespace):
        torch = require_torch()
        self.torch = torch
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        self.model = build_dino_classifier(
            args.backbone,
            20,
            weights_path=args.weights,
            hub_repository=args.hub_repository,
        )
        set_frozen_backbone(self.model)
        self.model.to(self.device).eval()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def extract(self, tensors: list[np.ndarray]) -> tuple[np.ndarray, float]:
        batch = np.stack(tensors).astype(np.float32, copy=False)
        self._sync()
        started = time.perf_counter()
        with self.torch.inference_mode():
            values = self.model.extract_features(self.torch.from_numpy(batch).to(self.device))
        self._sync()
        elapsed = (time.perf_counter() - started) * 1000.0
        return values.float().cpu().numpy().astype(np.float32), elapsed


def _support_features(
    args: argparse.Namespace,
    extractor: _Extractor,
    records: list[dict[str, Any]],
    class_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recipe = _recipe(args)
    parts: list[np.ndarray] = []
    labels: list[int] = []
    view_counts: list[int] = []
    tensors: list[np.ndarray] = []

    def flush() -> None:
        if tensors:
            values, _ = extractor.extract(tensors)
            parts.append(values)
            tensors.clear()

    for record_index, row in enumerate(records):
        with Image.open(args.dataset_root / row["image_path"]) as source:
            image = source.convert("RGB")
        try:
            samples: list[tuple[Image.Image, int]] = [(image, 0)]
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
                samples.append((sample.image, view))
            for sample_image, view in samples:
                tensors.append(prepare_rgb(sample_image, (224, 224), MEAN, STD, reducing_gap=1.0))
                labels.append(class_index[str(row["class_id"])])
                view_counts.append(view)
                if sample_image is not image:
                    sample_image.close()
                if len(tensors) >= args.batch_size:
                    flush()
        finally:
            image.close()
    flush()
    return (
        l2_normalize(np.concatenate(parts)),
        np.asarray(labels, dtype=np.int64),
        np.asarray(view_counts, dtype=np.int64),
    )


def _query_tensors(
    image: Image.Image,
    detections: list[Detection],
    *,
    margin_ratio: float,
    crop_mode: str,
    neighbor_mask: bool,
    distance_bias: float,
    shared_scale: bool,
) -> list[np.ndarray]:
    original_width, original_height = image_original_size(image)
    scale_x = image.width / original_width
    scale_y = image.height / original_height
    tensors = []
    for detection in detections:
        box = classifier_crop_box(
            detection,
            original_width,
            original_height,
            margin_ratio=margin_ratio,
            crop_mode=crop_mode,
        )
        scaled_box = (
            int(np.floor(box[0] * scale_x)),
            int(np.floor(box[1] * scale_y)),
            int(np.ceil(box[2] * scale_x)),
            int(np.ceil(box[3] * scale_y)),
        )
        tensors.append(prepare_rgb(image.crop(scaled_box), (224, 224), MEAN, STD, reducing_gap=1.0))
    if neighbor_mask:
        masks = np.stack(
            [
                classifier_neighbor_ownership_mask(
                    detections,
                    index,
                    image_width=original_width,
                    image_height=original_height,
                    output_size=224,
                    margin_ratio=margin_ratio,
                    distance_bias=distance_bias,
                    shared_scale=shared_scale,
                )
                for index in range(len(detections))
            ]
        )
        return list(apply_classifier_background_masks(np.stack(tensors), masks))
    return tensors


def _query_features(
    args: argparse.Namespace,
    extractor: _Extractor,
    runtime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    records = {int(row["image_id"]): row for row in _jsonl(args.detector_manifest)}
    feature_parts = []
    targets = []
    difficulties = []
    image_ids = []
    model_latencies: list[float] = []
    path_latencies: list[float] = []
    image_counts: Counter[str] = Counter()
    image_status_counts: dict[str, Counter[str]] = {}
    warmed = False
    metadata = runtime.metadata.embedder
    for trace in _jsonl(args.trace):
        record = records[int(trace["image_id"])]
        difficulty = str(record["difficulty"]).upper()
        image_counts[difficulty] += 1
        image_status_counts.setdefault(difficulty, Counter())[str(trace["status"])] += 1
        if trace["status"] != "SEGMENTATION":
            continue
        detections = [
            Detection(
                float(row["bbox"]["x"]),
                float(row["bbox"]["y"]),
                float(row["bbox"]["x"] + row["bbox"]["width"]),
                float(row["bbox"]["y"] + row["bbox"]["height"]),
                1.0,
            )
            for row in trace["decision"]["segmentations"]
        ]
        image = decode_image(
            (args.dataset_root / record["image_path"]).read_bytes(),
            max_bytes=50_000_000,
            max_pixels=50_000_000,
            jpeg_draft_size=runtime.metadata.input.jpeg_draft_size,
        )
        try:
            started = time.perf_counter()
            tensors = _query_tensors(
                image,
                detections,
                margin_ratio=metadata.crop_margin_ratio,
                crop_mode=metadata.crop_mode,
                neighbor_mask=metadata.neighbor_mask,
                distance_bias=metadata.neighbor_distance_bias,
                shared_scale=metadata.neighbor_shared_scale,
            )
            if not warmed:
                for _ in range(args.warmup_count):
                    extractor.extract(tensors)
                warmed = True
                started = time.perf_counter()
            values, model_ms = extractor.extract(tensors)
            path_ms = (time.perf_counter() - started) * 1000.0
        finally:
            image.close()
        model_latencies.append(model_ms)
        path_latencies.append(path_ms)
        for diagnostic in trace["matched_classifier_diagnostics"]:
            detection_index = int(diagnostic["detection_index"])
            feature_parts.append(values[detection_index])
            targets.append(str(diagnostic["target_class_id"]))
            difficulties.append(difficulty)
            image_ids.append(int(trace["image_id"]))
    status_summary = {
        difficulty: {
            "image_count": image_counts[difficulty],
            "segmentation_image_count": image_status_counts[difficulty]["SEGMENTATION"],
            "image_recapture_count": image_status_counts[difficulty]["IMAGE_RECAPTURE"],
        }
        for difficulty in sorted(image_counts)
    }
    return (
        l2_normalize(np.stack(feature_parts)),
        np.asarray(targets),
        np.asarray(difficulties),
        np.asarray(image_ids, dtype=np.int64),
        {
            "model_only_per_segmentation_image": _latency(model_latencies),
            "roi_preprocess_and_model_per_segmentation_image": _latency(path_latencies),
            "image_status_by_difficulty": status_summary,
        },
    )


def _cache_identity(args: argparse.Namespace, recipe: DirectRoiRecipe) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "backbone": args.backbone,
        "weights_sha256": sha256_file(args.weights),
        "hub_repository": args.hub_repository,
        "support_manifest_sha256": sha256_file(args.support_manifest),
        "detector_manifest_sha256": sha256_file(args.detector_manifest),
        "trace_sha256": sha256_file(args.trace),
        "recipe_sha256": direct_roi_recipe_sha256(recipe),
        "seed": args.seed,
        "views_per_source": args.views_per_source,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recipe = _recipe(args)
    identity = _cache_identity(args, recipe)
    cache_path = args.output_dir / "features.npz"
    cache_metadata_path = args.output_dir / "features.json"
    records = sorted(
        _jsonl(args.support_manifest),
        key=lambda row: (str(row["class_id"]), str(row["image_sha256"])),
    )
    class_ids = sorted({str(row["class_id"]) for row in records})
    class_index = {class_id: index for index, class_id in enumerate(class_ids)}
    if len(records) != 200 or len(class_ids) != 20:
        raise ValueError("backbone A/B requires exactly 20 classes x 10 support images")
    runtime = load_runtime_package_v2(args.runtime)
    extraction = None
    if cache_path.is_file() and cache_metadata_path.is_file():
        extraction = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if extraction.get("identity") != identity:
            raise ValueError("backbone A/B feature cache identity differs")
        if extraction.get("feature_cache_sha256") != sha256_file(cache_path):
            raise ValueError("backbone A/B feature cache checksum mismatch")
        with np.load(cache_path, allow_pickle=False) as cache:
            support_features = cache["support_features"].copy()
            support_labels = cache["support_labels"].copy()
            view_counts = cache["view_counts"].copy()
            query_features = cache["query_features"].copy()
            query_target_ids = cache["query_target_ids"].copy()
            difficulties = cache["difficulties"].copy()
            image_ids = cache["image_ids"].copy()
    else:
        extractor = _Extractor(args)
        support_features, support_labels, view_counts = _support_features(
            args, extractor, records, class_index
        )
        query_features, query_target_ids, difficulties, image_ids, performance = _query_features(
            args, extractor, runtime
        )
        np.savez_compressed(
            cache_path,
            support_features=support_features,
            support_labels=support_labels,
            view_counts=view_counts,
            query_features=query_features,
            query_target_ids=query_target_ids,
            difficulties=difficulties,
            image_ids=image_ids,
        )
        extraction = {
            "identity": identity,
            "feature_cache_sha256": sha256_file(cache_path),
            "support_feature_shape": list(support_features.shape),
            "query_feature_shape": list(query_features.shape),
            "performance": performance,
        }
        cache_metadata_path.write_text(
            json.dumps(extraction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    targets = np.asarray([class_index[str(value)] for value in query_target_ids], dtype=np.int64)
    allowed_errors = int(np.floor(args.maximum_error_rate * args.ground_truth_count))
    candidates = []
    fitted: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    for maximum_view in range(args.views_per_source + 1):
        selected_views = view_counts <= maximum_view
        for alpha in args.alphas:
            weight, bias = fit_ridge_adapter(
                support_features[selected_views],
                support_labels[selected_views],
                alpha=alpha,
                class_count=len(class_ids),
            )
            fitted[(maximum_view, alpha)] = (weight, bias)
            candidates.append(
                {
                    "derived_views_per_source": maximum_view,
                    "training_feature_count": int(np.count_nonzero(selected_views)),
                    "alpha": alpha,
                    **_metrics(
                        query_features @ weight + bias,
                        targets,
                        allowed_errors=allowed_errors,
                    ),
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
    weight, bias = fitted[(selected["derived_views_per_source"], selected["alpha"])]
    logits = query_features @ weight + bias
    approval_threshold = float(selected["approval_threshold"])
    top3_safety_threshold = _select_top3_safety_threshold(
        logits,
        targets,
        approval_threshold=approval_threshold,
    )
    overall_policy = _policy_metrics(
        logits,
        targets,
        approval_threshold=approval_threshold,
        top3_safety_threshold=top3_safety_threshold,
        all_ground_truth_count=args.ground_truth_count,
    )
    by_difficulty = {}
    detector_records = _jsonl(args.detector_manifest)
    for difficulty in sorted(set(str(value) for value in difficulties)):
        mask = difficulties == difficulty
        difficulty_gt = sum(
            len(row["annotations"])
            for row in detector_records
            if str(row["difficulty"]).upper() == difficulty
        )
        by_difficulty[difficulty] = _policy_metrics(
            logits[mask],
            targets[mask],
            approval_threshold=approval_threshold,
            top3_safety_threshold=top3_safety_threshold,
            all_ground_truth_count=difficulty_gt,
        )
    report = {
        "schema_version": "2.0",
        "candidate_id": f"{args.backbone}-catalog-10shot-deterministic-derived-view-ridge",
        "evidence_role": "development_backbone_ab_probe",
        "promotion_evidence": False,
        "comparison_contract": {
            "original_support_count": len(records),
            "shots_per_class": len(records) // len(class_ids),
            "development_roi_pixels_used_for_fitting": False,
            "query_features_used_for_candidate_selection_only": True,
            "query_count": len(targets),
            "all_ground_truth_count": args.ground_truth_count,
            "allowed_approved_error_count": allowed_errors,
            "recipe_sha256": direct_roi_recipe_sha256(recipe),
            "seed": args.seed,
        },
        "source": identity,
        "selected": selected,
        "selected_policy": {
            "approval_threshold": approval_threshold,
            "top3_safety_threshold": top3_safety_threshold,
            "overall": overall_policy,
            "by_difficulty": by_difficulty,
        },
        "extraction": extraction,
        "candidates": candidates,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "selected_policy": report["selected_policy"],
                "performance": extraction.get("performance"),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare a frozen DINOv3 Catalog backbone")
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--hub-repository", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-count", type=int, default=20)
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
