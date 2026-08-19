from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ....runtime.onnx import prepare_rgb
from ....training.models import build_dino_classifier, require_torch


def lower_tail_threshold(values: np.ndarray, maximum_false_rate: float) -> float:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("calibration values must be a non-empty vector")
    maximum_false_count = math.floor(len(values) * maximum_false_rate + 1e-12)
    ordered = np.sort(values.astype(np.float64))
    if maximum_false_count == 0:
        return float(np.nextafter(ordered[0], -np.inf))
    threshold = float(ordered[maximum_false_count - 1])
    if int(np.count_nonzero(values <= threshold)) > maximum_false_count:
        threshold = float(np.nextafter(threshold, -np.inf))
    return threshold


def _records(
    dataset_root: Path, challenge_annotation: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = []
    for annotation_name in ("multi_object_instances.json", challenge_annotation):
        annotation_path = dataset_root / "annotations" / annotation_name
        payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        rows = []
        for image in sorted(payload["images"], key=lambda row: int(row["id"])):
            path = (annotation_path.parent / image["file_name"]).resolve()
            path.relative_to(dataset_root)
            rows.append(
                {
                    "image_id": int(image["id"]),
                    "path": path,
                    "status": str(image.get("status", "ANNOTATED")),
                    "reason_codes": list(image.get("reason_codes", [])),
                }
            )
        groups.append(rows)
    return groups[0], groups[1]


def _center_view(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _batches(rows: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _embed(
    records: list[dict[str, Any]],
    *,
    scales: list[float],
    model: Any,
    torch: Any,
    device: Any,
    batch_size: int,
) -> dict[float, np.ndarray]:
    results: dict[float, list[np.ndarray]] = {scale: [] for scale in scales}
    for batch in _batches(records, batch_size):
        images = []
        for record in batch:
            with Image.open(record["path"]) as source:
                images.append(ImageOps.exif_transpose(source).convert("RGB").copy())
        for scale in scales:
            tensors = np.stack(
                [
                    prepare_rgb(
                        _center_view(image, scale),
                        (224, 224),
                        (0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225),
                        reducing_gap=1.0,
                    )
                    for image in images
                ]
            )
            with torch.inference_mode():
                features = model.extract_features(
                    torch.from_numpy(tensors).to(device, dtype=torch.float32)
                )
                features = torch.nn.functional.normalize(features, p=2.0, dim=1)
            results[scale].append(features.cpu().numpy().astype(np.float32))
    return {scale: np.concatenate(parts) for scale, parts in results.items()}


def quality_features(
    embeddings: dict[float, np.ndarray], support_features: np.ndarray, support_labels: np.ndarray
) -> dict[str, np.ndarray]:
    support = support_features.astype(np.float64)
    support /= np.maximum(np.linalg.norm(support, axis=1, keepdims=True), 1e-12)
    prototypes = np.stack([support[support_labels == label].mean(axis=0) for label in range(20)])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    original = embeddings[1.0].astype(np.float64)
    support_scores = original @ support.T
    prototype_scores = original @ prototypes.T
    ordered_prototypes = np.sort(prototype_scores, axis=1)
    features = {
        "maximum_support_cosine": support_scores.max(axis=1),
        "mean_top3_support_cosine": np.sort(support_scores, axis=1)[:, -3:].mean(axis=1),
        "maximum_prototype_cosine": prototype_scores.max(axis=1),
        "prototype_margin": ordered_prototypes[:, -1] - ordered_prototypes[:, -2],
    }
    original_class = prototype_scores.argmax(axis=1)
    agreements = []
    for scale, values in sorted(embeddings.items()):
        if scale == 1.0:
            continue
        scaled = values.astype(np.float64)
        features[f"crop_consistency_{scale:.2f}"] = np.sum(original * scaled, axis=1)
        agreements.append((scaled @ prototypes.T).argmax(axis=1) == original_class)
    if agreements:
        features["crop_class_agreement"] = np.stack(agreements).mean(axis=0)
    return features


def _flag_metrics(flags: np.ndarray, records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = np.asarray([row["status"] == "RECAPTURE" for row in records])
    recapture_total = int(expected.sum())
    normal_total = int((~expected).sum())
    true_positive = int(np.count_nonzero(flags & expected))
    false_positive = int(np.count_nonzero(flags & ~expected))
    reasons = sorted({reason for row in records for reason in row["reason_codes"]})
    by_reason = {}
    for reason in reasons:
        mask = np.asarray([reason in row["reason_codes"] for row in records])
        total = int(mask.sum())
        caught = int(np.count_nonzero(flags & mask))
        by_reason[reason] = {
            "caught": caught,
            "total": total,
            "recall": caught / total if total else None,
        }
    return {
        "true_recapture_count": true_positive,
        "recapture_sample_count": recapture_total,
        "recapture_recall": true_positive / recapture_total if recapture_total else None,
        "false_recapture_count": false_positive,
        "normal_sample_count": normal_total,
        "false_recapture_rate": false_positive / normal_total if normal_total else None,
        "by_reason": by_reason,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.resolve()
    development_records, scan_records = _records(root, args.challenge_annotation)
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
    scales = [1.0, 0.85, 0.7]
    development_embeddings = _embed(
        development_records,
        scales=scales,
        model=model,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
    )
    scan_embeddings = _embed(
        scan_records,
        scales=scales,
        model=model,
        torch=torch,
        device=device,
        batch_size=args.batch_size,
    )
    cache = np.load(args.training_features)
    development = quality_features(
        development_embeddings, cache["support_features"], cache["support_labels"]
    )
    scan = quality_features(scan_embeddings, cache["support_features"], cache["support_labels"])
    feature_results = []
    development_anomalies = []
    scan_anomalies = []
    for name in sorted(development):
        threshold = lower_tail_threshold(development[name], args.maximum_false_recapture_rate)
        development_flags = development[name] <= threshold
        scan_flags = scan[name] <= threshold
        median = float(np.median(development[name]))
        mad = float(np.median(np.abs(development[name] - median)))
        if mad > 1e-9:
            scale = mad * 1.4826
            development_anomalies.append((median - development[name]) / scale)
            scan_anomalies.append((median - scan[name]) / scale)
        feature_results.append(
            {
                "feature": name,
                "threshold": threshold,
                "development_false_recapture_count": int(development_flags.sum()),
                "development_false_recapture_rate": float(development_flags.mean()),
                "scan": _flag_metrics(scan_flags, scan_records),
            }
        )
    if not development_anomalies:
        raise ValueError("quality features have no continuous development variation")
    development_aggregate = np.max(np.stack(development_anomalies), axis=0)
    scan_aggregate = np.max(np.stack(scan_anomalies), axis=0)
    aggregate_threshold = -lower_tail_threshold(
        -development_aggregate, args.maximum_false_recapture_rate
    )
    aggregate_development_flags = development_aggregate >= aggregate_threshold
    aggregate_scan_flags = scan_aggregate >= aggregate_threshold
    report = {
        "evaluation": "pretrained_quality_anomaly_probe",
        "promotion_status": "diagnostic_only",
        "model": "DINOv3 ConvNeXt Tiny frozen full-image features",
        "training_original_count": int(len(cache["support_labels"])),
        "development_normal_image_count": len(development_records),
        "scan_image_count": len(scan_records),
        "selection_policy": "thresholds calibrated on normal multi-object development only",
        "maximum_false_recapture_rate": args.maximum_false_recapture_rate,
        "features": feature_results,
        "aggregate_robust_outlier": {
            "threshold": aggregate_threshold,
            "development_false_recapture_count": int(aggregate_development_flags.sum()),
            "development_false_recapture_rate": float(aggregate_development_flags.mean()),
            "scan": _flag_metrics(aggregate_scan_flags, scan_records),
        },
        "limitations": [
            "The multi-object development set is used only for threshold selection, not fitting weights.",
            "The challenge set is evaluated once by the probe and is not an independent locked test.",
            "This probe has no ONNX export, provider parity or Worker latency evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe pretrained full-image quality signals")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--challenge-annotation", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hub-repository",
        default="facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751",
    )
    parser.add_argument("--maximum-false-recapture-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
