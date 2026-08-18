from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from ...contracts.model_package import load_model_package, sha256_file
from ...evaluation.detector import _metrics, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import classifier_crop_box, prepare_rgb
from ...training.data import read_manifest
from ...training.fewshot_adapter import (
    adapter_spec_from_dict,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
)
from ...training.models import require_torch
from .proposal_classifier_verifier import _load_predictions, verifier_features
from .proposal_ranker import (
    proposal_labels,
    proposal_qualities,
    select_ranked_predictions,
)


def validate_classifier_source(
    checkpoint: dict[str, Any], manifest_metadata: dict[str, Any]
) -> str:
    classifier = manifest_metadata["classifier"]
    source = str(classifier["selected_source"])
    if source not in {"single_objects", "single_objects_2"}:
        raise ValueError("classifier source must be single_objects or single_objects_2")
    if bool(classifier["mixed_sources"]):
        raise ValueError("mixed classifier sources are forbidden")
    if str(checkpoint["dataset_version"]) != str(classifier["source_dataset_version"]):
        raise ValueError("classifier checkpoint dataset version differs from selected source")
    return source


def embedding_features(
    classifier_features: np.ndarray,
    raw_embeddings: np.ndarray,
    adapted_embeddings: np.ndarray,
) -> np.ndarray:
    if raw_embeddings.shape != adapted_embeddings.shape:
        raise ValueError("raw and adapted embedding shapes differ")
    if len(classifier_features) != len(raw_embeddings):
        raise ValueError("classifier features and embeddings are not aligned")
    similarity = np.sum(raw_embeddings * adapted_embeddings, axis=1, keepdims=True)
    displacement = np.linalg.norm(raw_embeddings - adapted_embeddings, axis=1, keepdims=True)
    return np.column_stack(
        (
            classifier_features.astype(np.float32),
            raw_embeddings.astype(np.float32),
            adapted_embeddings.astype(np.float32),
            similarity.astype(np.float32),
            displacement.astype(np.float32),
        )
    )


def _flush_embedding_batch(model, tensors: list[np.ndarray], *, device: str):
    torch = require_torch()
    batch = torch.as_tensor(np.asarray(tensors), dtype=torch.float32, device=device)
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device == "cuda",
        ):
            raw = model.extract_features(batch)
            if isinstance(raw, tuple):
                adapted = model.classifier.adapt(*raw)
                raw = raw[0]
            else:
                adapted = model.classifier.adapt(raw)
    return (
        raw.detach().float().cpu().numpy(),
        adapted.detach().float().cpu().numpy(),
    )


def collect_embeddings(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    dataset_root: Path,
    package,
    checkpoint: dict[str, Any],
    batch_size: int,
    cpu: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --cpu to opt into CPU extraction")
    device = "cpu" if cpu else "cuda"
    model = build_ten_shot_classifier(
        backbone_kind=str(checkpoint["backbone_kind"]),
        weights_path=None,
        hub_repository=f"facebookresearch/dinov3:{checkpoint['source_revision']}",
        spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    model.to(device).eval()

    metadata = package.metadata.classifier
    pending: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    adapted_parts: list[np.ndarray] = []
    counts = []

    def flush() -> None:
        if not pending:
            return
        raw, adapted = _flush_embedding_batch(model, pending, device=device)
        raw_parts.append(raw)
        adapted_parts.append(adapted)
        pending.clear()

    for record, prediction in zip(records, predictions):
        counts.append(len(prediction["scores"]))
        with Image.open(dataset_root / record["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            try:
                for box, score, class_id in zip(
                    prediction["boxes_xyxy"],
                    prediction["scores"],
                    prediction["class_ids"],
                ):
                    detection = Detection(*box, float(score), int(class_id))
                    crop_box = classifier_crop_box(
                        detection,
                        image.width,
                        image.height,
                        margin_ratio=metadata.crop_margin_ratio,
                        crop_mode=metadata.crop_mode,
                    )
                    pending.append(
                        prepare_rgb(
                            image.crop(crop_box),
                            metadata.input_size,
                            metadata.mean,
                            metadata.std,
                            reducing_gap=metadata.resize_reducing_gap,
                        )
                    )
                    if len(pending) >= batch_size:
                        flush()
            finally:
                image.close()
    flush()
    return (
        np.concatenate(raw_parts),
        np.concatenate(adapted_parts),
        np.asarray(counts, dtype=np.int64),
    )


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    zero_error = [
        row
        for row in candidates
        if row["metrics"]["false_positive_count"] == 0
        and row["metrics"]["false_negative_count"] == 0
    ]
    return max(
        zero_error or candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
    ]
    predictions = _load_predictions(args.predictions, records)
    package = load_model_package(args.package)
    torch = require_torch()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest_metadata = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))
    classifier_source = validate_classifier_source(checkpoint, manifest_metadata)

    if args.embedding_cache.is_file() and not args.refresh_cache:
        cache = np.load(args.embedding_cache)
        raw_embeddings = cache["raw_embeddings"].astype(np.float32)
        adapted_embeddings = cache["adapted_embeddings"].astype(np.float32)
        counts = cache["counts"]
    else:
        raw_embeddings, adapted_embeddings, counts = collect_embeddings(
            records,
            predictions,
            dataset_root=args.dataset_root,
            package=package,
            checkpoint=checkpoint,
            batch_size=args.batch_size,
            cpu=args.cpu,
        )
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.embedding_cache,
            raw_embeddings=raw_embeddings.astype(np.float16),
            adapted_embeddings=adapted_embeddings.astype(np.float16),
            counts=counts,
        )
    logits_cache = np.load(args.logits_cache)
    final_logits = logits_cache["logits"]
    ranking_logits = logits_cache["ranking_logits"]
    logits_counts = logits_cache["counts"]
    expected_counts = [len(row["scores"]) for row in predictions]
    if counts.tolist() != expected_counts or logits_counts.tolist() != expected_counts:
        raise ValueError("embedding or logits cache does not match prediction counts")

    feature_parts = []
    label_parts = []
    quality_parts = []
    fold_parts = []
    offset = 0
    for record, prediction, count in zip(records, predictions, counts):
        end = offset + int(count)
        classifier = verifier_features(
            record,
            prediction,
            final_logits[offset:end],
            ranking_logits[offset:end],
        )
        feature_parts.append(
            embedding_features(
                classifier,
                raw_embeddings[offset:end],
                adapted_embeddings[offset:end],
            )
        )
        boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
        label_parts.append(proposal_labels(record, boxes))
        quality_parts.append(proposal_qualities(record, boxes))
        fold_parts.append(np.full(int(count), int(record["fold"]), dtype=np.int64))
        offset = end
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    qualities = np.concatenate(quality_parts)
    candidate_folds = np.concatenate(fold_parts)

    verified_scores = np.zeros(len(labels), dtype=np.float64)
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        training = candidate_folds != held_out_fold
        held_out = candidate_folds == held_out_fold
        if args.target_kind == "binary":
            model = ExtraTreesClassifier(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                class_weight="balanced",
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], labels[training])
            held_out_scores = model.predict_proba(features[held_out])[:, 1]
        else:
            model = ExtraTreesRegressor(
                n_estimators=args.estimators,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                n_jobs=-1,
                random_state=args.seed + held_out_fold,
            )
            model.fit(features[training], qualities[training])
            held_out_scores = model.predict(features[held_out])
        verified_scores[held_out] = np.clip(held_out_scores, 0.0, 1.0)
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_candidate_count": int(training.sum()),
                "training_positive_count": int(labels[training].sum()),
            }
        )

    ranked = []
    offset = 0
    for prediction, count in zip(predictions, counts):
        end = offset + int(count)
        ranked.append({**prediction, "scores": verified_scores[offset:end].tolist()})
        offset = end

    candidates = []
    for score_threshold, nms_threshold, nms_mode in product(
        args.score_thresholds, args.nms_thresholds, args.nms_modes
    ):
        selected_predictions = select_ranked_predictions(
            ranked,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_threshold,
            nms_mode=nms_mode,
        )
        metrics = _metrics(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        candidates.append(
            {
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                "nms_mode": nms_mode,
                "metrics": metrics,
            }
        )
    selected = _select_candidate(candidates)
    selected_predictions = select_ranked_predictions(
        ranked,
        score_threshold=selected["score_threshold"],
        nms_iou_threshold=selected["nms_iou_threshold"],
        nms_mode=selected["nms_mode"],
    )
    full_recall = [row for row in candidates if row["metrics"]["false_negative_count"] == 0]
    zero_false_positive = [row for row in candidates if row["metrics"]["false_positive_count"] == 0]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_proposal_embedding_verifier",
        "folds": sorted(folds),
        "input_candidate_count": len(labels),
        "input_positive_count": int(labels.sum()),
        "model": (
            "ExtraTreesClassifier" if args.target_kind == "binary" else "ExtraTreesRegressor"
        ),
        "target_kind": args.target_kind,
        "embedding": "DINOv3 ConvNeXt-Tiny raw plus residual-adapted",
        "embedding_dimension": int(raw_embeddings.shape[1]),
        "classifier_source": classifier_source,
        "classifier_checkpoint": args.checkpoint.name,
        "classifier_checkpoint_sha256": sha256_file(args.checkpoint),
        "classifier_dataset_version": checkpoint["dataset_version"],
        "feature_count": int(features.shape[1]),
        "fold_diagnostics": fold_diagnostics,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
        "full_recall_selected": (
            min(
                full_recall,
                key=lambda row: (
                    row["metrics"]["false_positive_count"],
                    -row["score_threshold"],
                ),
            )
            if full_recall
            else None
        ),
        "zero_false_positive_selected": (
            min(
                zero_false_positive,
                key=lambda row: (
                    row["metrics"]["false_negative_count"],
                    -row["score_threshold"],
                ),
            )
            if zero_false_positive
            else None
        ),
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in selected_predictions),
            encoding="utf-8",
        )
    if args.ranked_predictions_output:
        args.ranked_predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.ranked_predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in ranked),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit a DINO embedding verifier for detector proposals"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--logits-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--ranked-predictions-output", type=Path)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument(
        "--nms-modes",
        choices=["class_agnostic", "class_aware"],
        nargs="+",
        default=["class_agnostic"],
    )
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.2)
    parser.add_argument("--target-kind", choices=["binary", "iou"], default="binary")
    parser.add_argument("--seed", type=int, default=20260814)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
