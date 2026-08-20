from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...contracts.model_package import load_model_package
from ...pipeline.ports import Detection
from ...runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from ...training.calibration import binomial_rate_upper_bound
from ...training.classifier_allowlist import (
    ClassifierAllowlist,
    audit_classifier_allowlist,
    sha256_file,
    write_allowlist_audit,
)
from ...training.models import build_dino_classifier, require_torch, set_frozen_backbone
from ...training.synthetic_roi import (
    ClutterRoiRecipe,
    DirectRoiRecipe,
    augment_clutter_roi,
    augment_direct_roi,
    clutter_roi_recipe_sha256,
    direct_roi_recipe_sha256,
    prepare_direct_roi_source,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    version = config.get("experiment", {}).get("candidate_version")
    if version not in {"1.1.1", "1.1.2", "1.1.3"}:
        raise ValueError("200-only classifier config must target version 1.1.1 through 1.1.3")
    dataset = config.get("dataset", {})
    if dataset.get("allowed_source") != "single_objects":
        raise ValueError("1.1.1 classifier source must be exactly single_objects")
    if dataset.get("image_count") != 200 or dataset.get("class_count") != 20:
        raise ValueError("1.1.1 classifier source must be exactly 20 classes x 10 images")
    forbidden = set(dataset.get("fitting_forbidden_sources", []))
    required_forbidden = {
        "multi_object_scenes",
        "operational_collections",
        "gt_crops",
        "detector_rois",
    }
    if not required_forbidden <= forbidden:
        raise ValueError("1.1.1 config must forbid every development ROI fitting source")
    training = config.get("training", {})
    if training.get("backbone") != "dinov3_convnext_tiny":
        raise ValueError("1.1.1 baseline requires DINOv3 ConvNeXt-Tiny")
    if training.get("backbone_trainable") is not False:
        raise ValueError("1.1.1 baseline freezes the pretrained backbone")
    expected_head = (
        "nested_small_sample_family" if version == "1.1.3" else "regularized_linear_ridge"
    )
    if training.get("head") != expected_head:
        raise ValueError(f"{version} requires head={expected_head}")
    _head_candidates(config)
    selection = config.get("selection", {})
    if selection.get("outer_folds") != [0, 1, 2] or selection.get("inner_policy") != (
        "next_fold_train_remaining_fold_validate"
    ):
        raise ValueError("1.1.1 baseline requires the fixed three-fold nested policy")
    if selection.get("development_evaluation_used_for_fitting") is not False:
        raise ValueError("E/M/H and operational evaluation cannot be used for fitting")
    expected_mode = (
        "direct_single_object" if version == "1.1.1" else "neighbor_masked_boundary_clutter"
    )
    augmentation_mode = config.get("augmentation_mode", "direct_single_object")
    if augmentation_mode != expected_mode:
        raise ValueError(f"{version} requires augmentation_mode={expected_mode}")
    return config


def _recipe(config: dict[str, Any]) -> DirectRoiRecipe | ClutterRoiRecipe:
    values = config["augmentation"]
    recipe_type = (
        DirectRoiRecipe
        if config.get("augmentation_mode", "direct_single_object") == "direct_single_object"
        else ClutterRoiRecipe
    )
    recipe = recipe_type(
        **{key: values[key] for key in recipe_type.__dataclass_fields__ if key in values}
    )
    recipe.validate()
    return recipe


def _recipe_sha256(recipe: DirectRoiRecipe | ClutterRoiRecipe) -> str:
    if isinstance(recipe, DirectRoiRecipe):
        return direct_roi_recipe_sha256(recipe)
    return clutter_roi_recipe_sha256(recipe)


def _prepare_neighbor_masked_clutter(sample) -> np.ndarray:
    detections = [Detection(*sample.bbox_xyxy, score=1.0)]
    detections.extend(
        Detection(*row["bbox_xyxy"], score=1.0) for row in sample.provenance["distractors"]
    )
    crop_box = classifier_crop_box(
        detections[0],
        sample.image.width,
        sample.image.height,
        margin_ratio=0.05,
        crop_mode="box_resize",
    )
    tensor = prepare_rgb(
        sample.image.crop(crop_box),
        (224, 224),
        MEAN,
        STD,
        reducing_gap=1.0,
    )
    mask = classifier_neighbor_ownership_mask(
        detections,
        0,
        image_width=sample.image.width,
        image_height=sample.image.height,
        output_size=224,
        margin_ratio=0.05,
        distance_bias=0.0,
        shared_scale=False,
    )
    return apply_classifier_background_masks(tensor[None], mask[None])[0]


def _audit(config: dict[str, Any], dataset_root: Path, manifest: Path) -> ClassifierAllowlist:
    dataset = config["dataset"]
    allowlist = audit_classifier_allowlist(
        dataset_root,
        manifest,
        expected_manifest_sha256=str(dataset["manifest_sha256"]),
        allowed_directory=str(dataset["allowed_source"]),
        expected_class_count=int(dataset["class_count"]),
        expected_shots_per_class=int(dataset["shots_per_class"]),
    )
    if allowlist.audit["source_image_set_sha256"] != dataset["source_image_set_sha256"]:
        raise ValueError("classifier allowlist image set checksum does not match the 1.1.1 lock")
    return allowlist


def _build_model(weights: Path):
    model = build_dino_classifier(
        "dinov3_convnext_tiny",
        20,
        weights_path=weights,
        classifier_head_kind="linear",
    )
    set_frozen_backbone(model)
    return model


def _extract_features(model, tensors: list[np.ndarray], *, device, batch_size: int) -> np.ndarray:
    torch = require_torch()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = np.asarray(tensors[start : start + batch_size], dtype=np.float32)
            values = model.extract_features(torch.from_numpy(batch).to(device))
            parts.append(values.float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def _feature_cache_identity(
    *,
    config_path: Path,
    allowlist: ClassifierAllowlist,
    weights: Path,
    recipe: DirectRoiRecipe | ClutterRoiRecipe,
    seed: int,
    train_views: int,
    validation_views: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "policy": "bread_classifier_200_only_1.1.1",
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": allowlist.audit["manifest_sha256"],
        "source_image_set_sha256": allowlist.audit["source_image_set_sha256"],
        "actual_access_image_set_sha256": allowlist.audit["actual_access_image_set_sha256"],
        "backbone_weight_sha256": sha256_file(weights),
        "augmentation_mode": (
            "direct_single_object"
            if isinstance(recipe, DirectRoiRecipe)
            else "neighbor_masked_boundary_clutter"
        ),
        "augmentation_recipe_sha256": _recipe_sha256(recipe),
        "augmentation_seed": seed,
        "train_views_per_source": train_views,
        "validation_views_per_source": validation_views,
    }


def prepare_feature_cache(
    *,
    config_path: Path,
    config: dict[str, Any],
    dataset_root: Path,
    allowlist: ClassifierAllowlist,
    weights: Path,
    output_dir: Path,
    cpu: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    training = config["training"]
    seed = int(config["experiment"]["augmentation_seed"])
    train_views = int(training["train_views_per_source"])
    validation_views = int(training["validation_views_per_source"])
    if train_views < 1 or validation_views < 1:
        raise ValueError("training and validation views must be positive")
    recipe = _recipe(config)
    identity = _feature_cache_identity(
        config_path=config_path,
        allowlist=allowlist,
        weights=weights,
        recipe=recipe,
        seed=seed,
        train_views=train_views,
        validation_views=validation_views,
    )
    cache_dir = output_dir / "cache"
    cache_path = cache_dir / "features.npz"
    metadata_path = cache_dir / "features.json"
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise ValueError("1.1.1 feature cache identity differs from the requested contract")
        if metadata.get("feature_cache_sha256") != sha256_file(cache_path):
            raise ValueError("1.1.1 feature cache checksum mismatch")
        cache = np.load(cache_path)
        return {key: cache[key] for key in cache.files}, metadata

    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model = _build_model(weights).to(device).eval()
    batch_size = int(training["feature_batch_size"])
    train_features: list[np.ndarray] = []
    train_labels: list[int] = []
    train_folds: list[int] = []
    validation_features: list[np.ndarray] = []
    validation_labels: list[int] = []
    validation_folds: list[int] = []
    validation_source_indices: list[int] = []
    dataset_root = dataset_root.resolve()
    images: list[Image.Image] = []
    cutouts: list[Image.Image] = []
    cutout_recipe = DirectRoiRecipe(
        crop_mode="border_connected_composite",
        border_color_distance=42,
        mask_feather_radius=0.8,
    )
    for record in allowlist.records:
        image_path = dataset_root / str(record["image_path"])
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
        images.append(image)
        cutouts.append(
            prepare_direct_roi_source(
                image, recipe if isinstance(recipe, DirectRoiRecipe) else cutout_recipe
            )
        )
    for source_index, record in enumerate(allowlist.records):
        image = images[source_index]
        cutout = cutouts[source_index]
        distractors = [
            (cutouts[index], str(other["image_sha256"]), int(other["category_id"]))
            for index, other in enumerate(allowlist.records)
            if int(other["category_id"]) != int(record["category_id"])
        ]
        train_tensors = [prepare_rgb(image, (224, 224), MEAN, STD, reducing_gap=1.0)]
        for view in range(train_views):
            view_seed = seed + source_index * 1_000_003 + view
            if isinstance(recipe, DirectRoiRecipe):
                sample = augment_direct_roi(
                    image,
                    source_sha256=str(record["image_sha256"]),
                    category_id=int(record["category_id"]),
                    seed=view_seed,
                    recipe=recipe,
                    prepared_cutout=cutout,
                )
                tensor = prepare_rgb(sample.image, (224, 224), MEAN, STD, reducing_gap=1.0)
            else:
                sample = augment_clutter_roi(
                    cutout,
                    target_sha256=str(record["image_sha256"]),
                    target_category_id=int(record["category_id"]),
                    distractors=distractors,
                    seed=view_seed,
                    recipe=recipe,
                )
                tensor = _prepare_neighbor_masked_clutter(sample)
            train_tensors.append(tensor)
            sample.image.close()
        extracted = _extract_features(model, train_tensors, device=device, batch_size=batch_size)
        train_features.extend(extracted)
        train_labels.extend([int(record["category_id"]) - 1] * len(extracted))
        train_folds.extend([int(record["fold"])] * len(extracted))
        validation_tensors = [train_tensors[0]]
        for view in range(validation_views):
            view_seed = seed + 1_000_000_007 + source_index * 1_000_003 + view
            if isinstance(recipe, DirectRoiRecipe):
                sample = augment_direct_roi(
                    image,
                    source_sha256=str(record["image_sha256"]),
                    category_id=int(record["category_id"]),
                    seed=view_seed,
                    recipe=recipe,
                    prepared_cutout=cutout,
                )
                tensor = prepare_rgb(sample.image, (224, 224), MEAN, STD, reducing_gap=1.0)
            else:
                sample = augment_clutter_roi(
                    cutout,
                    target_sha256=str(record["image_sha256"]),
                    target_category_id=int(record["category_id"]),
                    distractors=distractors,
                    seed=view_seed,
                    recipe=recipe,
                )
                tensor = _prepare_neighbor_masked_clutter(sample)
            validation_tensors.append(tensor)
            sample.image.close()
        extracted = _extract_features(
            model, validation_tensors, device=device, batch_size=batch_size
        )
        validation_features.extend(extracted)
        validation_labels.extend([int(record["category_id"]) - 1] * len(extracted))
        validation_folds.extend([int(record["fold"])] * len(extracted))
        validation_source_indices.extend([source_index] * len(extracted))
        if (source_index + 1) % 20 == 0:
            print(json.dumps({"prepared_sources": source_index + 1}), flush=True)
    for image in images:
        image.close()
    for cutout in cutouts:
        cutout.close()
    values = {
        "train_features": np.asarray(train_features, dtype=np.float32),
        "train_labels": np.asarray(train_labels, dtype=np.int64),
        "train_folds": np.asarray(train_folds, dtype=np.int64),
        "validation_features": np.asarray(validation_features, dtype=np.float32),
        "validation_labels": np.asarray(validation_labels, dtype=np.int64),
        "validation_folds": np.asarray(validation_folds, dtype=np.int64),
        "validation_source_indices": np.asarray(validation_source_indices, dtype=np.int64),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **values)
    metadata = {
        "identity": identity,
        "feature_cache_sha256": sha256_file(cache_path),
        "shapes": {key: list(value.shape) for key, value in values.items()},
        "device": str(device),
    }
    _write_json(metadata_path, metadata)
    return values, metadata


def fit_ridge_head(
    features: np.ndarray, labels: np.ndarray, *, alpha: float, class_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    design = np.concatenate([features, np.ones((len(features), 1), dtype=np.float64)], axis=1)
    targets = np.eye(class_count, dtype=np.float64)[labels]
    regularization = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularization[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + regularization,
        design.T @ targets,
    )
    return coefficients[:-1].astype(np.float32), coefficients[-1].astype(np.float32)


def predict_ridge(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float32) @ weight + bias


def _head_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    training = config.get("training", {})
    if training.get("head") == "regularized_linear_ridge":
        ridge = [float(value) for value in training.get("ridge_candidates", [])]
        if not ridge or any(value <= 0 for value in ridge) or len(set(ridge)) != len(ridge):
            raise ValueError("ridge_candidates must contain unique positive values")
        return [{"kind": "regularized_linear_ridge", "alpha": value} for value in ridge]
    candidates = training.get("head_candidates", [])
    if not candidates or not all(isinstance(value, dict) for value in candidates):
        raise ValueError("nested_small_sample_family requires head_candidates")
    normalized = []
    for value in candidates:
        kind = value.get("kind")
        if kind == "regularized_linear_ridge":
            alpha = float(value.get("alpha", 0.0))
            if alpha <= 0:
                raise ValueError("ridge head alpha must be positive")
            normalized.append({"kind": kind, "alpha": alpha})
        elif kind == "cosine_prototype":
            normalized.append({"kind": kind})
        elif kind == "shrinkage_lda":
            shrinkage = float(value.get("shrinkage", -1.0))
            if not 0.0 <= shrinkage <= 1.0:
                raise ValueError("LDA shrinkage must be in [0, 1]")
            normalized.append({"kind": kind, "shrinkage": shrinkage})
        else:
            raise ValueError(f"unsupported 1.1.3 head candidate: {kind}")
    canonical = [json.dumps(value, sort_keys=True) for value in normalized]
    if len(canonical) != len(set(canonical)):
        raise ValueError("head_candidates must be unique")
    return normalized


def fit_small_sample_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    candidate: dict[str, Any],
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    kind = candidate["kind"]
    if kind == "regularized_linear_ridge":
        return fit_ridge_head(
            features,
            labels,
            alpha=float(candidate["alpha"]),
            class_count=class_count,
        )
    if kind == "cosine_prototype":
        normalized = np.asarray(features, dtype=np.float64)
        normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
        prototypes = np.stack(
            [
                normalized[np.asarray(labels) == category].mean(axis=0)
                for category in range(class_count)
            ]
        )
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
        return prototypes.T.astype(np.float32), np.zeros(class_count, dtype=np.float32)
    if kind == "shrinkage_lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        classifier = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage=float(candidate["shrinkage"])
        ).fit(features, labels)
        if classifier.coef_.shape != (class_count, features.shape[1]):
            raise ValueError("shrinkage LDA did not fit every configured class")
        return classifier.coef_.T.astype(np.float32), classifier.intercept_.astype(np.float32)
    raise ValueError(f"unsupported small-sample head: {kind}")


def _classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    ranking = np.argsort(-logits, axis=1, kind="stable")
    top1 = ranking[:, 0]
    top3 = np.any(ranking[:, :3] == targets[:, None], axis=1)
    return {
        "sample_count": len(targets),
        "top1_error_count": int(np.count_nonzero(top1 != targets)),
        "top1_accuracy": float(np.mean(top1 == targets)),
        "top3_miss_count": int(np.count_nonzero(~top3)),
        "top3_accuracy": float(np.mean(top3)),
    }


def _selection_key(
    logits: np.ndarray, targets: np.ndarray, candidate_index: int
) -> tuple[float, ...]:
    metrics = _classification_metrics(logits, targets)
    true_logits = logits[np.arange(len(targets)), targets]
    negatives = logits.copy()
    negatives[np.arange(len(targets)), targets] = -np.inf
    true_margin = true_logits - np.max(negatives, axis=1)
    return (
        float(metrics["top3_miss_count"]),
        float(metrics["top1_error_count"]),
        -float(np.median(true_margin)),
        float(candidate_index),
    )


def nested_oof_fit(
    cache: dict[str, np.ndarray], *, head_candidates: list[dict[str, Any]], class_count: int
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    train_features = cache["train_features"]
    train_labels = cache["train_labels"]
    train_folds = cache["train_folds"]
    validation_features = cache["validation_features"]
    validation_labels = cache["validation_labels"]
    validation_folds = cache["validation_folds"]
    oof_logits = np.empty((len(validation_features), class_count), dtype=np.float32)
    outer_reports = []
    selected_heads = []
    for outer_fold in (0, 1, 2):
        inner_train_fold = (outer_fold + 1) % 3
        inner_validation_fold = (outer_fold + 2) % 3
        candidates = []
        for candidate_index, candidate in enumerate(head_candidates):
            weight, bias = fit_small_sample_head(
                train_features[train_folds == inner_train_fold],
                train_labels[train_folds == inner_train_fold],
                candidate=candidate,
                class_count=class_count,
            )
            mask = validation_folds == inner_validation_fold
            logits = predict_ridge(validation_features[mask], weight, bias)
            candidates.append(
                {
                    "head": candidate,
                    "metrics": _classification_metrics(logits, validation_labels[mask]),
                    "selection_key": list(
                        _selection_key(logits, validation_labels[mask], candidate_index)
                    ),
                }
            )
        selected = min(candidates, key=lambda value: tuple(value["selection_key"]))
        selected_head = dict(selected["head"])
        selected_heads.append(json.dumps(selected_head, sort_keys=True))
        outer_train = train_folds != outer_fold
        weight, bias = fit_small_sample_head(
            train_features[outer_train],
            train_labels[outer_train],
            candidate=selected_head,
            class_count=class_count,
        )
        held_out = validation_folds == outer_fold
        held_out_logits = predict_ridge(validation_features[held_out], weight, bias)
        oof_logits[held_out] = held_out_logits
        outer_reports.append(
            {
                "outer_fold": outer_fold,
                "inner_train_fold": inner_train_fold,
                "inner_validation_fold": inner_validation_fold,
                "candidates": candidates,
                "selected_head": selected_head,
                "outer_metrics": _classification_metrics(
                    held_out_logits, validation_labels[held_out]
                ),
            }
        )
    counts = Counter(selected_heads)
    canonical_candidates = [json.dumps(value, sort_keys=True) for value in head_candidates]
    final_head_text = min(
        canonical_candidates,
        key=lambda value: (-counts[value], canonical_candidates.index(value)),
    )
    return oof_logits, outer_reports, json.loads(final_head_text)


def l2_normalized_logit_margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(logits, dtype=np.float64), axis=1)
    return (
        (ordered[:, -1] - ordered[:, -2]) / np.maximum(np.linalg.norm(logits, axis=1), 1e-12)
    ).astype(np.float32)


def one_view_top3_safety(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    order = np.argsort(-logits, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(logits.shape[1])[None, :]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=1, keepdims=True)
    ranking_logits = 1.0 / (ranks + 1.0) + probability * 1e-3
    ranking_shifted = ranking_logits - np.max(ranking_logits, axis=1, keepdims=True)
    ranking_probability = np.exp(ranking_shifted)
    ranking_probability /= np.sum(ranking_probability, axis=1, keepdims=True)
    return np.sum(
        ranking_probability * np.log(np.maximum(ranking_probability, 1e-12)), axis=1
    ).astype(np.float32)


def select_finite_oof_policy(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    ranking = np.argsort(-logits, axis=1, kind="stable")
    predictions = ranking[:, 0]
    top3_hit = np.any(ranking[:, :3] == targets[:, None], axis=1)
    approval_scores = l2_normalized_logit_margin(logits)
    top1_error = predictions != targets
    if np.any(top1_error):
        approval_threshold = float(np.nextafter(np.max(approval_scores[top1_error]), np.inf))
    else:
        approval_threshold = float(np.nextafter(np.min(approval_scores), -np.inf))
    approved = approval_scores >= approval_threshold
    safety_scores = one_view_top3_safety(logits)
    top3_miss = ~top3_hit
    if np.any(top3_miss):
        safety_threshold = float(np.nextafter(np.max(safety_scores[top3_miss]), np.inf))
    else:
        safety_threshold = float(np.min(safety_scores))
    recapture = (~approved) & (safety_scores < safety_threshold)
    unknown = (~approved) & (~recapture)
    approved_error_count = int(np.count_nonzero(approved & top1_error))
    candidate_out_count = int(np.count_nonzero(unknown & top3_miss))
    return {
        "approval_metric": "l2_normalized_logit_margin",
        "approval_threshold": approval_threshold,
        "top3_safety_metric": "inverse_entropy",
        "top3_safety_threshold": safety_threshold,
        "sample_count": len(targets),
        "approved_count": int(np.count_nonzero(approved)),
        "approved_rate": float(np.mean(approved)),
        "approved_error_count": approved_error_count,
        "approved_error_rate": approved_error_count / len(targets),
        "unknown_count": int(np.count_nonzero(unknown)),
        "unknown_candidate_out_count": candidate_out_count,
        "unknown_candidate_out_rate": candidate_out_count / len(targets),
        "segment_recapture_count": int(np.count_nonzero(recapture)),
        "finite_oof_zero_approved_errors": approved_error_count == 0,
        "finite_oof_zero_unknown_candidate_out": candidate_out_count == 0,
    }


def _export_model(
    *,
    weights_path: Path,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    checkpoint_path: Path,
    onnx_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> dict[str, str]:
    torch = require_torch()
    model = _build_model(weights_path)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.from_numpy(head_weight.T))
        model.classifier.bias.copy_(torch.from_numpy(head_bias))
    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "1.0",
            "architecture": "dinov3_convnext_tiny_frozen_regularized_linear",
            "model_state_dict": copy.deepcopy(model.state_dict()),
            **checkpoint_metadata,
        },
        checkpoint_path,
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_file(onnx_path),
    }


def assemble_development_package(
    *,
    base_package_dir: Path,
    output_package_dir: Path,
    classifier_onnx: Path,
    config: dict[str, Any],
    config_sha256: str,
    allowlist: ClassifierAllowlist,
    policy: dict[str, Any],
    selected_head: dict[str, Any],
    weights_path: Path,
) -> dict[str, Any]:
    load_model_package(base_package_dir)
    metadata = json.loads((base_package_dir / "metadata.json").read_text(encoding="utf-8"))
    output_package_dir.mkdir(parents=True, exist_ok=True)
    classifier_filename = str(metadata["classifier"]["filename"])
    for filename, expected in metadata["checksums"].items():
        if filename == classifier_filename:
            continue
        source = base_package_dir / filename
        if sha256_file(source) != expected:
            raise ValueError(f"base detector package checksum mismatch: {filename}")
        destination = output_package_dir / filename
        if destination.exists() and sha256_file(destination) != expected:
            raise FileExistsError(f"1.1.1 package destination differs: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    classifier_destination = output_package_dir / classifier_filename
    if classifier_destination.exists() and sha256_file(classifier_destination) != sha256_file(
        classifier_onnx
    ):
        raise FileExistsError("1.1.1 classifier package destination differs")
    if not classifier_destination.exists():
        shutil.copy2(classifier_onnx, classifier_destination)
    candidate_version = str(config["experiment"]["candidate_version"])
    metadata["schema_version"] = "2.0"
    metadata["worker_version"] = candidate_version
    metadata["promotion_status"] = "development"
    metadata["dataset_version"] = f"bread-classifier-200-only-{candidate_version}"
    metadata["detector"]["version"] = candidate_version
    metadata["classifier"]["version"] = candidate_version
    metadata["classifier"]["approval_threshold"] = policy["approval_threshold"]
    metadata["classifier"].pop("approval_thresholds", None)
    metadata["classifier"]["temperature"] = 1.0
    neighbor = metadata["classifier"]["neighbor_mask_inference"]
    neighbor["views"] = [
        {"name": "normalized", "distance_bias": 0.0, "weight": 1.0, "shared_scale": False}
    ]
    neighbor["approval_metric"] = "l2_normalized_logit_margin"
    neighbor["ranking_aggregation"] = "weighted_reciprocal_rank"
    neighbor["top3_safety_metric"] = "inverse_entropy"
    neighbor["top3_safety_threshold"] = policy["top3_safety_threshold"]
    metadata["checksums"][classifier_filename] = sha256_file(classifier_destination)
    metadata["sources"]["classifier"] = {
        "architecture": (
            "DINOv3 ConvNeXt-Tiny frozen backbone plus 200-only "
            f"{selected_head['kind']} with "
            f"{config.get('augmentation_mode', 'direct_single_object')}"
        ),
        "revision": str(config["training"]["backbone_revision"]),
        "weight_filename": weights_path.name,
        "weight_sha256": sha256_file(weights_path),
        "training_pipeline_version": candidate_version,
        "training_contract_sha256": config_sha256,
        "training_dataset_version": f"bread-classifier-200-only-{candidate_version}",
        "training_manifest_sha256": allowlist.audit["manifest_sha256"],
    }
    approved_count = int(policy["approved_count"])
    approved_errors = int(policy["approved_error_count"])
    metadata["calibration"] = {
        "sample_count": int(policy["sample_count"]),
        "approved_precision": (1.0 - approved_errors / approved_count if approved_count else 1.0),
        "approval_coverage": float(policy["approved_rate"]),
        "false_approval_rate_upper_95": binomial_rate_upper_bound(approved_errors, approved_count),
        "risk_control_satisfied": bool(
            approved_count > 0
            and binomial_rate_upper_bound(approved_errors, approved_count) <= 0.001
        ),
    }
    metadata.pop("promotion", None)
    _write_json(output_package_dir / "metadata.json", metadata)
    package = load_model_package(output_package_dir)
    return {
        "path": output_package_dir.as_posix(),
        "metadata_sha256": sha256_file(output_package_dir / "metadata.json"),
        "classifier_onnx_sha256": sha256_file(package.classifier_path),
        "worker_version": package.metadata.worker_version,
        "detector_version": package.metadata.detector.version,
        "classifier_version": package.metadata.classifier.version,
        "promotion_status": package.metadata.promotion_status,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    candidate_version = str(config["experiment"]["candidate_version"])
    allowlist = _audit(config, args.dataset_root, args.manifest)
    write_allowlist_audit(args.output_dir / "reports" / "allowlist-audit.json", allowlist)
    weights_sha256 = sha256_file(args.weights)
    if weights_sha256 != config["training"]["backbone_weight_sha256"]:
        raise ValueError("DINOv3 backbone checksum does not match the 1.1.1 config")
    cache, cache_metadata = prepare_feature_cache(
        config_path=config_path,
        config=config,
        dataset_root=args.dataset_root,
        allowlist=allowlist,
        weights=args.weights,
        output_dir=args.output_dir,
        cpu=args.cpu,
    )
    head_candidates = _head_candidates(config)
    oof_logits, nested_report, final_head = nested_oof_fit(
        cache, head_candidates=head_candidates, class_count=20
    )
    targets = cache["validation_labels"]
    policy = select_finite_oof_policy(oof_logits, targets)
    oof_metrics = _classification_metrics(oof_logits, targets)
    final_weight, final_bias = fit_small_sample_head(
        cache["train_features"],
        cache["train_labels"],
        candidate=final_head,
        class_count=20,
    )
    checkpoint_path = args.output_dir / "models" / "classifier.pt"
    onnx_path = args.output_dir / "models" / "classifier.onnx"
    model_hashes = _export_model(
        weights_path=args.weights,
        head_weight=final_weight,
        head_bias=final_bias,
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        checkpoint_metadata={
            "candidate_version": candidate_version,
            "manifest_sha256": allowlist.audit["manifest_sha256"],
            "source_image_set_sha256": allowlist.audit["source_image_set_sha256"],
            "actual_access_image_set_sha256": allowlist.audit["actual_access_image_set_sha256"],
            "config_sha256": sha256_file(config_path),
            "backbone_weight_sha256": weights_sha256,
            "augmentation_seed": int(config["experiment"]["augmentation_seed"]),
            "feature_cache_sha256": cache_metadata["feature_cache_sha256"],
            "selected_head": final_head,
            "policy": policy,
        },
    )
    package_report = assemble_development_package(
        base_package_dir=args.base_package,
        output_package_dir=args.output_package,
        classifier_onnx=onnx_path,
        config=config,
        config_sha256=sha256_file(config_path),
        allowlist=allowlist,
        policy=policy,
        selected_head=final_head,
        weights_path=args.weights,
    )
    report = {
        "schema_version": "1.0",
        "candidate_id": str(config["experiment"]["name"]),
        "lifecycle": "active",
        "hypothesis": config["experiment"]["hypothesis"],
        "fitting_source": {
            "manifest_sha256": allowlist.audit["manifest_sha256"],
            "source_image_set_sha256": allowlist.audit["source_image_set_sha256"],
            "actual_access_image_set_sha256": allowlist.audit["actual_access_image_set_sha256"],
            "source_count": len(allowlist.records),
            "allowlist_exact": True,
            "development_roi_fitting_used": False,
        },
        "augmentation": {
            "seed": int(config["experiment"]["augmentation_seed"]),
            "mode": config.get("augmentation_mode", "direct_single_object"),
            "recipe": asdict(_recipe(config)),
            "recipe_sha256": _recipe_sha256(_recipe(config)),
        },
        "feature_cache": cache_metadata,
        "nested_fold_selection": nested_report,
        "selected_final_head": final_head,
        "internal_oof": {
            "role": "development_selection_only_not_independent_generalization_evidence",
            "metrics": oof_metrics,
            "policy": policy,
        },
        "model": model_hashes,
        "package": package_report,
        "development_evaluation_accessed": False,
        "next_stage": "run_fixed_emh_and_operational_end_to_end_regression_once",
    }
    _write_json(args.output_dir / "reports" / "training-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Bread 1.1.1+ classifier using only the frozen 200-image allowlist"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
