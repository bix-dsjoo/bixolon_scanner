from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

from ..inference import _prepare_rgb
from ..package import load_model_package, sha256_file
from .bread_data_scale import (
    _evaluate_crossfit,
    _fit_calibration,
    _prepare_evaluation,
    cross_fold_calibrations,
)
from .data import read_manifest
from .fewshot_adapter import (
    AdapterSpec,
    adapter_spec_from_dict,
    build_residual_cosine_head,
    build_ten_shot_classifier,
    compatible_proxy_state_dict,
    supervised_contrastive_loss,
)
from .models import build_dino_classifier, require_torch
from .small_data import build_frofa_training_set, fit_linear_svm_head
from .synthetic_roi import (
    DirectRoiRecipe,
    augment_direct_roi,
    direct_roi_recipe_sha256,
    prepare_direct_roi_source,
)
from .ten_shot_candidates import (
    CandidateResult,
    challenger_required,
    create_experiment_lock,
    create_uniform_parameter_soup,
    freeze_for_challenger,
    l2_sp_penalty,
    select_candidate,
    validate_seed_matrix,
)
from .ten_shot_training import (
    HeadTrainingConfig,
    feature_cache_fingerprint,
    save_head_checkpoint,
    train_adapter_head,
)

PHASES = ("prepare", "baseline", "train", "challenger", "soup", "calibrate", "lock")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("experiment", "audit", "augmentation", "training", "evaluation")
    if any(not isinstance(config.get(key), dict) for key in required):
        raise ValueError(f"10-shot config requires sections: {', '.join(required)}")
    if int(config["experiment"].get("shots_per_class", 0)) != 10:
        raise ValueError("strict registration requires exactly 10 shots per class")
    validate_seed_matrix(config["experiment"].get("seeds", []))
    augmentation = config["augmentation"]
    forbidden = {
        "detector_in_training": False,
        "mixup": False,
        "cutmix": False,
        "super_resolution": False,
    }
    for key, expected in forbidden.items():
        if augmentation.get(key) is not expected:
            raise ValueError(f"strict 10-shot config requires {key}={expected}")
    if augmentation.get("background_source") != "procedural-neutral-only":
        raise ValueError("operating and other bread backgrounds are forbidden")
    if int(augmentation.get("views_per_source", 0)) < 1:
        raise ValueError("views_per_source must be positive")
    training = config["training"]
    if training.get("runtime_support_cache") is not False:
        raise ValueError("runtime support/cache fusion is forbidden")
    if (
        training.get("distillation") is not False
        or training.get("legacy_classifier_initialization") is not False
    ):
        raise ValueError("legacy classifier weights, logits and distillation are forbidden")
    challenger = training.get("challenger", {})
    if (
        challenger.get("trainable_scope") != "backbone.stages[-1]"
        or challenger.get("full_backbone_finetune") is not False
    ):
        raise ValueError("challenger may train only the final ConvNeXt stage")
    soup_config = training.get("parameter_soup", {})
    if soup_config.get("enabled", False):
        member_seeds = tuple(int(seed) for seed in soup_config.get("member_seeds", ()))
        if len(member_seeds) < 2 or not set(member_seeds) <= set(
            validate_seed_matrix(config["experiment"]["seeds"])
        ):
            raise ValueError("parameter soup requires at least two configured seeds")
        if soup_config.get("average_scope") != "full_model":
            raise ValueError("parameter soup average_scope must be full_model")
        if soup_config.get("selection_scope") != "development_capture_session_3fold_only":
            raise ValueError("parameter soup selection must be development-only")
    inference = config.get("inference", {})
    crop_scale = inference.get("center_crop_scale")
    if crop_scale is not None and not 0.5 <= float(crop_scale) < 1.0:
        raise ValueError("inference center_crop_scale must be in [0.5, 1.0)")
    if crop_scale is not None and bool(training.get("tta", {}).get("enabled", False)):
        raise ValueError("selected center crop and TTA cannot be enabled together")
    quantum = inference.get("logit_quantum")
    phase = float(inference.get("logit_phase", 0.0))
    bias_span = float(inference.get("tie_break_bias_span", 0.0))
    divisor = float(inference.get("logit_divisor", 1.0))
    if quantum is not None and float(quantum) <= 0:
        raise ValueError("inference logit_quantum must be positive")
    if quantum is None and (phase != 0.0 or bias_span != 0.0):
        raise ValueError("logit phase and tie-break require logit_quantum")
    if quantum is not None and not 0.0 <= phase < float(quantum):
        raise ValueError("logit_phase must be in [0, logit_quantum)")
    if bias_span < 0.0 or (quantum is not None and bias_span >= float(quantum)):
        raise ValueError("tie_break_bias_span must be in [0, logit_quantum)")
    if divisor <= 0:
        raise ValueError("inference logit_divisor must be positive")
    threshold_guard = float(config["evaluation"].get("approval_threshold_provider_guard", 0.0))
    if not 0.0 <= threshold_guard <= 0.01:
        raise ValueError("approval threshold provider guard must be in [0, 0.01]")
    return config


def _inference_crop_scale(config: dict[str, Any]) -> float | None:
    value = config.get("inference", {}).get("center_crop_scale")
    return None if value is None else float(value)


def _apply_inference_logit_policy(logits: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    inference = config.get("inference", {})
    quantum_value = inference.get("logit_quantum")
    values = np.asarray(logits, dtype=np.float32)
    if quantum_value is not None:
        quantum = np.float32(quantum_value)
        phase = np.float32(inference.get("logit_phase", 0.0))
        values = np.round((values + phase) / quantum) * quantum - phase
    bias_span = np.float32(inference.get("tie_break_bias_span", 0.0))
    if bias_span:
        values = values + np.linspace(0.0, -bias_span, values.shape[1], dtype=np.float32)
    divisor = np.float32(inference.get("logit_divisor", 1.0))
    return np.asarray(values / divisor, dtype=np.float32)


def _apply_approval_threshold_guard(
    calibration: dict[str, Any],
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    guard: float,
    maximum_false_approval_rate: float,
) -> dict[str, Any]:
    if guard <= 0:
        return calibration
    from .calibration import binomial_rate_upper_bound, softmax

    probabilities = softmax(logits, float(calibration["temperature"]))
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    threshold = max(0.0, float(calibration["approval_threshold"]) - guard)
    approved = confidence >= threshold
    approved_count = int(approved.sum())
    errors = int((predictions[approved] != targets[approved]).sum())
    false_upper = binomial_rate_upper_bound(errors, approved_count)
    return calibration | {
        "approval_threshold": threshold,
        "approval_threshold_provider_guard": guard,
        "approved_count": approved_count,
        "approved_precision": (1.0 - errors / approved_count if approved_count else 1.0),
        "approval_coverage": approved_count / len(targets),
        "approved_false_rate_upper_95": false_upper,
        "risk_control_satisfied": bool(
            approved_count > 0 and false_upper <= maximum_false_approval_rate
        ),
    }


def _direct_recipe(config: dict[str, Any]) -> DirectRoiRecipe:
    values = config["augmentation"]
    recipe = DirectRoiRecipe(
        **{key: values[key] for key in DirectRoiRecipe.__dataclass_fields__ if key in values}
    )
    recipe.validate()
    return recipe


def _load_support_records(
    manifest_path: Path, metadata_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = sorted(
        read_manifest(manifest_path),
        key=lambda row: (int(row["category_id"]), str(row["image_path"])),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = int(metadata["class_count"])
    shots = int(metadata["shots_per_class"])
    if shots != 10 or len(records) != classes * shots:
        raise ValueError("manifest is not an exact class-balanced 10-shot dataset")
    if sha256_file(manifest_path) != metadata.get("manifest_sha256"):
        raise ValueError("10-shot manifest checksum does not match metadata")
    counts = Counter(int(row["category_id"]) for row in records)
    if counts != Counter({category: 10 for category in range(1, classes + 1)}):
        raise ValueError("manifest is not balanced or has non-contiguous labels")
    if len({str(row["image_sha256"]) for row in records}) != len(records):
        raise ValueError("manifest contains a duplicate source SHA")
    if any(row.get("split") != "train_support" for row in records):
        raise ValueError("development/test records cannot enter the training manifest")
    return records, metadata


def _device(cpu: bool):
    torch = require_torch()
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


def _build_backbone(args: argparse.Namespace, config: dict[str, Any], classes: int, device):
    model = build_dino_classifier(
        str(config["training"]["backbone_kind"]),
        classes,
        weights_path=args.weights,
        hub_repository=str(config["training"]["hub_repository"]),
    )
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model.backbone.to(device).eval()


def _extract_features(
    backbone, tensors: list[np.ndarray] | np.ndarray, *, device, batch_size: int
) -> np.ndarray:
    torch = require_torch()
    results = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            values = np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            results.append(backbone(torch.from_numpy(values).to(device)).float().cpu().numpy())
    return np.concatenate(results).astype(np.float32)


def _extract_feature_bundle(
    backbone, tensors: list[np.ndarray] | np.ndarray, *, device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    torch = require_torch()
    global_parts, patch_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            values = np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            output = backbone.forward_features(torch.from_numpy(values).to(device))
            global_parts.append(output["x_norm_clstoken"].float().cpu().numpy())
            patch_parts.append(output["x_norm_patchtokens"].float().cpu().numpy())
    return (
        np.concatenate(global_parts).astype(np.float32),
        np.concatenate(patch_parts).astype(np.float32),
    )


def _scaled_center_view(tensors: np.ndarray, scale: float) -> np.ndarray:
    if not 0.5 <= scale < 1.0:
        raise ValueError("TTA crop scale must be in [0.5, 1.0)")
    torch = require_torch()
    values = torch.from_numpy(np.array(tensors, dtype=np.float32, copy=True))
    height, width = values.shape[-2:]
    crop_height = max(1, round(height * scale))
    crop_width = max(1, round(width * scale))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = values[..., top : top + crop_height, left : left + crop_width]
    return torch.nn.functional.interpolate(
        cropped,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ).numpy()


def _extract_evaluation_features(
    backbone,
    tensors: np.ndarray,
    *,
    device,
    batch_size: int,
    crop_scale: float | None,
) -> np.ndarray:
    parts = []
    for start in range(0, len(tensors), batch_size):
        batch = np.asarray(tensors[start : start + batch_size], dtype=np.float32)
        if crop_scale is not None:
            batch = _scaled_center_view(batch, crop_scale)
        parts.append(_extract_features(backbone, batch, device=device, batch_size=batch_size))
    return np.concatenate(parts).astype(np.float32)


def _save_evaluation_feature_bundle(
    backbone,
    tensors: np.ndarray,
    *,
    device,
    batch_size: int,
    global_path: Path,
    patch_path: Path,
    tta_scale: float | None = None,
) -> None:
    global_values = []
    patch_memmap = None
    offset = 0
    for start in range(0, len(tensors), batch_size):
        batch = np.asarray(tensors[start : start + batch_size], dtype=np.float32)
        if tta_scale is not None:
            batch = _scaled_center_view(batch, tta_scale)
        global_batch, patch_batch = _extract_feature_bundle(
            backbone, batch, device=device, batch_size=batch_size
        )
        global_values.append(global_batch)
        if patch_memmap is None:
            patch_memmap = np.lib.format.open_memmap(
                patch_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(tensors), patch_batch.shape[1], patch_batch.shape[2]),
            )
        patch_memmap[offset : offset + len(patch_batch)] = patch_batch.astype(np.float16)
        offset += len(patch_batch)
    np.save(global_path, np.concatenate(global_values).astype(np.float32))
    if patch_memmap is None:
        raise ValueError("cannot extract an empty evaluation feature bundle")
    patch_memmap.flush()


def _classifier_tensor(image: Image.Image, package) -> np.ndarray:
    classifier = package.metadata.classifier
    return _prepare_rgb(
        image,
        classifier.input_size,
        classifier.mean,
        classifier.std,
        reducing_gap=classifier.resize_reducing_gap,
    )


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    records, metadata = _load_support_records(args.manifest, args.manifest_metadata)
    package = load_model_package(args.base_package)
    device = _device(args.cpu)
    backbone = _build_backbone(args, config, int(metadata["class_count"]), device)
    recipe = _direct_recipe(config)
    views = int(config["augmentation"]["views_per_source"])
    batch_size = int(config["training"]["batch_size"])
    base_seed = min(validate_seed_matrix(config["experiment"]["seeds"]))
    use_local_features = bool(config["training"].get("use_local_features", False))
    source_shas = {str(row["image_sha256"]) for row in records}
    support_tensors: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    prepared = args.output_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    local_patch_path = prepared / "training_patch_features.float16.npy"
    local_patch_memmap = None
    local_patch_offset = 0
    for source_index, record in enumerate(records):
        path = args.dataset_root / str(record["image_path"])
        if sha256_file(path) != record["image_sha256"]:
            raise ValueError(f"source checksum changed: {record['image_path']}")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
        support_tensors.append(_classifier_tensor(image, package))
        prepared_cutout = prepare_direct_roi_source(image, recipe)
        source_tensors: list[np.ndarray] = []
        for view in range(views):
            seed = base_seed + source_index * 1_000_003 + view
            sample = augment_direct_roi(
                image,
                source_sha256=str(record["image_sha256"]),
                category_id=int(record["category_id"]),
                seed=seed,
                recipe=recipe,
                prepared_cutout=prepared_cutout,
            )
            if sample.provenance["source_sha256"] not in source_shas:
                raise RuntimeError("augmentation provenance escaped the 200 source SHAs")
            source_tensors.append(_classifier_tensor(sample.image, package))
            provenance.append(
                {**sample.provenance, "source_index": source_index, "view_index": view}
            )
        if use_local_features:
            source_features, source_patches = _extract_feature_bundle(
                backbone, source_tensors, device=device, batch_size=batch_size
            )
            if local_patch_memmap is None:
                local_patch_memmap = np.lib.format.open_memmap(
                    local_patch_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(
                        len(records) * views,
                        source_patches.shape[1],
                        source_patches.shape[2],
                    ),
                )
            local_patch_memmap[local_patch_offset : local_patch_offset + len(source_patches)] = (
                source_patches.astype(np.float16)
            )
            local_patch_offset += len(source_patches)
        else:
            source_features = _extract_features(
                backbone, source_tensors, device=device, batch_size=batch_size
            )
        feature_parts.append(source_features)
        label_parts.append(
            np.full(len(source_features), int(record["category_id"]) - 1, dtype=np.int64)
        )
        source_parts.append(np.full(len(source_features), source_index, dtype=np.int64))
    if use_local_features:
        support_features, support_local_patches = _extract_feature_bundle(
            backbone, support_tensors, device=device, batch_size=batch_size
        )
    else:
        support_features = _extract_features(
            backbone, support_tensors, device=device, batch_size=batch_size
        )
        support_local_patches = None
    features = np.concatenate(feature_parts).astype(np.float32)
    labels = np.concatenate(label_parts).astype(np.int64)
    source_indices = np.concatenate(source_parts).astype(np.int64)
    support_labels = np.asarray([int(row["category_id"]) - 1 for row in records], dtype=np.int64)
    support_proxy_ids = np.asarray(
        [0 if str(row.get("side")) == "normal" else 1 for row in records],
        dtype=np.int64,
    )
    if use_local_features:
        if local_patch_memmap is None:
            raise RuntimeError("local patch cache was not initialized")
        local_patch_memmap.flush()
        del local_patch_memmap

    # Exact frozen DINOv3 + c2FroFA baseline cache from the same 200 clean images.
    torch = require_torch()
    patch_batches = []
    with torch.inference_mode():
        for start in range(0, len(support_tensors), batch_size):
            tensor = torch.from_numpy(
                np.asarray(support_tensors[start : start + batch_size], dtype=np.float32)
            ).to(device)
            patch_batches.append(
                backbone.forward_features(tensor)["x_prenorm"].float().cpu().numpy()
            )
    cache_path = prepared / "training_features.npz"
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        source_indices=source_indices,
        support_features=support_features,
        support_labels=support_labels,
        support_proxy_ids=support_proxy_ids,
    )
    if support_local_patches is not None:
        np.save(
            prepared / "support_local_patch_features.npy",
            support_local_patches.astype(np.float16),
        )
    np.save(
        prepared / "support_patch_features.npy", np.concatenate(patch_batches).astype(np.float32)
    )
    np.savez(
        prepared / "support_patch_norm.npz",
        weight=backbone.norm.weight.detach().float().cpu().numpy(),
        bias=backbone.norm.bias.detach().float().cpu().numpy(),
        epsilon=np.asarray(float(backbone.norm.eps), dtype=np.float64),
    )
    _write_jsonl(prepared / "augmentation_provenance.jsonl", provenance)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": metadata["dataset_version"],
        "manifest_sha256": metadata["manifest_sha256"],
        "source_sha256_count": len(source_shas),
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_cache_fingerprint": feature_cache_fingerprint(
            manifest_sha256=metadata["manifest_sha256"],
            backbone_sha256=sha256_file(args.weights),
            synthetic_recipe_sha256=direct_roi_recipe_sha256(recipe),
            background_sha256=[],
        ),
        "feature_shape": list(features.shape),
        "support_shape": list(support_features.shape),
        "local_patch_shape": (
            [
                len(records) * views,
                int(support_local_patches.shape[1]),
                int(support_local_patches.shape[2]),
            ]
            if support_local_patches is not None
            else None
        ),
        "augmentation_recipe": asdict(recipe),
        "augmentation_recipe_sha256": direct_roi_recipe_sha256(recipe),
        "detector_executed_for_training": False,
        "background_source": "procedural-neutral-only",
        "development_or_test_training_count": 0,
    }
    _write_json(prepared / "training_features.json", report)

    if args.evaluation_manifest:
        if not args.evaluation_dataset_root:
            raise ValueError("evaluation dataset root is required")
        development = [
            row
            for row in read_manifest(args.evaluation_manifest)
            if row.get("record_type") == "detection" and row.get("split") == "development"
        ]
        namespace = argparse.Namespace(
            output_dir=args.output_dir,
            dataset_root=args.evaluation_dataset_root,
            production_package=args.base_package,
            provider=args.provider,
            cuda_dll_dir=args.cuda_dll_dir,
            resume=args.resume,
        )
        _prepare_evaluation(
            development,
            namespace,
            {
                "evaluation": {
                    "match_iou_threshold": float(
                        config["evaluation"].get("match_iou_threshold", 0.5)
                    )
                }
            },
        )
        tensors = np.load(prepared / "evaluation_tensors.npy", mmap_mode="r")
        crop_scale = _inference_crop_scale(config)
        if use_local_features:
            _save_evaluation_feature_bundle(
                backbone,
                tensors,
                device=device,
                batch_size=batch_size,
                global_path=prepared / "evaluation_features.npy",
                patch_path=prepared / "evaluation_patch_features.float16.npy",
                tta_scale=crop_scale,
            )
        else:
            np.save(
                prepared / "evaluation_features.npy",
                _extract_evaluation_features(
                    backbone,
                    tensors,
                    device=device,
                    batch_size=batch_size,
                    crop_scale=crop_scale,
                ),
            )
        tta = config["training"].get("tta", {})
        if bool(tta.get("enabled", False)):
            if use_local_features:
                _save_evaluation_feature_bundle(
                    backbone,
                    tensors,
                    device=device,
                    batch_size=batch_size,
                    global_path=prepared / "evaluation_tta_features.npy",
                    patch_path=prepared / "evaluation_tta_patch_features.float16.npy",
                    tta_scale=float(tta.get("crop_scale", 0.93)),
                )
            else:
                tta_features = []
                for start in range(0, len(tensors), batch_size):
                    view = _scaled_center_view(
                        tensors[start : start + batch_size],
                        float(tta.get("crop_scale", 0.93)),
                    )
                    tta_features.append(
                        _extract_features(backbone, view, device=device, batch_size=batch_size)
                    )
                np.save(
                    prepared / "evaluation_tta_features.npy",
                    np.concatenate(tta_features).astype(np.float32),
                )


def _development_fold_top1(logits: np.ndarray, prepared: Path) -> tuple[float, float, float]:
    rows = read_manifest(prepared / "evaluation_records.jsonl")
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    return tuple(
        float((logits[folds == fold].argmax(1) == targets[folds == fold]).mean())
        for fold in range(3)
    )


def baseline(args: argparse.Namespace, config: dict[str, Any]) -> None:
    prepared = args.output_dir / "prepared"
    patches = np.load(prepared / "support_patch_features.npy", mmap_mode="r")
    norm = np.load(prepared / "support_patch_norm.npz")
    cache = np.load(prepared / "training_features.npz")
    settings = config["training"]["baseline"]
    evaluation_features = np.load(prepared / "evaluation_features.npy")
    runs = []
    for seed in validate_seed_matrix(config["experiment"]["seeds"]):
        features, labels = build_frofa_training_set(
            patches,
            cache["support_labels"],
            layer_norm_weight=norm["weight"],
            layer_norm_bias=norm["bias"],
            layer_norm_epsilon=float(norm["epsilon"]),
            magnitude=float(settings["frofa_brightness_magnitude"]),
            views=int(settings["frofa_views"]),
            seed=seed,
        )
        head = fit_linear_svm_head(
            features,
            labels,
            num_classes=len(np.unique(labels)),
            regularization_c=float(settings["linear_svm_regularization_c"]),
            max_iterations=int(settings["linear_svm_max_iterations"]),
            seed=seed,
        )
        logits = evaluation_features @ head.weights.T + head.bias
        runs.append({"seed": seed, "fold_top1": _development_fold_top1(logits, prepared)})
    _write_json(
        args.output_dir / "reports" / "baseline.json",
        {
            "schema_version": "1.0",
            "recipe": "frozen_dinov3_c2frofa_linear_svm",
            "training_source_count": len(cache["support_features"]),
            "runs": runs,
            "test_accessed": False,
        },
    )


def _head_logits(
    head,
    features: np.ndarray,
    device,
    batch_size: int,
    patch_features: np.ndarray | None = None,
) -> np.ndarray:
    torch = require_torch()
    values = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.array(
                    features[start : start + batch_size],
                    dtype=np.float32,
                    copy=True,
                )
            ).to(device)
            patches = None
            if patch_features is not None:
                patches = torch.from_numpy(
                    np.asarray(patch_features[start : start + batch_size], dtype=np.float32)
                ).to(device)
            values.append(head(batch, patches).float().cpu().numpy())
    return np.concatenate(values)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fuse_tta_logits(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    disagreement_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if primary.shape != secondary.shape or primary.ndim != 2:
        raise ValueError("TTA logits must be aligned 2D matrices")
    if disagreement_weight < 0:
        raise ValueError("TTA disagreement weight cannot be negative")
    first = _softmax_rows(primary)
    second = _softmax_rows(secondary)
    mean = (first + second) * 0.5
    epsilon = 1e-12
    js = 0.5 * (
        (first * np.log(np.clip(first / np.clip(mean, epsilon, None), epsilon, None))).sum(axis=1)
        + (second * np.log(np.clip(second / np.clip(mean, epsilon, None), epsilon, None))).sum(
            axis=1
        )
    )
    mismatch = primary.argmax(axis=1) != secondary.argmax(axis=1)
    disagreement = js + mismatch.astype(np.float64) * 0.25
    fused = (primary + secondary) * 0.5
    fused = fused / (1.0 + disagreement_weight * disagreement[:, None])
    return fused.astype(np.float32), disagreement.astype(np.float32)


def _head_development_logits(
    head,
    prepared: Path,
    *,
    device,
    batch_size: int,
    spec: AdapterSpec,
    tta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    patches = (
        np.load(prepared / "evaluation_patch_features.float16.npy", mmap_mode="r")
        if spec.use_local_features
        else None
    )
    primary = _head_logits(
        head,
        np.load(prepared / "evaluation_features.npy", mmap_mode="r"),
        device,
        batch_size,
        patches,
    )
    if not bool(tta.get("enabled", False)):
        return primary, None
    tta_patches = (
        np.load(prepared / "evaluation_tta_patch_features.float16.npy", mmap_mode="r")
        if spec.use_local_features
        else None
    )
    secondary = _head_logits(
        head,
        np.load(prepared / "evaluation_tta_features.npy", mmap_mode="r"),
        device,
        batch_size,
        tta_patches,
    )
    return fuse_tta_logits(
        primary,
        secondary,
        disagreement_weight=float(tta.get("disagreement_weight", 4.0)),
    )


def _model_logits(model, tensors: np.ndarray, *, device, batch_size: int) -> np.ndarray:
    torch = require_torch()
    values = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.from_numpy(
                np.array(tensors[start : start + batch_size], dtype=np.float32, copy=True)
            ).to(device)
            values.append(model(batch).float().cpu().numpy())
    return np.concatenate(values)


def _model_development_logits(
    model,
    tensors: np.ndarray,
    *,
    device,
    batch_size: int,
    tta: dict[str, Any],
    primary_crop_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if primary_crop_scale is None:
        primary = _model_logits(model, tensors, device=device, batch_size=batch_size)
    else:
        primary_parts = []
        for start in range(0, len(tensors), batch_size):
            view = _scaled_center_view(
                np.asarray(tensors[start : start + batch_size], dtype=np.float32),
                primary_crop_scale,
            )
            primary_parts.append(_model_logits(model, view, device=device, batch_size=batch_size))
        primary = np.concatenate(primary_parts)
    if not bool(tta.get("enabled", False)):
        return primary, None
    secondary_logits = []
    for start in range(0, len(tensors), batch_size):
        view = _scaled_center_view(
            np.asarray(tensors[start : start + batch_size], dtype=np.float32),
            float(tta.get("crop_scale", 0.93)),
        )
        secondary_logits.append(_model_logits(model, view, device=device, batch_size=batch_size))
    secondary = np.concatenate(secondary_logits)
    return fuse_tta_logits(
        primary,
        secondary,
        disagreement_weight=float(tta.get("disagreement_weight", 4.0)),
    )


def train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    prepared = args.output_dir / "prepared"
    report = json.loads((prepared / "training_features.json").read_text(encoding="utf-8"))
    cache_path = prepared / "training_features.npz"
    if sha256_file(cache_path) != report["feature_cache_sha256"]:
        raise ValueError("training feature cache checksum changed")
    cache = np.load(cache_path)
    metadata = json.loads(args.manifest_metadata.read_text(encoding="utf-8"))
    training = config["training"]
    spec = AdapterSpec(
        hidden_size=int(training.get("hidden_size", 768)),
        bottleneck_size=int(training["adapter_bottleneck"]),
        num_classes=int(metadata["class_count"]),
        cosine_scale=float(training["cosine_scale"]),
        cosine_margin=float(training["cosine_margin"]),
        proxies_per_class=int(training.get("proxies_per_class", 1)),
        proxy_temperature=float(training.get("proxy_temperature", 10.0)),
        use_local_features=bool(training.get("use_local_features", False)),
    )
    device = _device(args.cpu)
    patch_features = (
        np.load(prepared / "training_patch_features.float16.npy", mmap_mode="r")
        if spec.use_local_features
        else None
    )
    support_patch_features = (
        np.load(prepared / "support_local_patch_features.npy", mmap_mode="r")
        if spec.use_local_features
        else None
    )
    results = []
    for seed in validate_seed_matrix(config["experiment"]["seeds"]):
        training_config = HeadTrainingConfig(
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            contrastive_weight=float(training["contrastive_weight"]),
            contrastive_temperature=float(training["contrastive_temperature"]),
            seed=seed,
        )
        head, history = train_adapter_head(
            cache["features"],
            cache["labels"],
            cache["source_indices"],
            support_features=cache["support_features"],
            support_labels=cache["support_labels"],
            patch_features=patch_features,
            support_patch_features=support_patch_features,
            support_proxy_ids=(
                cache.get("support_proxy_ids") if spec.proxies_per_class > 1 else None
            ),
            spec=spec,
            config=training_config,
            device=str(device),
        )
        run_dir = args.output_dir / "runs" / "main" / str(seed)
        checkpoint = run_dir / "best.pt"
        save_head_checkpoint(
            checkpoint,
            head=head,
            spec=spec,
            training_config=training_config,
            history=history,
            dataset_version=metadata["dataset_version"],
            manifest_sha256=metadata["manifest_sha256"],
            feature_cache_sha256=report["feature_cache_sha256"],
            backbone_kind=training["backbone_kind"],
            backbone_revision=str(training["hub_repository"]).split(":", 1)[-1],
            backbone_weight_sha256=sha256_file(args.weights),
            backbone_weight_filename=args.weights.name,
            image_size=int(training["image_size"]),
        )
        logits, disagreement = _head_development_logits(
            head,
            prepared,
            device=device,
            batch_size=int(training["batch_size"]),
            spec=spec,
            tta=training.get("tta", {}),
        )
        logits = _apply_inference_logit_policy(logits, config)
        np.save(run_dir / "development_logits.npy", logits)
        if disagreement is not None:
            np.save(run_dir / "development_tta_disagreement.npy", disagreement)
        recipe_name = (
            "frozen_local_multiproxy_cosface_supcon_tta"
            if spec.use_local_features
            else "frozen_adapter_cosface_supcon"
        )
        results.append(
            CandidateResult(
                recipe_name,
                seed,
                str(checkpoint),
                sha256_file(checkpoint),
                _development_fold_top1(logits, prepared),
            )
        )
    selected = select_candidate(results)
    # Challenger is deliberately gated and never silently broadens to full fine-tuning.
    selection = {
        "schema_version": "1.0",
        "runs": [asdict(value) | {"mean_top1": value.mean_top1} for value in results],
        "selected": asdict(selected) | {"mean_top1": selected.mean_top1},
        "challenger_required": challenger_required(
            results, float(training["challenger"]["run_only_if_main_top1_below"])
        ),
        "challenger_scope": training["challenger"]["trainable_scope"],
        "test_accessed": False,
    }
    _write_json(args.output_dir / "classifier" / "selection.json", selection)
    (args.output_dir / "classifier").mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected.checkpoint, args.output_dir / "classifier" / "best.pt")


def challenger(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Evaluate the single allowed challenger: final ConvNeXt stage + L2-SP."""
    torch = require_torch()
    selection_path = args.output_dir / "classifier" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection["challenger_required"]:
        selection["challenger"] = {"status": "not_required", "reason": "main mean Top-1 >= 0.95"}
        _write_json(selection_path, selection)
        return
    records, metadata = _load_support_records(args.manifest, args.manifest_metadata)
    package = load_model_package(args.base_package)
    recipe = _direct_recipe(config)
    views = int(config["augmentation"]["views_per_source"])
    training = config["training"]
    settings = training["challenger"]
    device = _device(args.cpu)
    spec = AdapterSpec(
        hidden_size=int(training.get("hidden_size", 768)),
        bottleneck_size=int(training["adapter_bottleneck"]),
        num_classes=int(metadata["class_count"]),
        cosine_scale=float(training["cosine_scale"]),
        cosine_margin=float(training["cosine_margin"]),
        proxies_per_class=int(training.get("proxies_per_class", 1)),
        proxy_temperature=float(training.get("proxy_temperature", 10.0)),
        use_local_features=bool(training.get("use_local_features", False)),
    )

    class DirectDataset(torch.utils.data.Dataset):
        def __init__(self, seed: int, cache_path: Path):
            self.images = []
            self.cutouts = []
            for record in records:
                with Image.open(args.dataset_root / str(record["image_path"])) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB").copy()
                    self.images.append(image)
                    self.cutouts.append(prepare_direct_roi_source(image, recipe))
            expected_shape = (
                len(records) * views,
                3,
                int(training["image_size"]),
                int(training["image_size"]),
            )
            if cache_path.is_file():
                cached = np.load(cache_path, mmap_mode="r")
                if cached.shape != expected_shape or cached.dtype != np.float16:
                    raise ValueError("challenger tensor cache contract mismatch")
            else:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cached = np.lib.format.open_memmap(
                    cache_path, mode="w+", dtype=np.float16, shape=expected_shape
                )
                for source_index, record in enumerate(records):
                    for view in range(views):
                        index = source_index * views + view
                        sample = augment_direct_roi(
                            self.images[source_index],
                            source_sha256=str(record["image_sha256"]),
                            category_id=int(record["category_id"]),
                            seed=seed + source_index * 1_000_003 + view,
                            recipe=recipe,
                            prepared_cutout=self.cutouts[source_index],
                        )
                        cached[index] = _classifier_tensor(sample.image, package).astype(np.float16)
                    if (source_index + 1) % 20 == 0:
                        print(
                            json.dumps(
                                {
                                    "challenger_cache_seed": seed,
                                    "prepared_sources": source_index + 1,
                                    "total_sources": len(records),
                                }
                            ),
                            flush=True,
                        )
                cached.flush()
                del cached
            self.tensors = np.load(cache_path, mmap_mode="r")

        def __len__(self):
            return len(records) * views

        def __getitem__(self, index: int):
            source_index, _ = divmod(index, views)
            return (
                torch.from_numpy(np.array(self.tensors[index], copy=True)),
                int(records[source_index]["category_id"]) - 1,
            )

    evaluation_tensors = np.load(
        args.output_dir / "prepared" / "evaluation_tensors.npy", mmap_mode="r"
    )
    results: list[CandidateResult] = []
    for seed in validate_seed_matrix(config["experiment"]["seeds"]):
        main_checkpoint = torch.load(
            args.output_dir / "runs" / "main" / str(seed) / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = build_ten_shot_classifier(
            backbone_kind=training["backbone_kind"],
            weights_path=args.weights,
            hub_repository=training["hub_repository"],
            spec=spec,
        )
        model.classifier.load_state_dict(
            compatible_proxy_state_dict(main_checkpoint["head_state_dict"])
        )
        model = model.to(device)
        frozen = freeze_for_challenger(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(settings["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        generator = torch.Generator().manual_seed(seed)
        run_dir = args.output_dir / "runs" / "challenger" / str(seed)
        loader = torch.utils.data.DataLoader(
            DirectDataset(seed, run_dir / "augmented_tensors.float16.npy"),
            batch_size=int(training["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        history = []
        for epoch in range(1, int(settings.get("epochs", 10)) + 1):
            model.train()
            total = samples = 0
            for pixels, labels in loader:
                pixels, labels = pixels.to(device, dtype=torch.float32), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                features = model.extract_features(pixels)
                if isinstance(features, tuple):
                    logits, adapted = model.classifier.training_logits(
                        features[0], labels, features[1]
                    )
                else:
                    logits, adapted = model.classifier.training_logits(features, labels)
                ce = torch.nn.functional.cross_entropy(logits, labels)
                contrastive = supervised_contrastive_loss(
                    adapted, labels, temperature=float(training["contrastive_temperature"])
                )
                penalty = l2_sp_penalty(model, frozen["reference"])
                loss = (
                    ce
                    + float(training["contrastive_weight"]) * contrastive
                    + float(settings["l2_sp_weight"]) * penalty
                )
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(labels)
                samples += len(labels)
            history.append({"epoch": epoch, "loss": total / samples})
        model.eval()
        logits, disagreement = _model_development_logits(
            model,
            evaluation_tensors,
            device=device,
            batch_size=int(training["batch_size"]),
            tta=training.get("tta", {}),
            primary_crop_scale=_inference_crop_scale(config),
        )
        logits = _apply_inference_logit_policy(logits, config)
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "best.pt"
        checkpoint = {
            **{
                key: value
                for key, value in main_checkpoint.items()
                if key not in {"head_state_dict", "history", "architecture"}
            },
            "architecture": "ten_shot_residual_cosine_challenger",
            "adapter_spec": asdict(spec),
            "model_state_dict": model.state_dict(),
            "history": history,
            "challenger": {
                "trainable_scope": "backbone.stages[-1]",
                "full_backbone_finetune": False,
                "learning_rate": float(settings["learning_rate"]),
                "l2_sp_weight": float(settings["l2_sp_weight"]),
            },
        }
        torch.save(checkpoint, checkpoint_path)
        np.save(run_dir / "development_logits.npy", logits)
        if disagreement is not None:
            np.save(run_dir / "development_tta_disagreement.npy", disagreement)
        results.append(
            CandidateResult(
                "last_stage_l2sp",
                seed,
                str(checkpoint_path),
                sha256_file(checkpoint_path),
                _development_fold_top1(logits, args.output_dir / "prepared"),
            )
        )
    challenger_selected = select_candidate(results)
    current = CandidateResult(
        recipe=selection["selected"]["recipe"],
        seed=int(selection["selected"]["seed"]),
        checkpoint=selection["selected"]["checkpoint"],
        checkpoint_sha256=selection["selected"]["checkpoint_sha256"],
        fold_top1=tuple(selection["selected"]["fold_top1"]),
    )
    final = select_candidate([current, challenger_selected])
    selection["challenger"] = {
        "status": "evaluated_once",
        "runs": [asdict(value) | {"mean_top1": value.mean_top1} for value in results],
    }
    selection["selected"] = asdict(final) | {"mean_top1": final.mean_top1}
    _write_json(selection_path, selection)
    shutil.copy2(final.checkpoint, args.output_dir / "classifier" / "best.pt")


def soup(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Create and evaluate one fixed, single-runtime-model parameter soup."""
    torch = require_torch()
    settings = config["training"].get("parameter_soup", {})
    if not bool(settings.get("enabled", False)):
        return
    seeds = tuple(int(seed) for seed in settings["member_seeds"])
    checkpoint_paths = tuple(
        args.output_dir / "runs" / "challenger" / str(seed) / "best.pt" for seed in seeds
    )
    output_path = args.output_dir / "runs" / "soup" / "best.pt"
    provenance = create_uniform_parameter_soup(
        checkpoint_paths,
        output_path,
        member_seeds=seeds,
    )
    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    spec = adapter_spec_from_dict(checkpoint["adapter_spec"])
    model = build_ten_shot_classifier(
        backbone_kind=checkpoint["backbone_kind"],
        weights_path=args.weights,
        hub_repository=config["training"]["hub_repository"],
        spec=spec,
    )
    model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
    device = _device(args.cpu)
    model = model.to(device).eval()
    logits, _ = _model_development_logits(
        model,
        np.load(
            args.output_dir / "prepared" / "evaluation_tensors.npy",
            mmap_mode="r",
        ),
        device=device,
        batch_size=int(config["training"]["batch_size"]),
        tta=config["training"].get("tta", {}),
        primary_crop_scale=_inference_crop_scale(config),
    )
    logits = _apply_inference_logit_policy(logits, config)
    np.save(args.output_dir / "runs" / "soup" / "development_logits.npy", logits)
    folds = _development_fold_top1(logits, args.output_dir / "prepared")
    result = CandidateResult(
        recipe="uniform_full_model_parameter_soup",
        seed=min(seeds),
        checkpoint=str(output_path),
        checkpoint_sha256=sha256_file(output_path),
        fold_top1=folds,
    )
    selection_path = args.output_dir / "classifier" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    current_value = selection["selected"]
    current = CandidateResult(
        recipe=current_value["recipe"],
        seed=int(current_value["seed"]),
        checkpoint=current_value["checkpoint"],
        checkpoint_sha256=current_value["checkpoint_sha256"],
        fold_top1=tuple(current_value["fold_top1"]),
    )
    final = (
        result
        if current.recipe == "uniform_full_model_parameter_soup"
        else select_candidate([current, result])
    )
    selection["parameter_soup"] = {
        "status": "evaluated_once",
        "provenance": provenance,
        "result": asdict(result) | {"mean_top1": result.mean_top1},
    }
    selection["selected"] = asdict(final) | {"mean_top1": final.mean_top1}
    _write_json(selection_path, selection)
    shutil.copy2(final.checkpoint, args.output_dir / "classifier" / "best.pt")


def calibrate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    torch = require_torch()
    checkpoint_path = args.output_dir / "classifier" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = _device(args.cpu)
    prepared = args.output_dir / "prepared"
    if checkpoint["architecture"] == "ten_shot_residual_cosine":
        spec = adapter_spec_from_dict(checkpoint["adapter_spec"])
        head = build_residual_cosine_head(spec)
        head.load_state_dict(compatible_proxy_state_dict(checkpoint["head_state_dict"]))
        logits, _ = _head_development_logits(
            head.to(device).eval(),
            prepared,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
            spec=spec,
            tta=config["training"].get("tta", {}),
        )
    elif checkpoint["architecture"] == "ten_shot_residual_cosine_challenger":
        model = build_ten_shot_classifier(
            backbone_kind=checkpoint["backbone_kind"],
            weights_path=args.weights,
            hub_repository=config["training"]["hub_repository"],
            spec=adapter_spec_from_dict(checkpoint["adapter_spec"]),
        )
        model.load_state_dict(compatible_proxy_state_dict(checkpoint["model_state_dict"]))
        model = model.to(device).eval()
        tensors = np.load(prepared / "evaluation_tensors.npy", mmap_mode="r")
        logits, _ = _model_development_logits(
            model,
            tensors,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
            tta=config["training"].get("tta", {}),
            primary_crop_scale=_inference_crop_scale(config),
        )
    else:
        raise ValueError("unsupported selected classifier architecture")
    logits = _apply_inference_logit_policy(logits, config)
    rows = read_manifest(prepared / "evaluation_records.jsonl")
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    evaluation_config = {
        "experiment": {
            "fold_count": 3,
            "expected_num_classes": checkpoint["num_classes"],
            "bootstrap_repetitions": 1000,
            "seed": checkpoint["training_config"]["seed"],
            "max_false_approval_rate": float(
                config["evaluation"]["maximum_false_approval_rate_upper_95"]
            ),
            "confidence_level": float(config["evaluation"]["confidence_level"]),
        }
    }
    fold_calibrations = cross_fold_calibrations(logits, targets, folds, evaluation_config)
    detector_report = json.loads((prepared / "detector_report.json").read_text(encoding="utf-8"))
    crossfit = _evaluate_crossfit(
        logits, rows, detector_report, fold_calibrations, evaluation_config
    )
    calibration = _fit_calibration(logits, targets, evaluation_config)
    if int(calibration.get("approved_count", 0)) == 0:
        from .calibration import (
            binomial_rate_upper_bound,
            fit_temperature,
            softmax,
            topk_accuracy,
        )

        matched = targets >= 0
        matched_logits = logits[matched]
        matched_targets = targets[matched]
        temperature = fit_temperature(matched_logits, matched_targets)
        probabilities = softmax(matched_logits, temperature)
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        order = np.argsort(-confidence, kind="stable")
        errors = predictions[order] != matched_targets[order]
        first_error = int(np.flatnonzero(errors)[0]) if bool(errors.any()) else len(errors)
        _write_json(
            args.output_dir / "reports" / "calibration_failure.json",
            {
                "schema_version": "1.0",
                "status": "failed",
                "reason": "zero_approved_samples_at_risk_control_threshold",
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "test_accessed": False,
                "sample_count": len(matched_targets),
                "temperature": temperature,
                "top1_accuracy": float((predictions == matched_targets).mean()),
                "overall_top3_accuracy": topk_accuracy(probabilities, matched_targets, 3),
                "first_error_rank": first_error + 1 if first_error < len(errors) else None,
                "maximum_zero_error_approval_count": first_error,
                "maximum_zero_error_approval_coverage": first_error / len(errors),
                "zero_error_false_approval_rate_upper_95": binomial_rate_upper_bound(
                    0, first_error
                ),
                "required_false_approval_rate_upper_95": float(
                    config["evaluation"]["maximum_false_approval_rate_upper_95"]
                ),
                "promotion_status": "experiment_only",
            },
        )
        raise RuntimeError("calibration failed: zero approved development samples")
    matched = targets >= 0
    calibration = _apply_approval_threshold_guard(
        calibration,
        logits[matched],
        targets[matched],
        guard=float(config["evaluation"].get("approval_threshold_provider_guard", 0.0)),
        maximum_false_approval_rate=float(
            config["evaluation"]["maximum_false_approval_rate_upper_95"]
        ),
    )
    from .calibration import binomial_rate_upper_bound, softmax, topk_accuracy

    matched_targets = targets[matched]
    matched_folds = folds[matched]
    probabilities = softmax(logits[matched], float(calibration["temperature"]))
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    ordered_errors = predictions[order] != matched_targets[order]
    first_error_index = (
        int(np.flatnonzero(ordered_errors)[0])
        if bool(ordered_errors.any())
        else len(ordered_errors)
    )
    coverage_85_count = int(np.ceil(len(ordered_errors) * 0.85))
    coverage_85_errors = int(ordered_errors[:coverage_85_count].sum())
    approval_mask = confidence >= float(calibration["approval_threshold"])
    fold_threshold_audit = {}
    for fold in sorted(set(matched_folds.tolist())):
        fold_mask = matched_folds == fold
        fold_approved = fold_mask & approval_mask
        approved_count = int(fold_approved.sum())
        error_count = int((predictions[fold_approved] != matched_targets[fold_approved]).sum())
        fold_threshold_audit[str(fold)] = {
            "sample_count": int(fold_mask.sum()),
            "approved_count": approved_count,
            "approval_coverage": approved_count / int(fold_mask.sum()),
            "observed_error_count": error_count,
            "observed_approved_precision": (
                1.0 - error_count / approved_count if approved_count else 1.0
            ),
            "false_approval_rate_upper_95": binomial_rate_upper_bound(error_count, approved_count),
        }
    calibration.update(
        {
            "schema_version": "1.0",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "development_only": True,
            "test_accessed": False,
            "selection_rule": (
                "maximum coverage with 95% false-approval upper bound <= 0.5%; "
                "then fixed provider guard with risk recomputed"
            ),
            "inference_center_crop_scale": _inference_crop_scale(config),
            "fold_calibrations": fold_calibrations,
            "fold_calibrations_role": (
                "diagnostic_only; each two-fold subset is too small for a zero-error "
                "0.5% Clopper-Pearson upper bound"
            ),
            "global_threshold_fold_audit": fold_threshold_audit,
        }
    )
    _write_json(args.output_dir / "reports" / "crossfit_evaluation.json", crossfit)
    _write_json(args.output_dir / "reports" / "calibration.json", calibration)
    fold_top1 = _development_fold_top1(logits, prepared)
    fold_observed_precision = all(
        int(value["approved_count"]) > 0
        and float(value["observed_approved_precision"])
        >= float(config["evaluation"]["minimum_approved_precision"])
        for value in fold_threshold_audit.values()
    )
    checks = {
        "top1_floor": float(crossfit["overall_top1_accuracy"])
        >= float(config["evaluation"]["minimum_top1_accuracy"]),
        "overall_top3_floor": float(crossfit["overall_top3_accuracy"])
        >= float(config["evaluation"]["minimum_overall_top3_accuracy"]),
        "global_approval_samples_present": int(calibration["approved_count"]) > 0,
        "global_approval_risk_upper_bound": bool(calibration["risk_control_satisfied"]),
        "global_approval_coverage_floor": float(calibration["approval_coverage"])
        >= float(config["evaluation"]["minimum_approval_coverage"]),
        "development_fold_top1_floor": min(fold_top1)
        >= float(config["evaluation"]["minimum_top1_accuracy"]),
        "global_threshold_all_folds_observed_precision": fold_observed_precision,
        "coverage_85_has_no_false_approvals": coverage_85_errors == 0,
    }
    target_top1 = float((predictions == matched_targets).mean()) >= float(
        config["evaluation"]["target_top1_accuracy"]
    )
    candidate_tier = (
        "target_candidate"
        if all(checks.values()) and target_top1
        else "waiver_candidate"
        if all(checks.values())
        else "experiment_only"
    )
    decision = {
        "schema_version": "1.0",
        "promotion_status": "experiment_only",
        "candidate_tier": candidate_tier,
        "promotion_deferred_until_locked_regression_parity_and_benchmark": True,
        "test_accessed": False,
        "onnx_exported": False,
        "benchmark_executed": False,
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "metrics": {
            "sample_count": len(matched_targets),
            "top1_accuracy": float((predictions == matched_targets).mean()),
            "fold_top1_accuracy": list(fold_top1),
            "top1_target_met": target_top1,
            "overall_top3_accuracy": topk_accuracy(probabilities, matched_targets, 3),
            "approval_count": int(calibration["approved_count"]),
            "approval_coverage": float(calibration["approval_coverage"]),
            "approved_precision": float(calibration["approved_precision"]),
            "false_approval_rate_upper_95": float(calibration["approved_false_rate_upper_95"]),
            "first_error_rank": (
                first_error_index + 1 if first_error_index < len(ordered_errors) else None
            ),
            "coverage_85_approved_count": coverage_85_count,
            "coverage_85_error_count": coverage_85_errors,
            "coverage_85_precision": 1.0 - coverage_85_errors / coverage_85_count,
            "coverage_85_false_approval_rate_upper_95": binomial_rate_upper_bound(
                coverage_85_errors, coverage_85_count
            ),
        },
    }
    _write_json(args.output_dir / "reports" / "development_decision.json", decision)


def lock(args: argparse.Namespace, config: dict[str, Any]) -> None:
    selection = json.loads(
        (args.output_dir / "classifier" / "selection.json").read_text(encoding="utf-8")
    )["selected"]
    selected = CandidateResult(
        recipe=selection["recipe"],
        seed=int(selection["seed"]),
        checkpoint=selection["checkpoint"],
        checkpoint_sha256=selection["checkpoint_sha256"],
        fold_top1=tuple(selection["fold_top1"]),
    )
    create_experiment_lock(
        args.output_dir / "lock" / "pretest-lock.json",
        config_path=args.config,
        manifest_path=args.manifest,
        manifest_metadata_path=args.manifest_metadata,
        checkpoint_path=args.output_dir / "classifier" / "best.pt",
        calibration_path=args.output_dir / "reports" / "calibration.json",
        selected=replace(
            selected,
            checkpoint=str(args.output_dir / "classifier" / "best.pt"),
            checkpoint_sha256=sha256_file(args.output_dir / "classifier" / "best.pt"),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a strict versioned 10-shot classifier experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--evaluation-dataset-root", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase", choices=("all",) + PHASES, default="all")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    for phase in PHASES if args.phase == "all" else (args.phase,):
        globals()[phase](args, config)


if __name__ == "__main__":
    main()
