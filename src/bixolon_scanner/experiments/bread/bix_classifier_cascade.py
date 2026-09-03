from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ...configuration import load_json_config
from ...pipeline.ports import Detection
from ...runtime.onnx import (
    apply_classifier_background_masks,
    classifier_crop_box,
    classifier_neighbor_ownership_mask,
    prepare_rgb,
)
from ...training.classifier_allowlist import sha256_file
from ...training.models import DINO_V3_HUB_REPOSITORY, require_torch
from .classifier_200_only import (
    MEAN,
    STD,
    _build_model,
    _classification_metrics,
    _extract_features,
    fit_small_sample_head,
    predict_ridge,
)
from .single2_augmentation_ablation import generate_and_extract, load_feature_cache

BANKS = ("clean", "appearance", "geometry", "context", "probe", "reject")
TRAIN_BANKS = ("clean", "appearance", "geometry", "context")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def bix_source_view_fold(*, side: str, view: str) -> int:
    if view == "vertical" or view == "ground_60_dir_02":
        return 0
    if view == "ground_30_dir_01" or (view == "ground_60_dir_01" and side == "normal"):
        return 1
    if view == "ground_30_dir_02" or (view == "ground_60_dir_01" and side == "flipped"):
        return 2
    raise ValueError(f"unmapped source view: side={side}, view={view}")


def build_source_records(source_dir: Path, dataset_root: Path) -> list[dict[str, Any]]:
    source_dir = source_dir.resolve()
    dataset_root = dataset_root.resolve()
    try:
        source_dir.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("source directory must be inside the configured dataset root") from exc
    rows: list[dict[str, Any]] = []
    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        prefix, category_text, class_slug = class_dir.name.split("_", 2)
        if prefix != "bread" or not category_text.isdigit():
            raise ValueError(f"invalid class directory: {class_dir.name}")
        category_id = int(category_text)
        class_id = f"bread_{category_id:02d}"
        files = sorted(path for path in class_dir.iterdir() if path.is_file())
        for path in files:
            stem_prefix = f"{class_id}_"
            if not path.stem.startswith(stem_prefix):
                raise ValueError(f"invalid source filename: {path.name}")
            descriptor = path.stem[len(stem_prefix) :]
            side, separator, view = descriptor.partition("_")
            if not separator or side not in {"normal", "flipped"}:
                raise ValueError(f"invalid source view descriptor: {path.name}")
            with Image.open(path) as source:
                source.verify()
                width, height = source.size
            sha256 = sha256_file(path)
            rows.append(
                {
                    "schema_version": "1.0",
                    "record_type": "classification",
                    "source_dataset": "bix_bakery_dataset",
                    "image_path": path.relative_to(dataset_root).as_posix(),
                    "image_sha256": sha256,
                    "width": width,
                    "height": height,
                    "category_id": category_id,
                    "class_id": class_id,
                    "class_name": class_slug.replace("_", " ").title(),
                    "side": side,
                    "view": view,
                    "fold": bix_source_view_fold(side=side, view=view),
                    "target": category_id - 1,
                    "physical_item_id": None,
                    "capture_session_id": None,
                }
            )
    rows.sort(key=lambda row: (int(row["category_id"]), str(row["image_path"])))
    if len(rows) != 200:
        raise ValueError(f"classifier source must contain exactly 200 images, found {len(rows)}")
    class_counts = Counter(int(row["category_id"]) for row in rows)
    if class_counts != Counter({category_id: 10 for category_id in range(1, 21)}):
        raise ValueError("classifier source must contain 20 classes x 10 images")
    hashes = [str(row["image_sha256"]) for row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("classifier source contains duplicate image content")
    return rows


def validate_source_directory(config: dict[str, Any], source_dir: Path) -> None:
    allowed = Path(str(config["dataset"]["allowed_root"])).resolve()
    if source_dir.resolve() != allowed:
        raise ValueError(f"classifier image access is locked to {allowed}")


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _masked_tensor(
    image: Image.Image,
    provenance: dict[str, Any],
    size: int,
    *,
    margin_ratio: float = 0.05,
    distance_bias: float = 0.0,
) -> np.ndarray:
    target_box = provenance.get("target_bbox_xyxy")
    if target_box is None:
        return prepare_rgb(image, (size, size), MEAN, STD, reducing_gap=1.0)
    detections = [Detection(*target_box, score=1.0)]
    detections.extend(Detection(*row["bbox_xyxy"], score=1.0) for row in provenance["distractors"])
    crop_box = classifier_crop_box(
        detections[0],
        image.width,
        image.height,
        margin_ratio=margin_ratio,
        crop_mode="box_resize",
    )
    tensor = prepare_rgb(
        image.crop(crop_box),
        (size, size),
        MEAN,
        STD,
        reducing_gap=1.0,
    )
    mask = classifier_neighbor_ownership_mask(
        detections,
        0,
        image_width=image.width,
        image_height=image.height,
        output_size=size,
        margin_ratio=margin_ratio,
        distance_bias=distance_bias,
        shared_scale=False,
    )
    return apply_classifier_background_masks(tensor[None], mask[None])[0]


def _bank_tensors(
    *,
    bank: str,
    size: int,
    records: list[dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    tensors: list[np.ndarray] = []
    labels: list[int] = []
    folds: list[int] = []
    views: list[int] = []
    if bank == "clean":
        for record in records:
            with Image.open(dataset_root / str(record["image_path"])) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                tensors.append(prepare_rgb(image, (size, size), MEAN, STD, reducing_gap=1.0))
            labels.append(int(record["target"]))
            folds.append(int(record["fold"]))
            views.append(0)
        return tensors, np.asarray(labels), np.asarray(folds), np.asarray(views)

    rows = _manifest_rows(output_dir / "manifests" / f"{bank}.jsonl")
    for row in rows:
        path = Path(str(row["derived_path"]))
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensors.append(_masked_tensor(image, dict(row["provenance"]), size))
        labels.append(int(row["target"]))
        folds.append(int(row["fold"]))
        views.append(int(row["view_index"]))
    return tensors, np.asarray(labels), np.asarray(folds), np.asarray(views)


def _build_vit(weights: Path):
    torch = require_torch()
    backbone = torch.hub.load(
        DINO_V3_HUB_REPOSITORY,
        "dinov3_vitb16",
        source="github",
        trust_repo=True,
        verbose=False,
        pretrained=False,
    )
    backbone.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )

    class VitWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def extract_features(self, pixel_values):
            return self.wrapped.forward_features(pixel_values, masks=None)["x_norm_clstoken"]

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return VitWrapper(backbone)


def prepare_cascade_features(
    *,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    convnext_weights: Path,
    vit_weights: Path,
    cpu: bool,
    reuse: bool,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    cascade_dir = output_dir / "cascade-features"
    required = [cascade_dir / f"convnext-192-{bank}.npz" for bank in BANKS] + [
        cascade_dir / f"vit-160-{bank}.npz" for bank in ("clean", "probe", "reject")
    ]
    if reuse and all(path.is_file() for path in required):
        result: dict[str, dict[str, dict[str, np.ndarray]]] = {
            "convnext_192": {},
            "convnext_224": {},
            "vit_160": {},
        }
        _, cached_224 = load_feature_cache(output_dir)
        for bank in BANKS:
            values = np.load(cascade_dir / f"convnext-192-{bank}.npz")
            result["convnext_192"][bank] = {key: values[key] for key in values.files}
            result["convnext_224"][bank] = cached_224[bank]
        for bank in ("clean", "probe", "reject"):
            values = np.load(cascade_dir / f"vit-160-{bank}.npz")
            result["vit_160"][bank] = {key: values[key] for key in values.files}
        return result

    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    batch_size = int(config["training"]["feature_batch_size"])
    convnext = _build_model(convnext_weights).to(device).eval()
    vit = _build_vit(vit_weights).to(device).eval()
    result = {"convnext_192": {}, "convnext_224": {}, "vit_160": {}}
    cached_224, _ = load_feature_cache(output_dir)
    cascade_dir.mkdir(parents=True, exist_ok=True)

    for bank in BANKS:
        tensors, labels, folds, views = _bank_tensors(
            bank=bank,
            size=192,
            records=records,
            dataset_root=dataset_root,
            output_dir=output_dir,
        )
        feature = _extract_features(convnext, tensors, device=device, batch_size=batch_size)
        values = {"features": feature, "labels": labels, "folds": folds, "views": views}
        result["convnext_192"][bank] = values
        result["convnext_224"][bank] = cached_224[bank]
        np.savez_compressed(cascade_dir / f"convnext-192-{bank}.npz", **values)

    del convnext
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for bank in ("clean", "probe", "reject"):
        tensors, labels, folds, views = _bank_tensors(
            bank=bank,
            size=160,
            records=records,
            dataset_root=dataset_root,
            output_dir=output_dir,
        )
        feature = _extract_features(vit, tensors, device=device, batch_size=batch_size)
        values = {"features": feature, "labels": labels, "folds": folds, "views": views}
        result["vit_160"][bank] = values
        np.savez_compressed(cascade_dir / f"vit-160-{bank}.npz", **values)
    _write_json(
        cascade_dir / "metadata.json",
        {
            "schema_version": "1.0",
            "device": str(device),
            "convnext_weights_sha256": sha256_file(convnext_weights),
            "vit_weights_sha256": sha256_file(vit_weights),
            "source_count": len(records),
            "external_image_accessed": False,
        },
    )
    return result


def _training_matrix(
    features: dict[str, dict[str, dict[str, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    folds: list[np.ndarray] = []
    for resolution in ("convnext_192", "convnext_224"):
        for bank in TRAIN_BANKS:
            row = features[resolution][bank]
            values.append(row["features"])
            labels.append(row["labels"])
            folds.append(row["folds"])
    return np.concatenate(values), np.concatenate(labels), np.concatenate(folds)


def _normalized_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    return logits / np.maximum(np.linalg.norm(logits, axis=1, keepdims=True), 1e-12)


def _margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.sort(_normalized_logits(logits), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def route_predictions(
    logits_192: np.ndarray,
    logits_224: np.ndarray,
    logits_vit: np.ndarray,
    *,
    primary_margin: float,
    fallback_margin: float,
    verifier_margin: float,
) -> dict[str, np.ndarray]:
    score_192 = _normalized_logits(logits_192)
    score_224 = _normalized_logits(logits_224)
    score_vit = _normalized_logits(logits_vit)
    pred_192 = score_192.argmax(axis=1)
    pred_224 = score_224.argmax(axis=1)
    pred_vit = score_vit.argmax(axis=1)
    fallback = _margin(score_192) < primary_margin
    fused = _normalized_logits(score_192 + score_224)
    pred_fused = fused.argmax(axis=1)
    fallback_candidate = fallback & (pred_192 == pred_224) & (_margin(fused) >= fallback_margin)
    verifier = fallback_candidate
    approved = ~fallback
    approved |= verifier & (pred_vit == pred_fused) & (_margin(score_vit) >= verifier_margin)
    final_scores = score_192.copy()
    # Once the detail path is selected, its higher-resolution ranking is the
    # canonical Top-3. The fused score is used only for agreement/margin safety.
    final_scores[fallback] = score_224[fallback]
    predictions = final_scores.argmax(axis=1)
    return {
        "scores": final_scores,
        "predictions": predictions,
        "approved": approved,
        "fallback": fallback,
        "verifier": verifier,
    }


def _cascade_metrics(result: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    ranking = np.argsort(-result["scores"], axis=1, kind="stable")
    top3 = np.any(ranking[:, :3] == labels[:, None], axis=1)
    correct = result["predictions"] == labels
    approved = result["approved"]
    return {
        "sample_count": len(labels),
        "approved_count": int(np.count_nonzero(approved)),
        "correct_approved_count": int(np.count_nonzero(approved & correct)),
        "wrong_approved_count": int(np.count_nonzero(approved & ~correct)),
        "unknown_count": int(np.count_nonzero(~approved)),
        "correct_approval_rate": float(np.mean(approved & correct)),
        "top1_accuracy": float(np.mean(correct)),
        "top3_accuracy": float(np.mean(top3)),
        "top3_miss_count": int(np.count_nonzero(~top3)),
        "fallback_count": int(np.count_nonzero(result["fallback"])),
        "verifier_count": int(np.count_nonzero(result["verifier"])),
    }


def _fit_logits(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    candidate: dict[str, Any],
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight, bias = fit_small_sample_head(
        train_features,
        train_labels,
        candidate=candidate,
        class_count=class_count,
    )
    return predict_ridge(evaluation_features, weight, bias), weight, bias


def _select_inner_policy(
    *,
    config: dict[str, Any],
    features: dict[str, dict[str, dict[str, np.ndarray]]],
    train_matrix: tuple[np.ndarray, np.ndarray, np.ndarray],
    inner_train_fold: int,
    inner_validation_fold: int,
) -> dict[str, Any]:
    train_features, train_labels, train_folds = train_matrix
    probe_192 = features["convnext_192"]["probe"]
    probe_224 = features["convnext_224"]["probe"]
    vit_clean = features["vit_160"]["clean"]
    vit_probe = features["vit_160"]["probe"]
    reject_192 = features["convnext_192"]["reject"]
    reject_224 = features["convnext_224"]["reject"]
    vit_reject = features["vit_160"]["reject"]
    probe_mask = probe_192["folds"] == inner_validation_fold
    reject_mask = reject_192["folds"] == inner_validation_fold
    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    for shared_head in config["training"]["shared_head_candidates"]:
        shared_train = train_folds == inner_train_fold
        logits_192, _, _ = _fit_logits(
            train_features[shared_train],
            train_labels[shared_train],
            probe_192["features"][probe_mask],
            candidate=shared_head,
            class_count=20,
        )
        shared_weight, shared_bias = fit_small_sample_head(
            train_features[shared_train],
            train_labels[shared_train],
            candidate=shared_head,
            class_count=20,
        )
        logits_224 = predict_ridge(probe_224["features"][probe_mask], shared_weight, shared_bias)
        reject_logits_192 = predict_ridge(
            reject_192["features"][reject_mask], shared_weight, shared_bias
        )
        reject_logits_224 = predict_ridge(
            reject_224["features"][reject_mask], shared_weight, shared_bias
        )
        for verifier_head in config["training"]["verifier_head_candidates"]:
            verifier_train = vit_clean["folds"] == inner_train_fold
            verifier_weight, verifier_bias = fit_small_sample_head(
                vit_clean["features"][verifier_train],
                vit_clean["labels"][verifier_train],
                candidate=verifier_head,
                class_count=20,
            )
            logits_vit = predict_ridge(
                vit_probe["features"][probe_mask], verifier_weight, verifier_bias
            )
            reject_logits_vit = predict_ridge(
                vit_reject["features"][reject_mask], verifier_weight, verifier_bias
            )
            for primary_margin in config["routing"]["primary_margin_candidates"]:
                for fallback_margin in config["routing"]["fallback_margin_candidates"]:
                    for verifier_margin in config["routing"]["verifier_margin_candidates"]:
                        policy = {
                            "primary_margin": float(primary_margin),
                            "fallback_margin": float(fallback_margin),
                            "verifier_margin": float(verifier_margin),
                        }
                        result = route_predictions(logits_192, logits_224, logits_vit, **policy)
                        metrics = _cascade_metrics(result, probe_192["labels"][probe_mask])
                        reject_result = route_predictions(
                            reject_logits_192,
                            reject_logits_224,
                            reject_logits_vit,
                            **policy,
                        )
                        selection_key = (
                            metrics["wrong_approved_count"],
                            metrics["top3_miss_count"],
                            -metrics["correct_approved_count"],
                            int(np.count_nonzero(reject_result["approved"])),
                            metrics["fallback_count"],
                            metrics["verifier_count"],
                            candidate_index,
                        )
                        candidates.append(
                            {
                                "shared_head": shared_head,
                                "verifier_head": verifier_head,
                                "policy": policy,
                                "metrics": metrics,
                                "reject_approved_count": int(
                                    np.count_nonzero(reject_result["approved"])
                                ),
                                "selection_key": list(selection_key),
                            }
                        )
                        candidate_index += 1
    return min(candidates, key=lambda row: tuple(row["selection_key"]))


def evaluate_oof(
    *, config: dict[str, Any], features: dict[str, dict[str, dict[str, np.ndarray]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_matrix = _training_matrix(features)
    probe_192 = features["convnext_192"]["probe"]
    probe_224 = features["convnext_224"]["probe"]
    vit_clean = features["vit_160"]["clean"]
    vit_probe = features["vit_160"]["probe"]
    reject_192 = features["convnext_192"]["reject"]
    reject_224 = features["convnext_224"]["reject"]
    vit_reject = features["vit_160"]["reject"]
    count = len(probe_192["labels"])
    class_count = int(config["dataset"]["class_count"])
    oof_192 = np.empty((count, class_count), dtype=np.float32)
    oof_224 = np.empty_like(oof_192)
    oof_vit = np.empty_like(oof_192)
    oof_scores = np.empty_like(oof_192)
    oof_approved = np.zeros(count, dtype=bool)
    oof_fallback = np.zeros(count, dtype=bool)
    oof_verifier = np.zeros(count, dtype=bool)
    reject_count = len(reject_192["labels"])
    reject_approved = np.zeros(reject_count, dtype=bool)
    fold_reports: list[dict[str, Any]] = []
    train_features, train_labels, train_folds = train_matrix

    for outer_fold in (0, 1, 2):
        inner_train = (outer_fold + 1) % 3
        inner_validation = (outer_fold + 2) % 3
        selected = _select_inner_policy(
            config=config,
            features=features,
            train_matrix=train_matrix,
            inner_train_fold=inner_train,
            inner_validation_fold=inner_validation,
        )
        train_mask = train_folds != outer_fold
        shared_weight, shared_bias = fit_small_sample_head(
            train_features[train_mask],
            train_labels[train_mask],
            candidate=selected["shared_head"],
            class_count=class_count,
        )
        verifier_mask = vit_clean["folds"] != outer_fold
        verifier_weight, verifier_bias = fit_small_sample_head(
            vit_clean["features"][verifier_mask],
            vit_clean["labels"][verifier_mask],
            candidate=selected["verifier_head"],
            class_count=class_count,
        )
        probe_mask = probe_192["folds"] == outer_fold
        logits_192 = predict_ridge(probe_192["features"][probe_mask], shared_weight, shared_bias)
        logits_224 = predict_ridge(probe_224["features"][probe_mask], shared_weight, shared_bias)
        logits_vit = predict_ridge(
            vit_probe["features"][probe_mask], verifier_weight, verifier_bias
        )
        result = route_predictions(logits_192, logits_224, logits_vit, **selected["policy"])
        oof_192[probe_mask] = logits_192
        oof_224[probe_mask] = logits_224
        oof_vit[probe_mask] = logits_vit
        oof_scores[probe_mask] = result["scores"]
        oof_approved[probe_mask] = result["approved"]
        oof_fallback[probe_mask] = result["fallback"]
        oof_verifier[probe_mask] = result["verifier"]

        reject_mask = reject_192["folds"] == outer_fold
        reject_result = route_predictions(
            predict_ridge(reject_192["features"][reject_mask], shared_weight, shared_bias),
            predict_ridge(reject_224["features"][reject_mask], shared_weight, shared_bias),
            predict_ridge(vit_reject["features"][reject_mask], verifier_weight, verifier_bias),
            **selected["policy"],
        )
        reject_approved[reject_mask] = reject_result["approved"]
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "inner_train_fold": inner_train,
                "inner_validation_fold": inner_validation,
                "selected": selected,
                "outer_metrics": _cascade_metrics(result, probe_192["labels"][probe_mask]),
                "outer_reject_approved_count": int(np.count_nonzero(reject_result["approved"])),
            }
        )

    combined = {
        "scores": oof_scores,
        "predictions": oof_scores.argmax(axis=1),
        "approved": oof_approved,
        "fallback": oof_fallback,
        "verifier": oof_verifier,
    }
    by_probe_kind = {}
    for index, name in enumerate(("appearance", "geometry", "context")):
        mask = probe_192["views"] == index
        scoped = {key: value[mask] for key, value in combined.items()}
        by_probe_kind[name] = _cascade_metrics(scoped, probe_192["labels"][mask])
    report = {
        "primary_192": _classification_metrics(oof_192, probe_192["labels"]),
        "detail_224": _classification_metrics(oof_224, probe_192["labels"]),
        "convnext_equal_fusion": _classification_metrics(
            _normalized_logits(oof_192) + _normalized_logits(oof_224),
            probe_192["labels"],
        ),
        "verifier_160": _classification_metrics(oof_vit, probe_192["labels"]),
        "selective_cascade": _cascade_metrics(combined, probe_192["labels"]),
        "selective_cascade_by_probe_kind": by_probe_kind,
        "synthetic_quality_reject": {
            "sample_count": reject_count,
            "approved_count": int(np.count_nonzero(reject_approved)),
            "not_approved_count": int(np.count_nonzero(~reject_approved)),
            "not_approved_rate": float(np.mean(~reject_approved)),
            "warning": "Synthetic quality variants are diagnostic proxies, not real recapture ground truth.",
        },
    }
    return report, fold_reports


def fit_final_package(
    *,
    config: dict[str, Any],
    features: dict[str, dict[str, dict[str, np.ndarray]]],
    folds: list[dict[str, Any]],
    output_dir: Path,
    source_manifest: Path,
    convnext_weights: Path,
    vit_weights: Path,
) -> dict[str, Any]:
    selected_text = [
        json.dumps(
            {
                "shared_head": row["selected"]["shared_head"],
                "verifier_head": row["selected"]["verifier_head"],
            },
            sort_keys=True,
        )
        for row in folds
    ]
    counts = Counter(selected_text)
    selected_heads = json.loads(
        min(selected_text, key=lambda value: (-counts[value], selected_text.index(value)))
    )
    policy = {
        key: float(np.median([row["selected"]["policy"][key] for row in folds]))
        for key in ("primary_margin", "fallback_margin", "verifier_margin")
    }
    train_features, train_labels, _ = _training_matrix(features)
    shared_weight, shared_bias = fit_small_sample_head(
        train_features,
        train_labels,
        candidate=selected_heads["shared_head"],
        class_count=20,
    )
    vit_clean = features["vit_160"]["clean"]
    verifier_weight, verifier_bias = fit_small_sample_head(
        vit_clean["features"],
        vit_clean["labels"],
        candidate=selected_heads["verifier_head"],
        class_count=20,
    )
    package_dir = output_dir / "classifier-package"
    package_dir.mkdir(parents=True, exist_ok=True)
    shared_path = package_dir / "convnext-shared-head.npz"
    verifier_path = package_dir / "vit160-verifier-head.npz"
    np.savez_compressed(shared_path, weight=shared_weight, bias=shared_bias)
    np.savez_compressed(verifier_path, weight=verifier_weight, bias=verifier_bias)
    package = {
        "schema_version": "1.0",
        "status": "EXPERIMENTAL_NOT_RUNTIME_ACTIVATED",
        "source_manifest_sha256": sha256_file(source_manifest),
        "training_image_root": "datasets/bix_bakery_dataset/single_objects only",
        "shared_convnext": {
            "input_sizes": [192, 224],
            "backbone_frozen": True,
            "backbone_weights_sha256": sha256_file(convnext_weights),
            "head": selected_heads["shared_head"],
            "head_file": shared_path.name,
            "head_sha256": sha256_file(shared_path),
        },
        "vit160_verifier": {
            "input_size": 160,
            "backbone_frozen": True,
            "backbone_weights_sha256": sha256_file(vit_weights),
            "head": selected_heads["verifier_head"],
            "head_file": verifier_path.name,
            "head_sha256": sha256_file(verifier_path),
        },
        "routing": policy,
        "limitations": [
            "No external image was used for fitting or evaluation.",
            "Physical item identities and independent capture sessions are unavailable.",
            "This package is not an ONNX Runtime/Catalog deployment bundle.",
        ],
    }
    _write_json(package_dir / "metadata.json", package)
    return package


def run(
    *,
    config_path: Path,
    source_dir: Path,
    convnext_weights: Path,
    vit_weights: Path,
    output_dir: Path,
    cpu: bool,
    reuse: bool,
) -> dict[str, Any]:
    config = load_json_config(config_path)
    validate_source_directory(config, source_dir)
    dataset_root = source_dir.parent.resolve()
    records = build_source_records(source_dir, dataset_root)
    source_manifest = output_dir / "source-manifest.jsonl"
    _write_jsonl(source_manifest, records)
    if not reuse:
        generate_and_extract(
            config=config,
            records=records,
            dataset_root=dataset_root,
            output_dir=output_dir,
            weights=convnext_weights,
            cpu=cpu,
        )
    features = prepare_cascade_features(
        config=config,
        records=records,
        dataset_root=dataset_root,
        output_dir=output_dir,
        convnext_weights=convnext_weights,
        vit_weights=vit_weights,
        cpu=cpu,
        reuse=reuse,
    )
    evaluation, folds = evaluate_oof(config=config, features=features)
    package = fit_final_package(
        config=config,
        features=features,
        folds=folds,
        output_dir=output_dir,
        source_manifest=source_manifest,
        convnext_weights=convnext_weights,
        vit_weights=vit_weights,
    )
    report = {
        "schema_version": "1.0",
        "experiment": config["experiment"],
        "data_contract": {
            "source_directory": str(source_dir.resolve()),
            "source_count": len(records),
            "class_count": 20,
            "source_content_unique_count": len({row["image_sha256"] for row in records}),
            "fold_source_counts": dict(
                sorted(Counter(int(row["fold"]) for row in records).items())
            ),
            "external_images_used": False,
            "derived_source_leakage_count": 0,
            "limitation": "View-held-out evaluation only; physical-item and capture-session independence cannot be measured.",
        },
        "training_contract": {
            "stored_positive_roi_count": 5000,
            "shared_convnext_training_feature_count": 10000,
            "shared_resolutions": [192, 224],
            "verifier_support_count": 200,
            "probe_count": 600,
            "synthetic_quality_reject_count": 800,
        },
        "evaluation": evaluation,
        "folds": folds,
        "final_experimental_package": package,
        "interpretation_guardrail": "Results are source-derived OOF diagnostics, not store generalization or zero-error certification.",
    }
    _write_json(output_dir / "cascade-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the BIX 192/224 selective classifier with a frozen ViT160 verifier"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--convnext-weights", type=Path, required=True)
    parser.add_argument("--vit-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run(
        config_path=args.config,
        source_dir=args.source_dir,
        convnext_weights=args.convnext_weights,
        vit_weights=args.vit_weights,
        output_dir=args.output_dir,
        cpu=args.cpu,
        reuse=args.reuse,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "cascade-report.json"),
                "selective_cascade": report["evaluation"]["selective_cascade"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
