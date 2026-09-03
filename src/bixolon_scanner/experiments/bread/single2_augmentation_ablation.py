from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from ...configuration import load_json_config
from ...runtime.onnx import prepare_rgb
from ...training.classifier_allowlist import sha256_file
from ...training.models import require_torch
from ...training.synthetic_roi import (
    ClutterRoiRecipe,
    DirectRoiRecipe,
    augment_clutter_roi,
    augment_direct_roi,
    prepare_direct_roi_source,
)
from .classifier_200_only import (
    MEAN,
    STD,
    _build_model,
    _classification_metrics,
    _extract_features,
    _prepare_neighbor_masked_clutter,
    _selection_key,
    fit_small_sample_head,
    predict_ridge,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_view_fold(record: dict[str, Any]) -> int:
    """Hold out source views, never individual derived images."""
    view = str(record["view"])
    side = str(record["side"])
    if view.startswith("ground_0") or view == "vertical":
        return 0
    if view == "ground_30_dir_01" or (view == "ground_60_dir_01" and side == "normal"):
        return 1
    if view == "ground_30_dir_02" or (view == "ground_60_dir_01" and side == "flipped"):
        return 2
    raise ValueError(f"unmapped source view: side={side}, view={view}")


def _load_source_records(manifest_path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda row: (int(row["category_id"]), str(row["image_path"])))
    if len(rows) != 200:
        raise ValueError(f"single_objects_2 manifest must contain 200 rows, found {len(rows)}")
    if Counter(int(row["category_id"]) for row in rows) != Counter(
        {key: 10 for key in range(1, 21)}
    ):
        raise ValueError("single_objects_2 must contain 20 classes x 10 source images")
    hashes = [str(row["image_sha256"]) for row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("single_objects_2 contains duplicate image content")
    for row in rows:
        if Path(str(row["image_path"])).parts[0] != "single_objects_2":
            raise ValueError("manifest contains a fitting-forbidden source")
        path = dataset_root / str(row["image_path"])
        if not path.is_file() or sha256_file(path) != row["image_sha256"]:
            raise ValueError(f"source checksum mismatch: {row['image_path']}")
        row["fold"] = source_view_fold(row)
        row["target"] = int(row["category_id"]) - 1
    return rows


def _direct_recipe(values: dict[str, Any], output_size: int) -> DirectRoiRecipe:
    fields = DirectRoiRecipe.__dataclass_fields__
    recipe = DirectRoiRecipe(
        output_size=output_size,
        **{key: value for key, value in values.items() if key in fields and key != "output_size"},
    )
    recipe.validate()
    return recipe


def _clutter_recipe(values: dict[str, Any], output_size: int) -> ClutterRoiRecipe:
    fields = ClutterRoiRecipe.__dataclass_fields__
    recipe = ClutterRoiRecipe(
        output_size=output_size,
        **{key: value for key, value in values.items() if key in fields and key != "output_size"},
    )
    recipe.validate()
    return recipe


def _save_generated(
    image: Image.Image,
    *,
    path: Path,
    record: dict[str, Any],
    bank: str,
    view_index: int,
    seed: int,
    provenance: dict[str, Any],
    train_role: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return {
        "schema_version": "1.0",
        "bank": bank,
        "train_role": train_role,
        "view_index": view_index,
        "seed": seed,
        "derived_path": path.as_posix(),
        "derived_sha256": sha256_file(path),
        "source_path": record["image_path"],
        "source_sha256": record["image_sha256"],
        "category_id": record["category_id"],
        "class_id": record["class_id"],
        "target": record["target"],
        "fold": record["fold"],
        "provenance": provenance,
    }


def _quality_reject(
    image: Image.Image, *, seed: int, mode: int, size: int
) -> tuple[Image.Image, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    value = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (size, size))
    name: str
    parameters: dict[str, Any]
    if mode == 0:
        radius = float(rng.uniform(3.0, 5.5))
        value = value.filter(ImageFilter.GaussianBlur(radius))
        name, parameters = "strong_blur", {"radius": radius}
    elif mode == 1:
        factor = float(rng.choice([rng.uniform(0.18, 0.38), rng.uniform(1.75, 2.15)]))
        value = ImageEnhance.Brightness(value).enhance(factor)
        value = ImageEnhance.Contrast(value).enhance(float(rng.uniform(0.4, 0.7)))
        name, parameters = "severe_exposure", {"brightness": factor}
    elif mode == 2:
        width = int(rng.uniform(0.38, 0.55) * size)
        height = int(rng.uniform(0.35, 0.52) * size)
        left = int(rng.integers((size - width) // 4, max((size - width) * 3 // 4, 1)))
        top = int(rng.integers((size - height) // 4, max((size - height) * 3 // 4, 1)))
        draw = ImageDraw.Draw(value)
        neutral = int(rng.integers(175, 231))
        draw.rectangle((left, top, left + width, top + height), fill=(neutral,) * 3)
        name, parameters = "heavy_occlusion", {"box": [left, top, left + width, top + height]}
    else:
        edge = int(rng.integers(0, 4))
        loss = int(rng.uniform(0.22, 0.34) * size)
        boxes = [
            (loss, 0, size, size),
            (0, 0, size - loss, size),
            (0, loss, size, size),
            (0, 0, size, size - loss),
        ]
        value = value.crop(boxes[edge]).resize((size, size), Image.Resampling.BICUBIC)
        name, parameters = "edge_content_loss", {"edge": edge, "loss_pixels": loss}
    return value, {"mode": name, "parameters": parameters}


def _tensor(image: Image.Image, output_size: int) -> np.ndarray:
    return prepare_rgb(image, (output_size, output_size), MEAN, STD, reducing_gap=1.0)


def generate_and_extract(
    *,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    weights: Path,
    cpu: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    generation = config["generation"]
    size = int(generation["output_size"])
    views = int(generation["train_views_per_source_per_bank"])
    seed_base = int(config["experiment"]["seed"])
    appearance = _direct_recipe(config["banks"]["appearance"], size)
    geometry = _direct_recipe(config["banks"]["geometry"], size)
    context = _clutter_recipe(config["banks"]["context"], size)
    cutout_recipe = DirectRoiRecipe(
        output_size=size,
        crop_mode="border_connected_composite",
        border_color_distance=42,
        mask_feather_radius=0.8,
    )

    images: list[Image.Image] = []
    cutouts: list[Image.Image] = []
    for record in records:
        with Image.open(dataset_root / str(record["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
        images.append(image)
        cutouts.append(prepare_direct_roi_source(image, cutout_recipe))

    tensors: dict[str, list[np.ndarray]] = {
        key: [] for key in ("clean", "appearance", "geometry", "context", "probe", "reject")
    }
    metadata: dict[str, list[int]] = {
        key: []
        for key in (
            "clean_labels",
            "clean_folds",
            "clean_views",
            "appearance_labels",
            "appearance_folds",
            "appearance_views",
            "geometry_labels",
            "geometry_folds",
            "geometry_views",
            "context_labels",
            "context_folds",
            "context_views",
            "probe_labels",
            "probe_folds",
            "probe_views",
            "reject_labels",
            "reject_folds",
            "reject_views",
        )
    }
    provenance_rows: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ("appearance", "geometry", "context", "probe", "reject")
    }

    def add_meta(bank: str, record: dict[str, Any], view: int) -> None:
        metadata[f"{bank}_labels"].append(int(record["target"]))
        metadata[f"{bank}_folds"].append(int(record["fold"]))
        metadata[f"{bank}_views"].append(view)

    for source_index, record in enumerate(records):
        tensors["clean"].append(_tensor(images[source_index], size))
        add_meta("clean", record, 0)
        distractors = [
            (cutouts[index], str(other["image_sha256"]), int(other["category_id"]))
            for index, other in enumerate(records)
            if int(other["category_id"]) != int(record["category_id"])
        ]
        for bank_index, (bank, recipe) in enumerate(
            (("appearance", appearance), ("geometry", geometry))
        ):
            for view in range(views):
                seed = seed_base + bank_index * 100_000_019 + source_index * 100_003 + view
                sample = augment_direct_roi(
                    images[source_index],
                    source_sha256=str(record["image_sha256"]),
                    category_id=int(record["category_id"]),
                    seed=seed,
                    recipe=recipe,
                    prepared_cutout=cutouts[source_index],
                )
                path = (
                    output_dir
                    / "derived"
                    / bank
                    / str(record["class_id"])
                    / f"s{source_index:03d}_v{view:02d}.png"
                )
                provenance_rows[bank].append(
                    _save_generated(
                        sample.image,
                        path=path,
                        record=record,
                        bank=bank,
                        view_index=view,
                        seed=seed,
                        provenance=sample.provenance,
                        train_role="sku_positive",
                    )
                )
                tensors[bank].append(_tensor(sample.image, size))
                add_meta(bank, record, view)
        for view in range(views):
            seed = seed_base + 200_000_033 + source_index * 100_003 + view
            sample = augment_clutter_roi(
                cutouts[source_index],
                target_sha256=str(record["image_sha256"]),
                target_category_id=int(record["category_id"]),
                distractors=distractors,
                seed=seed,
                recipe=context,
            )
            path = (
                output_dir
                / "derived"
                / "context"
                / str(record["class_id"])
                / f"s{source_index:03d}_v{view:02d}.png"
            )
            provenance_rows["context"].append(
                _save_generated(
                    sample.image,
                    path=path,
                    record=record,
                    bank="context",
                    view_index=view,
                    seed=seed,
                    provenance={
                        **sample.provenance,
                        "target_bbox_xyxy": list(sample.bbox_xyxy),
                        "model_preprocessing": "neighbor_ownership_mask",
                    },
                    train_role="sku_positive",
                )
            )
            tensors["context"].append(_prepare_neighbor_masked_clutter(sample))
            add_meta("context", record, view)

        probe_recipes: list[tuple[str, DirectRoiRecipe | ClutterRoiRecipe]] = [
            ("appearance", appearance),
            ("geometry", geometry),
            ("context", context),
        ]
        for view, (probe_kind, recipe) in enumerate(probe_recipes):
            seed = seed_base + 300_000_049 + source_index * 100_003 + view
            if isinstance(recipe, DirectRoiRecipe):
                sample = augment_direct_roi(
                    images[source_index],
                    source_sha256=str(record["image_sha256"]),
                    category_id=int(record["category_id"]),
                    seed=seed,
                    recipe=recipe,
                    prepared_cutout=cutouts[source_index],
                )
                probe_tensor = _tensor(sample.image, size)
            else:
                sample = augment_clutter_roi(
                    cutouts[source_index],
                    target_sha256=str(record["image_sha256"]),
                    target_category_id=int(record["category_id"]),
                    distractors=distractors,
                    seed=seed,
                    recipe=recipe,
                )
                probe_tensor = _prepare_neighbor_masked_clutter(sample)
                sample.provenance["target_bbox_xyxy"] = list(sample.bbox_xyxy)
            path = (
                output_dir
                / "derived"
                / "probe"
                / str(record["class_id"])
                / f"s{source_index:03d}_{probe_kind}.png"
            )
            provenance_rows["probe"].append(
                _save_generated(
                    sample.image,
                    path=path,
                    record=record,
                    bank="probe",
                    view_index=view,
                    seed=seed,
                    provenance={**sample.provenance, "probe_kind": probe_kind},
                    train_role="validation_only",
                )
            )
            tensors["probe"].append(probe_tensor)
            add_meta("probe", record, view)

        for view in range(int(generation["reject_views_per_source"])):
            seed = seed_base + 400_000_057 + source_index * 100_003 + view
            reject, reject_provenance = _quality_reject(
                images[source_index], seed=seed, mode=view, size=size
            )
            path = (
                output_dir
                / "derived"
                / "reject"
                / str(record["class_id"])
                / f"s{source_index:03d}_v{view:02d}.png"
            )
            provenance_rows["reject"].append(
                _save_generated(
                    reject,
                    path=path,
                    record=record,
                    bank="reject",
                    view_index=view,
                    seed=seed,
                    provenance=reject_provenance,
                    train_role="quality_safety_only_not_sku_positive",
                )
            )
            tensors["reject"].append(_tensor(reject, size))
            add_meta("reject", record, view)

    for bank, rows in provenance_rows.items():
        _write_jsonl(output_dir / "manifests" / f"{bank}.jsonl", rows)

    torch = require_torch()
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model = _build_model(weights).to(device).eval()
    batch_size = int(config["training"]["feature_batch_size"])
    features: dict[str, dict[str, np.ndarray]] = {}
    cache_dir = output_dir / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for bank, bank_tensors in tensors.items():
        feature = _extract_features(model, bank_tensors, device=device, batch_size=batch_size)
        labels = np.asarray(metadata[f"{bank}_labels"], dtype=np.int64)
        folds = np.asarray(metadata[f"{bank}_folds"], dtype=np.int64)
        view_indices = np.asarray(metadata[f"{bank}_views"], dtype=np.int64)
        features[bank] = {
            "features": feature,
            "labels": labels,
            "folds": folds,
            "views": view_indices,
        }
        np.savez_compressed(
            cache_dir / f"{bank}.npz",
            features=feature,
            labels=labels,
            folds=folds,
            views=view_indices,
        )
    summary = {
        "device": str(device),
        "backbone_weight_sha256": sha256_file(weights),
        "bank_counts": {bank: len(value["labels"]) for bank, value in features.items()},
        "fold_source_counts": dict(sorted(Counter(int(row["fold"]) for row in records).items())),
        "recipes": {
            "appearance": asdict(appearance),
            "geometry": asdict(geometry),
            "context": asdict(context),
        },
    }
    _write_json(output_dir / "generation-summary.json", summary)
    return features, summary


def load_feature_cache(
    output_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    features: dict[str, dict[str, np.ndarray]] = {}
    for bank in ("clean", "appearance", "geometry", "context", "probe", "reject"):
        path = output_dir / "features" / f"{bank}.npz"
        if not path.is_file():
            raise ValueError(f"missing feature cache: {path}")
        values = np.load(path)
        features[bank] = {key: values[key] for key in values.files}
    summary_path = output_dir / "generation-summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing generation summary: {summary_path}")
    return features, json.loads(summary_path.read_text(encoding="utf-8"))


def _variant_cache(
    features: dict[str, dict[str, np.ndarray]], selectors: list[str]
) -> dict[str, np.ndarray]:
    selected_features: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    selected_folds: list[np.ndarray] = []
    for selector in selectors:
        bank, _, modifier = selector.partition(":")
        values = features[bank]
        mask = np.ones(len(values["labels"]), dtype=bool)
        repeat = 1
        if modifier == "first4":
            mask = values["views"] < 4
        elif modifier.startswith("repeat") and modifier[6:].isdigit():
            repeat = int(modifier[6:])
            if repeat < 1:
                raise ValueError(f"invalid repeat selector: {selector}")
        elif modifier:
            raise ValueError(f"unsupported bank selector: {selector}")
        selected_features.append(np.repeat(values["features"][mask], repeat, axis=0))
        selected_labels.append(np.repeat(values["labels"][mask], repeat, axis=0))
        selected_folds.append(np.repeat(values["folds"][mask], repeat, axis=0))
    return {
        "train_features": np.concatenate(selected_features),
        "train_labels": np.concatenate(selected_labels),
        "train_folds": np.concatenate(selected_folds),
        "validation_features": features["probe"]["features"],
        "validation_labels": features["probe"]["labels"],
        "validation_folds": features["probe"]["folds"],
    }


def _nested_oof(
    cache: dict[str, np.ndarray], candidates: list[dict[str, Any]], class_count: int
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    oof = np.empty((len(cache["validation_labels"]), class_count), dtype=np.float32)
    reports: list[dict[str, Any]] = []
    selected: list[str] = []
    for outer in (0, 1, 2):
        inner_train = (outer + 1) % 3
        inner_validation = (outer + 2) % 3
        candidate_reports = []
        for index, candidate in enumerate(candidates):
            mask = cache["train_folds"] == inner_train
            weight, bias = fit_small_sample_head(
                cache["train_features"][mask],
                cache["train_labels"][mask],
                candidate=candidate,
                class_count=class_count,
            )
            probe_mask = cache["validation_folds"] == inner_validation
            logits = predict_ridge(cache["validation_features"][probe_mask], weight, bias)
            candidate_reports.append(
                {
                    "head": candidate,
                    "metrics": _classification_metrics(
                        logits, cache["validation_labels"][probe_mask]
                    ),
                    "selection_key": list(
                        _selection_key(logits, cache["validation_labels"][probe_mask], index)
                    ),
                }
            )
        winner = min(candidate_reports, key=lambda value: tuple(value["selection_key"]))
        winner_head = dict(winner["head"])
        selected.append(json.dumps(winner_head, sort_keys=True))
        train_mask = cache["train_folds"] != outer
        weight, bias = fit_small_sample_head(
            cache["train_features"][train_mask],
            cache["train_labels"][train_mask],
            candidate=winner_head,
            class_count=class_count,
        )
        probe_mask = cache["validation_folds"] == outer
        logits = predict_ridge(cache["validation_features"][probe_mask], weight, bias)
        oof[probe_mask] = logits
        reports.append(
            {
                "outer_fold": outer,
                "selected_head": winner_head,
                "candidate_reports": candidate_reports,
                "metrics": _classification_metrics(logits, cache["validation_labels"][probe_mask]),
            }
        )
    counts = Counter(selected)
    candidate_text = [json.dumps(value, sort_keys=True) for value in candidates]
    final_text = min(
        candidate_text, key=lambda value: (-counts[value], candidate_text.index(value))
    )
    return oof, reports, json.loads(final_text)


def _difficulty_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    eval_records: list[dict[str, Any]],
    annotation_path: Path,
) -> dict[str, Any]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    difficulty_by_image = {
        int(row["id"]): Path(str(row["file_name"])).parent.name.upper()
        for row in annotation["images"]
    }
    result = {"ALL": _classification_metrics(logits, labels)}
    for difficulty in ("EASY", "MEDIUM", "HARD"):
        mask = np.asarray(
            [difficulty_by_image[int(row["image_id"])] == difficulty for row in eval_records],
            dtype=bool,
        )
        result[difficulty] = _classification_metrics(logits[mask], labels[mask])
    return result


def run_experiment(
    *,
    config_path: Path,
    dataset_root: Path,
    source_manifest: Path,
    weights: Path,
    evaluation_features_path: Path,
    evaluation_records_path: Path,
    annotation_path: Path,
    output_dir: Path,
    cpu: bool,
    reuse_features: bool = False,
) -> dict[str, Any]:
    config = load_json_config(config_path)
    if sha256_file(source_manifest) != config["dataset"]["manifest_sha256"]:
        raise ValueError("source manifest checksum differs from the experiment lock")
    records = _load_source_records(source_manifest, dataset_root)
    if reuse_features:
        features, generation = load_feature_cache(output_dir)
        if generation["backbone_weight_sha256"] != sha256_file(weights):
            raise ValueError("cached backbone weight checksum differs from the requested weight")
    else:
        features, generation = generate_and_extract(
            config=config,
            records=records,
            dataset_root=dataset_root,
            output_dir=output_dir,
            weights=weights,
            cpu=cpu,
        )
    eval_features = np.load(evaluation_features_path).astype(np.float32)
    eval_records = [
        json.loads(line)
        for line in evaluation_records_path.read_text(encoding="utf-8").splitlines()
    ]
    eval_labels = np.asarray([int(row["target"]) for row in eval_records], dtype=np.int64)
    if len(eval_features) != len(eval_labels):
        raise ValueError("development evaluation feature/record counts differ")

    candidates = [dict(value) for value in config["training"]["head_candidates"]]
    variants: dict[str, Any] = {}
    for name, selectors in config["training"]["variants"].items():
        cache = _variant_cache(features, list(selectors))
        oof, folds, final_head = _nested_oof(
            cache, candidates, int(config["dataset"]["class_count"])
        )
        weight, bias = fit_small_sample_head(
            cache["train_features"],
            cache["train_labels"],
            candidate=final_head,
            class_count=int(config["dataset"]["class_count"]),
        )
        eval_logits = predict_ridge(eval_features, weight, bias)
        reject_logits = predict_ridge(features["reject"]["features"], weight, bias)
        reject_margin = (
            np.sort(reject_logits, axis=1)[:, -1] - np.sort(reject_logits, axis=1)[:, -2]
        )
        variants[name] = {
            "selectors": selectors,
            "training_sample_count": int(len(cache["train_labels"])),
            "final_head": final_head,
            "source_grouped_probe_oof": _classification_metrics(oof, cache["validation_labels"]),
            "source_grouped_probe_folds": folds,
            "multi_object_development_diagnostic": _difficulty_metrics(
                eval_logits, eval_labels, eval_records, annotation_path
            ),
            "synthetic_reject_diagnostic": {
                "sample_count": len(reject_margin),
                "median_top1_top2_logit_margin": float(np.median(reject_margin)),
                "p95_top1_top2_logit_margin": float(np.quantile(reject_margin, 0.95)),
                "warning": "Synthetic rejects have inherited SKU content and are not approval-threshold ground truth.",
            },
        }
    ranking = sorted(
        variants,
        key=lambda name: (
            variants[name]["source_grouped_probe_oof"]["top3_miss_count"],
            variants[name]["source_grouped_probe_oof"]["top1_error_count"],
            variants[name]["training_sample_count"],
        ),
    )
    report = {
        "schema_version": "1.0",
        "experiment": config["experiment"],
        "data_contract": {
            "fitting_source": "single_objects_2 only",
            "source_count": len(records),
            "source_manifest_sha256": sha256_file(source_manifest),
            "all_derived_images_inherit_source_fold": True,
            "same_physical_item_limitation": "Each class is represented by one physical-item lineage; folds test held-out views, not unseen items.",
            "multi_object_used_for_fitting_or_head_selection": False,
            "multi_object_role": "post-selection development diagnostic only",
        },
        "generation": generation,
        "variants": variants,
        "ranking": ranking,
        "recommended_variant": ranking[0],
        "interpretation_guardrail": "This ablation does not certify zero FP/FN or store-level generalization.",
    }
    _write_json(output_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablate independent augmentation banks for bread single_objects_2"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--evaluation-records", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_experiment(
        config_path=args.config,
        dataset_root=args.dataset_root,
        source_manifest=args.source_manifest,
        weights=args.weights,
        evaluation_features_path=args.evaluation_features,
        evaluation_records_path=args.evaluation_records,
        annotation_path=args.annotations,
        output_dir=args.output_dir,
        cpu=args.cpu,
        reuse_features=args.reuse_features,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "recommended_variant": report["recommended_variant"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
