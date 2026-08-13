from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from argparse import Namespace
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from ...configuration import load_json_config, resolve_config_path
from ...contracts.model_package import load_model_package, sha256_file
from ...pipeline.ports import Detection
from ...runtime.onnx import build_onnx_adapters
from ...training.data import read_manifest
from ...training.train_detector import detector_optimizer_recipe
from ...training.train_detector import train as train_detector
from ..rpc200.worker_gate import predict_records
from .selective import (
    TARGET_MODE_VERSION,
    DetectorPolicy,
    IndexedDetection,
    PolicyEvaluationCache,
    _indexed_nms,
    assert_no_split_leakage,
    curve_metrics,
    evaluate_policy,
    policy_grid,
    select_candidate,
    sha256_paths,
)

PHASES = (
    "prepare",
    "train",
    "cache",
    "select",
    "lock",
    "test",
    "export-package",
    "parity",
    "benchmark",
    "finalize",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    config = load_json_config(path)
    if config.get("experiment", {}).get("model_version") != TARGET_MODE_VERSION:
        raise ValueError("detector target config must use model_version 0.2.5")
    if tuple(config["experiment"].get("seeds", ())) != (
        20260812,
        20260813,
        20260814,
    ):
        raise ValueError("0.2.5 detector target mode requires the three locked seeds")
    if int(config["experiment"].get("fold_count", 0)) != 3:
        raise ValueError("0.2.5 detector target mode requires three folds")
    policies = policy_grid(config["policy_grid"])
    if not policies:
        raise ValueError("detector target policy grid is empty")
    selection = config["selection"]
    if float(selection["maximum_risk_upper_95"]) != 0.005:
        raise ValueError("0.2.5 maximum risk upper bound must be 0.5%")
    if float(selection["minimum_error_catch_recall"]) != 0.99:
        raise ValueError("0.2.5 error catch recall must be 99%")
    return config


def _records(path: Path, *, evaluation_set: str) -> list[dict[str, Any]]:
    values = read_manifest(path)
    result = []
    for value in values:
        if value.get("record_type") != "detection":
            continue
        row = dict(value)
        row["evaluation_set"] = evaluation_set
        row.setdefault("source", evaluation_set)
        if row.get("split") not in {"development", "test"}:
            raise ValueError(f"{evaluation_set} record requires development/test split")
        result.append(row)
    if not result:
        raise ValueError(f"{evaluation_set} manifest has no detection records")
    return result


def _group_values(record: dict[str, Any]) -> dict[str, Any]:
    annotations = record["annotations"]
    width = float(record["width"])
    height = float(record["height"])
    image_area = width * height
    groups = {
        "difficulty": record.get("difficulty", "UNKNOWN"),
        "object_count": min(7, len(annotations)),
        "capture_session": record.get("capture_session_id", "UNKNOWN"),
        "camera": record.get("camera", "UNKNOWN"),
        "store": record.get("store", "UNKNOWN"),
        "lighting": record.get("lighting", "UNKNOWN"),
        "blur": record.get("blur", "UNKNOWN"),
        "exposure": record.get("exposure", "UNKNOWN"),
        "novel_object": record.get("novel_object", False),
    }
    boxes = [annotation.get("bbox_xywh", annotation.get("bbox")) for annotation in annotations]
    groups["small_object"] = any(float(box[2]) * float(box[3]) / image_area < 0.01 for box in boxes)
    groups["border_contact"] = any(
        float(box[0]) <= 0
        or float(box[1]) <= 0
        or float(box[0]) + float(box[2]) >= width
        or float(box[1]) + float(box[3]) >= height
        for box in boxes
    )
    groups["overlap_or_occlusion"] = bool(record.get("overlap_or_occlusion", False))
    return groups


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    sets = {
        "natural": _records(args.natural_manifest, evaluation_set="natural"),
        "hard": _records(args.hard_manifest, evaluation_set="hard"),
        "shift": _records(args.shift_manifest, evaluation_set="shift"),
    }
    training_records = [
        dict(row, evaluation_set="training")
        for row in read_manifest(args.training_manifest)
        if row.get("record_type") == "detection"
    ]
    all_records = [*training_records, *[row for rows in sets.values() for row in rows]]
    assert_no_split_leakage(all_records)
    training_hashes = {
        str(row["image_sha256"])
        for row in training_records
        if row.get("image_sha256") not in (None, "")
    }
    for name in ("hard", "shift"):
        overlap = sorted(
            {
                str(row["image_sha256"])
                for row in sets[name]
                if row.get("image_sha256") in training_hashes
            }
        )
        if overlap:
            raise ValueError(
                f"{name} evaluation overlaps detector training by SHA-256: {len(overlap)}"
            )
    for rows in sets.values():
        for row in rows:
            row["groups"] = _group_values(row)
    metadata = {
        name: _load_json(path.parent / "metadata.json")
        for name, path in {
            "natural": args.natural_manifest,
            "hard": args.hard_manifest,
            "shift": args.shift_manifest,
        }.items()
    }
    required_independence = (
        "capture_session_id",
        "physical_target_group_id",
        "image_sha256",
        "perceptual_group_id",
    )
    missing = {
        name: {
            field: sum(row.get(field) in (None, "") for row in rows)
            for field in required_independence
        }
        for name, rows in sets.items()
    }
    report = {
        "schema_version": "1.0",
        "model_version": TARGET_MODE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "test_accessed": False,
        "manifest_sha256": sha256_paths(
            {
                "natural": args.natural_manifest,
                "hard": args.hard_manifest,
                "shift": args.shift_manifest,
            }
        ),
        "sets": {
            name: {
                "dataset_version": metadata[name].get("dataset_version"),
                "development_count": sum(row["split"] == "development" for row in rows),
                "test_count": sum(row["split"] == "test" for row in rows),
                "missing_independence_fields": missing[name],
                "promotion_evidence_ready": not any(missing[name].values()),
            }
            for name, rows in sets.items()
        },
    }
    prepared = args.output_dir / "prepared"
    for name, rows in sets.items():
        _write_jsonl(prepared / f"{name}.jsonl", rows)
    _write_json(prepared / "audit.json", report)


def _training_namespace(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    seed: int,
    fold: int,
    output_dir: Path,
    final_training: bool = False,
    epochs: int | None = None,
) -> Namespace:
    values = config["training"]
    return Namespace(
        manifest=args.training_manifest,
        dataset_root=args.training_dataset_root,
        output_dir=output_dir,
        fold=fold,
        final_training=final_training,
        cache_dir=args.detector_image_cache,
        pretrained_name=values["pretrained_name"],
        image_size=int(values["image_size"]),
        batch_size=int(values["batch_size"]),
        workers=int(values["workers"]),
        epochs=int(values["epochs"] if epochs is None else epochs),
        patience=int(values["patience"]),
        learning_rate=float(values["learning_rate"]),
        head_lr_multiplier=float(values["head_lr_multiplier"]),
        class_head_prior_probability=float(values["class_head_prior_probability"]),
        warmup_epochs=int(values["warmup_epochs"]),
        weight_decay=float(values["weight_decay"]),
        min_score_threshold=float(values["min_score_threshold"]),
        max_score_threshold=float(values["max_score_threshold"]),
        threshold_steps=int(values["threshold_steps"]),
        nms_iou_threshold=float(values["nms_iou_threshold"]),
        match_iou_threshold=0.5,
        target_recall=0.99,
        max_queries=int(config["policy_grid"]["max_queries"]),
        checkpoint_selection_mode="selective_image_risk",
        maximum_risk_upper_95=float(config["selection"]["maximum_risk_upper_95"]),
        uncertainty_score_threshold=0.20,
        uncertainty_min_area_ratio=0.039,
        uncertainty_match_iou_threshold=0.5,
        min_object_area_ratio=float(config["policy_grid"]["min_object_area_ratio"]),
        seed=seed,
        cpu=bool(args.cpu),
        resume=bool(args.resume),
    )


def _legacy_recipe_is_neutral_extension(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    neutral = {
        "freeze_mode": "none",
        "frozen_modules_eval": False,
        "skip_epoch_validation": False,
    }
    if any(current.get(key) != value for key, value in neutral.items()):
        return False
    legacy = dict(current)
    for key in (*neutral, "workers"):
        legacy.pop(key, None)
    return stored == legacy


def _migrate_neutral_resume_recipe(namespace: Namespace) -> None:
    progress_path = namespace.output_dir / "training_progress.pt"
    run_path = namespace.output_dir / "run.json"
    if not (namespace.resume and progress_path.is_file() and run_path.is_file()):
        return
    current = detector_optimizer_recipe(namespace)
    run = _load_json(run_path)
    stored = run.get("optimizer_recipe")
    if stored == current:
        return
    if not isinstance(stored, dict) or not _legacy_recipe_is_neutral_extension(stored, current):
        return
    import torch

    progress = torch.load(progress_path, map_location="cpu", weights_only=False)
    if progress.get("optimizer_recipe") != stored:
        raise ValueError("run and progress optimizer recipes differ before migration")
    run["optimizer_recipe"] = current
    _write_json(run_path, run)
    progress["optimizer_recipe"] = current
    temporary = progress_path.with_suffix(".tmp")
    torch.save(progress, temporary)
    temporary.replace(progress_path)


def train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    for seed in config["experiment"]["seeds"]:
        for fold in range(int(config["experiment"]["fold_count"])):
            output = args.output_dir / "detector" / f"seed-{seed}" / f"fold-{fold}"
            if (
                args.resume
                and (output / "history.json").is_file()
                and not (output / "training_progress.pt").is_file()
            ):
                continue
            namespace = _training_namespace(
                args,
                config,
                seed=int(seed),
                fold=fold,
                output_dir=output,
            )
            _migrate_neutral_resume_recipe(namespace)
            train_detector(namespace)


def _predict_oof(
    records: list[dict[str, Any]],
    checkpoint_root: Path,
    dataset_root: Path,
    *,
    batch_size: int,
    minimum_score: float,
    device_name: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for fold in range(3):
        subset = [row for row in records if int(row.get("fold", -1)) == fold]
        if not subset:
            continue
        predictions.extend(
            predict_records(
                checkpoint_root / f"fold-{fold}" / "best",
                subset,
                dataset_root,
                batch_size=batch_size,
                minimum_score=minimum_score,
                device_name=device_name,
            )
        )
    by_key = {str(row["sample_key"]): row for row in predictions}
    return [by_key[f"{row['source']}:{row['image_id']}"] for row in records]


def _model_specs(
    args: argparse.Namespace, config: dict[str, Any]
) -> list[tuple[str, int, Path, bool]]:
    return [
        ("baseline", 0, args.baseline_detector_checkpoint, True),
        *[
            (
                f"seed-{seed}",
                int(seed),
                args.output_dir / "detector" / f"seed-{seed}",
                False,
            )
            for seed in config["experiment"]["seeds"]
        ],
    ]


def _predict_model_development(
    records: list[dict[str, Any]],
    checkpoint: Path,
    dataset_root: Path,
    *,
    single_checkpoint: bool,
    batch_size: int,
    minimum_score: float,
    device_name: str,
) -> list[dict[str, Any]]:
    if not single_checkpoint:
        return _predict_oof(
            records,
            checkpoint,
            dataset_root,
            batch_size=batch_size,
            minimum_score=minimum_score,
            device_name=device_name,
        )
    predictions = predict_records(
        checkpoint,
        records,
        dataset_root,
        batch_size=batch_size,
        minimum_score=minimum_score,
        device_name=device_name,
    )
    by_key = {str(row["sample_key"]): row for row in predictions}
    return [by_key[f"{row['source']}:{row['image_id']}"] for row in records]


def _attach_classifier_outputs(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    classifier_package: Path,
    dataset_root: Path,
    provider: str,
    cuda_dll_dir: Path | None,
    nms_iou_thresholds: Iterable[float],
    classification_batch_size: int = 32,
) -> None:
    if classification_batch_size < 1:
        raise ValueError("classification_batch_size must be positive")
    package = load_model_package(classifier_package)
    _, classifier, _ = build_onnx_adapters(package, provider, cuda_dll_dir=cuda_dll_dir)
    labels = package.metadata.classifier.labels
    temperature = float(package.metadata.classifier.temperature)
    for record, prediction in zip(records, predictions):
        with Image.open(dataset_root / record["image_path"]) as source:
            image = source.convert("RGB")
        raw_detections = [
            Detection(*[float(value) for value in box], float(score))
            for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
        ]
        candidate_indices = _classification_candidate_indices(prediction, nms_iou_thresholds)
        detections = [raw_detections[index] for index in candidate_indices]
        if not detections:
            prediction["classifications"] = {}
            continue
        logits = (
            np.concatenate(
                [
                    classifier.classify(
                        image, detections[start : start + classification_batch_size]
                    )
                    for start in range(0, len(detections), classification_batch_size)
                ],
                axis=0,
            ).astype(np.float64)
            / temperature
        )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        orders = np.argsort(-probabilities, axis=1, kind="stable")
        prediction["classifications"] = {
            str(candidate_indices[index]): {
                "top1_class_id": labels[int(order[0])].class_id,
                "top3_class_ids": [labels[int(value)].class_id for value in order[:3]],
                "confidence": float(probabilities[index, int(order[0])]),
                "recapture": bool(labels[int(order[0])].recapture),
                "touches_border": (
                    detections[index].x1
                    <= int(record["width"]) * float(package.metadata.quality.border_margin_ratio)
                    or detections[index].y1
                    <= int(record["height"]) * float(package.metadata.quality.border_margin_ratio)
                    or detections[index].x2
                    >= int(record["width"])
                    * (1.0 - float(package.metadata.quality.border_margin_ratio))
                    or detections[index].y2
                    >= int(record["height"])
                    * (1.0 - float(package.metadata.quality.border_margin_ratio))
                ),
            }
            for index, order in enumerate(orders)
        }


def _classification_candidate_indices(
    prediction: dict[str, Any], nms_iou_thresholds: Iterable[float]
) -> list[int]:
    """Return every raw ROI that can survive one of the locked NMS policies."""
    thresholds = sorted({float(value) for value in nms_iou_thresholds})
    if not thresholds:
        raise ValueError("classification candidate selection requires NMS thresholds")
    raw = [
        IndexedDetection(
            index=index,
            detection=Detection(*[float(value) for value in box], float(score)),
        )
        for index, (box, score) in enumerate(zip(prediction["boxes_xyxy"], prediction["scores"]))
    ]
    selected = {item.index for threshold in thresholds for item in _indexed_nms(raw, threshold)}
    return sorted(selected)


def cache(args: argparse.Namespace, config: dict[str, Any]) -> None:
    minimum_score = min(float(value) for value in config["policy_grid"]["score_thresholds"])
    minimum_uncertainty = min(
        float(value)
        for value in config["policy_grid"]["uncertainty_score_thresholds"]
        if value is not None
    )
    minimum_score = min(minimum_score, minimum_uncertainty)
    prepared = args.output_dir / "prepared"
    for model_id, _, checkpoint, single_checkpoint in _model_specs(args, config):
        for name in ("natural", "hard"):
            cache_path = args.output_dir / "cache" / model_id / f"{name}-development.jsonl"
            if args.resume and cache_path.is_file():
                continue
            records = [
                row
                for row in read_manifest(prepared / f"{name}.jsonl")
                if row["split"] == "development"
            ]
            predictions = _predict_model_development(
                records,
                checkpoint,
                args.evaluation_dataset_root,
                single_checkpoint=single_checkpoint,
                batch_size=int(config["training"]["inference_batch_size"]),
                minimum_score=minimum_score,
                device_name="cpu" if args.cpu else "cuda",
            )
            _attach_classifier_outputs(
                records,
                predictions,
                classifier_package=args.classifier_package,
                dataset_root=args.evaluation_dataset_root,
                provider=args.provider,
                cuda_dll_dir=args.cuda_dll_dir,
                nms_iou_thresholds=config["policy_grid"]["nms_iou_thresholds"],
            )
            _write_jsonl(
                cache_path,
                predictions,
            )


def _candidate_summaries(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    package = load_model_package(args.classifier_package)
    approval_threshold = float(package.metadata.classifier.approval_threshold)
    prepared = args.output_dir / "prepared"
    natural_records = [
        row for row in read_manifest(prepared / "natural.jsonl") if row["split"] == "development"
    ]
    hard_records = [
        row for row in read_manifest(prepared / "hard.jsonl") if row["split"] == "development"
    ]
    policies = policy_grid(config["policy_grid"])
    candidates: list[dict[str, Any]] = []
    family_points: dict[tuple[str, str], list[dict[str, Any]]] = {}
    family_members: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model_id, seed, _, _ in _model_specs(args, config):
        natural_predictions = read_manifest(
            args.output_dir / "cache" / model_id / "natural-development.jsonl"
        )
        hard_predictions = read_manifest(
            args.output_dir / "cache" / model_id / "hard-development.jsonl"
        )
        natural_evaluation_cache = PolicyEvaluationCache(natural_predictions)
        hard_evaluation_cache = PolicyEvaluationCache(hard_predictions)
        for policy in policies:
            natural = evaluate_policy(
                natural_records,
                natural_predictions,
                policy,
                approval_threshold=approval_threshold,
                cache=natural_evaluation_cache,
            )
            hard = evaluate_policy(
                hard_records,
                hard_predictions,
                policy,
                approval_threshold=approval_threshold,
                cache=hard_evaluation_cache,
            )
            natural.pop("rows")
            hard.pop("rows")
            candidate = {
                "model_id": model_id,
                "seed": int(seed),
                "natural": natural,
                "hard": hard,
            }
            family_policy = asdict(policy) | {"score_threshold": None}
            family_key = json.dumps(family_policy, sort_keys=True)
            family = (model_id, family_key)
            point = dict(natural["metrics"])
            point["score_threshold"] = policy.score_threshold
            family_points.setdefault(family, []).append(point)
            family_members.setdefault(family, []).append(candidate)
            candidates.append(candidate)
    curves = []
    for family, members in family_members.items():
        curve = curve_metrics(family_points[family])
        curves.append(
            {
                "model_id": family[0],
                "family_policy": json.loads(family[1]),
                **curve,
                "failure_auroc_by_threshold": [
                    {
                        "score_threshold": member["natural"]["policy"]["score_threshold"],
                        "failure_auroc": member["natural"]["metrics"]["failure_auroc"],
                    }
                    for member in members
                ],
            }
        )
        for member in members:
            member["augrc"] = curve["augrc"]
            member["aurc"] = curve["aurc"]
    return candidates, sorted(
        curves,
        key=lambda value: (
            value["model_id"],
            json.dumps(value["family_policy"], sort_keys=True),
        ),
    )


def select(args: argparse.Namespace, config: dict[str, Any]) -> None:
    candidates, curves = _candidate_summaries(args, config)
    settings = config["selection"]
    decision = select_candidate(
        candidates,
        maximum_risk_upper=float(settings["maximum_risk_upper_95"]),
        minimum_error_catch_recall=float(settings["minimum_error_catch_recall"]),
        minimum_group_sample_count=int(settings["minimum_group_sample_count"]),
    )
    sweep_path = args.output_dir / "reports" / "development-policy-sweep.json"
    _write_json(
        sweep_path,
        {
            "schema_version": "1.0",
            "model_version": TARGET_MODE_VERSION,
            "selection_scope": "development_oof_only",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "curves": curves,
        },
    )
    decision["policy_sweep_sha256"] = sha256_file(sweep_path)
    _write_json(args.output_dir / "reports" / "development-selection.json", decision)


def _tree_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    if not files:
        raise ValueError(f"cannot hash empty checkpoint directory: {path}")
    for value in files:
        digest.update(value.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(value)))
    return digest.hexdigest()


def _best_epoch(path: Path) -> int:
    history = _load_json(path / "history.json")
    records = [row for row in history if row.get("detector_quality_key") is not None]
    if not records:
        raise ValueError(f"detector history has no validation quality: {path}")
    best = max(records, key=lambda row: tuple(row["detector_quality_key"]))
    return int(best["epoch"])


def lock(args: argparse.Namespace, config: dict[str, Any]) -> None:
    decision_path = args.output_dir / "reports" / "development-selection.json"
    decision = _load_json(decision_path)
    selected = decision["selected"]
    seed = int(selected["seed"])
    final_dir = args.output_dir / "detector" / "final"
    if selected["model_id"] == "baseline":
        epochs = None
        if (final_dir / "best").exists():
            if not args.resume:
                raise ValueError("final detector directory already exists")
        else:
            shutil.copytree(args.baseline_detector_checkpoint, final_dir / "best")
    else:
        epochs = round(
            statistics.median(
                _best_epoch(args.output_dir / "detector" / f"seed-{seed}" / f"fold-{fold}")
                for fold in range(3)
            )
        )
        if not (args.resume and (final_dir / "best" / "config.json").is_file()):
            train_detector(
                _training_namespace(
                    args,
                    config,
                    seed=seed,
                    fold=0,
                    output_dir=final_dir,
                    final_training=True,
                    epochs=epochs,
                )
            )
    files = {
        "config": args.config,
        "training_manifest": args.training_manifest,
        "natural_manifest": args.natural_manifest,
        "hard_manifest": args.hard_manifest,
        "shift_manifest": args.shift_manifest,
        "selection": decision_path,
        "policy_sweep": args.output_dir / "reports" / "development-policy-sweep.json",
        "classifier_metadata": args.classifier_package / "metadata.json",
        "classifier_onnx": load_model_package(args.classifier_package).classifier_path,
    }
    hashes = sha256_paths(files)
    tree_paths = {
        "detector_checkpoint": final_dir / "best",
        "development_cache": args.output_dir / "cache",
    }
    hashes.update({name: _tree_sha256(path) for name, path in tree_paths.items()})
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    payload = {
        "schema_version": "1.0",
        "model_version": TARGET_MODE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "test_accessed": False,
        "selected_seed": seed,
        "selected_policy": selected["natural"]["policy"],
        "final_training_epochs": epochs,
        "paths": {name: str(path) for name, path in files.items()},
        "tree_paths": {name: str(path) for name, path in tree_paths.items()},
        "hashes": hashes,
        "lock_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    _write_json(args.output_dir / "lock" / "pretest-lock.json", payload)


def verify_lock(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = args.output_dir / "lock" / "pretest-lock.json"
    lock = _load_json(lock_path)
    current = sha256_paths({name: Path(path) for name, path in lock["paths"].items()})
    current.update({name: _tree_sha256(Path(path)) for name, path in lock["tree_paths"].items()})
    if current != lock["hashes"]:
        raise ValueError("0.2.5 pretest lock changed before test evaluation")
    return lock


def _predict_final(
    records: list[dict[str, Any]], args: argparse.Namespace, config: dict[str, Any]
) -> list[dict[str, Any]]:
    minimum_score = min(
        [float(value) for value in config["policy_grid"]["score_thresholds"]]
        + [
            float(value)
            for value in config["policy_grid"]["uncertainty_score_thresholds"]
            if value is not None
        ]
    )
    predictions = predict_records(
        args.output_dir / "detector" / "final" / "best",
        records,
        args.evaluation_dataset_root,
        batch_size=int(config["training"]["inference_batch_size"]),
        minimum_score=minimum_score,
        device_name="cpu" if args.cpu else "cuda",
    )
    _attach_classifier_outputs(
        records,
        predictions,
        classifier_package=args.classifier_package,
        dataset_root=args.evaluation_dataset_root,
        provider=args.provider,
        cuda_dll_dir=args.cuda_dll_dir,
        nms_iou_thresholds=config["policy_grid"]["nms_iou_thresholds"],
    )
    return predictions


def test(args: argparse.Namespace, config: dict[str, Any]) -> None:
    lock = verify_lock(args)
    policy = DetectorPolicy(**lock["selected_policy"])
    package = load_model_package(args.classifier_package)
    reports = {}
    for name in ("natural", "hard", "shift"):
        records = [
            row
            for row in read_manifest(args.output_dir / "prepared" / f"{name}.jsonl")
            if row["split"] == "test"
        ]
        predictions = _predict_final(records, args, config)
        report = evaluate_policy(
            records,
            predictions,
            policy,
            approval_threshold=float(package.metadata.classifier.approval_threshold),
        )
        report.pop("rows")
        reports[name] = report
    audit = _load_json(args.output_dir / "prepared" / "audit.json")
    payload = {
        "schema_version": "1.0",
        "model_version": TARGET_MODE_VERSION,
        "lock_sha256": lock["lock_sha256"],
        "test_accessed": True,
        "data_evidence_ready": all(
            value["promotion_evidence_ready"] for value in audit["sets"].values()
        ),
        "sets": reports,
    }
    _write_json(args.output_dir / "reports" / "locked-test.json", payload)


def export_package(args: argparse.Namespace, config: dict[str, Any]) -> None:
    verify_lock(args)
    from .export import export_models

    audit = _load_json(args.output_dir / "prepared" / "audit.json")
    selected = _load_json(args.output_dir / "reports" / "development-selection.json")["selected"]
    metrics = selected["natural"]["metrics"]
    policy = selected["natural"]["policy"]
    detector_report = {
        "selected_score_threshold": policy["score_threshold"],
        "nms_iou_threshold": policy["nms_iou_threshold"],
        "target_recall_satisfied": metrics["object_diagnostics"]["recall"] >= 0.99,
        "metrics": {
            "recall": metrics["object_diagnostics"]["recall"],
            "precision": metrics["object_diagnostics"]["precision"],
            "count_accuracy": metrics["object_diagnostics"]["exact_count_accuracy"],
        },
    }
    detector_report_path = args.output_dir / "reports" / "detector-export.json"
    _write_json(detector_report_path, detector_report)
    export_models(
        Namespace(
            detector_checkpoint=args.output_dir / "detector" / "final" / "best",
            classifier_checkpoint=None,
            calibration_report=None,
            reuse_classifier_package=args.classifier_package,
            detector_evaluation_report=detector_report_path,
            manifest_metadata=args.classifier_manifest_metadata,
            output_dir=args.output_dir / "package",
            package_version=TARGET_MODE_VERSION,
            detector_version=TARGET_MODE_VERSION,
            classifier_version=TARGET_MODE_VERSION,
            relabel_reused_classifier=True,
            detector_size=int(config["training"]["image_size"]),
            uncertainty_score_threshold=policy["uncertainty_score_threshold"],
            uncertainty_min_area_ratio=policy["uncertainty_min_area_ratio"],
            uncertainty_match_iou_threshold=policy["uncertainty_match_iou_threshold"],
            crop_margin=0.05,
            resize_reducing_gap=1.0,
            classifier_warmup_max_batch=7,
            jpeg_draft_size=1500,
            min_object_area_ratio=policy["min_object_area_ratio"],
            border_margin_ratio=float(config["fixed_worker_policy"]["border_margin_ratio"]),
            border_policy=config["fixed_worker_policy"]["border_policy"],
            min_sharpness=None,
            min_mean_luminance=None,
            max_mean_luminance=None,
            opset=18,
            detector_target_provenance={
                "selection_report_sha256": sha256_file(
                    args.output_dir / "reports" / "development-selection.json"
                ),
                "classifier_source_version": load_model_package(
                    args.classifier_package
                ).metadata.package_version,
                "evaluation_dataset_versions": {
                    name: str(audit["sets"][name]["dataset_version"])
                    for name in ("natural", "hard", "shift")
                },
            },
        )
    )


def parity(args: argparse.Namespace, config: dict[str, Any]) -> None:
    verify_lock(args)
    paths = args.parity_report or []
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("parity phase requires existing parity reports")
    reports = [_load_json(path) for path in paths]
    required = (
        "pytorch_cpu_tolerance",
        "pytorch_cuda_tolerance",
        "cpu_cuda_tolerance",
        "top1_equal",
        "top3_set_and_order_equal",
        "final_state_equal",
    )
    strict_reports = [report for report in reports if isinstance(report.get("checks"), dict)]
    provider_reports = [report for report in reports if str(report.get("provider", "")).lower()]
    if len(reports) == 1 and strict_reports:
        checks = {name: bool(reports[0]["checks"].get(name)) for name in required}
    elif len(strict_reports) == 1 and len(provider_reports) == 2:
        strict = strict_reports[0]
        by_provider = {str(report["provider"]).lower(): report for report in provider_reports}
        if set(by_provider) != {"cpu", "cuda"}:
            raise ValueError("detector parity evidence requires CPU and CUDA reports")
        cpu = by_provider["cpu"]
        cuda = by_provider["cuda"]
        package_metadata_hashes = {
            report.get("package_artifact_sha256", {}).get("metadata.json") for report in reports
        }
        same_package = len(package_metadata_hashes) == 1 and None not in package_metadata_hashes
        detector_passes = bool(cpu.get("detector", {}).get("passes")) and bool(
            cuda.get("detector", {}).get("passes")
        )
        checks = {name: bool(strict["checks"].get(name)) for name in required}
        checks["pytorch_cpu_tolerance"] &= bool(cpu.get("passes"))
        checks["pytorch_cuda_tolerance"] &= bool(cuda.get("passes"))
        checks["cpu_cuda_tolerance"] &= same_package and detector_passes
        checks["final_state_equal"] &= same_package and detector_passes
    else:
        by_provider = {str(report.get("provider", "")).lower(): report for report in reports}
        if not {"cpu", "cuda"} <= set(by_provider):
            raise ValueError("generic parity evidence requires both CPU and CUDA reports")
        cpu = by_provider["cpu"]
        cuda = by_provider["cuda"]
        same_package = cpu.get("package_artifact_sha256") == cuda.get("package_artifact_sha256")
        checks = {
            "pytorch_cpu_tolerance": bool(cpu.get("passes")),
            "pytorch_cuda_tolerance": bool(cuda.get("passes")),
            "cpu_cuda_tolerance": bool(same_package and cpu.get("passes") and cuda.get("passes")),
            "top1_equal": bool(
                cpu.get("classifier", {}).get("top3_equal")
                and cuda.get("classifier", {}).get("top3_equal")
            ),
            "top3_set_and_order_equal": bool(
                cpu.get("classifier", {}).get("top3_equal")
                and cuda.get("classifier", {}).get("top3_equal")
            ),
            "final_state_equal": bool(
                cpu.get("classifier", {}).get("status_equal")
                and cuda.get("classifier", {}).get("status_equal")
                and cpu.get("detector", {}).get("passes")
                and cuda.get("detector", {}).get("passes")
            ),
        }
    _write_json(
        args.output_dir / "reports" / "parity-gate.json",
        {
            "sources": [str(path) for path in paths],
            "passed": all(bool(checks.get(name)) for name in required),
            "checks": {name: bool(checks.get(name)) for name in required},
        },
    )


def benchmark(args: argparse.Namespace, config: dict[str, Any]) -> None:
    verify_lock(args)
    if args.benchmark_report is None or not args.benchmark_report.is_file():
        raise ValueError("benchmark phase requires an existing RTX 5080 benchmark report")
    report = _load_json(args.benchmark_report)
    full_path = report["by_path"]["full_path"]
    _write_json(
        args.output_dir / "reports" / "benchmark-gate.json",
        {
            "source": str(args.benchmark_report),
            "sample_count": int(full_path["sample_count"]),
            "p50_ms": float(full_path["p50_ms"]),
            "p95_ms": float(full_path["p95_ms"]),
            "p99_ms": float(full_path["p99_ms"]),
            "passed": float(full_path["p95_ms"])
            <= float(config["promotion"]["maximum_full_path_p95_ms"]),
        },
    )


def finalize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    lock = verify_lock(args)
    test_report = _load_json(args.output_dir / "reports" / "locked-test.json")
    parity_gate = _load_json(args.output_dir / "reports" / "parity-gate.json")
    benchmark_gate = _load_json(args.output_dir / "reports" / "benchmark-gate.json")
    natural = test_report["sets"]["natural"]["metrics"]
    hard = test_report["sets"]["hard"]["metrics"]
    unknown_top3 = natural.get("unknown_top3_accuracy")
    checks = {
        "independent_data_ready": bool(test_report["data_evidence_ready"]),
        "detector_pass_risk_u95": float(natural["detector_pass_risk_upper_95"]) <= 0.005,
        "e2e_approved_risk_u95": float(natural["e2e_approved_risk_upper_95"]) <= 0.005,
        "detector_silent_failure_zero": int(natural["gate_table"]["silent_failure"]) == 0,
        "e2e_approved_error_zero": int(natural["approved_error_count"]) == 0,
        "hard_error_catch_recall": hard.get("error_catch_recall") is not None
        and float(hard["error_catch_recall"]) >= 0.99,
        "unknown_top3_accuracy": unknown_top3 is not None and float(unknown_top3) >= 0.95,
        "parity": bool(parity_gate["passed"]),
        "full_path_latency": bool(benchmark_gate["passed"]),
    }
    passed = all(checks.values())
    report = {
        "schema_version": "1.0",
        "model_version": TARGET_MODE_VERSION,
        "lock_sha256": lock["lock_sha256"],
        "promotion_status": "production" if passed else "experiment_only",
        "manual_waiver_allowed": False,
        "passed": passed,
        "checks": checks,
        "failures": [name for name, value in checks.items() if not value],
        "natural": natural,
        "hard": hard,
        "shift": test_report["sets"]["shift"]["metrics"],
        "benchmark": benchmark_gate,
        "parity": parity_gate,
    }
    _write_json(args.output_dir / "reports" / "final-promotion.json", report)
    package_metadata_path = args.output_dir / "package" / "metadata.json"
    if passed and package_metadata_path.is_file():
        metadata = _load_json(package_metadata_path)
        metadata["promotion_status"] = "production"
        metadata["promotion"] = {
            "decision": "approved",
            "method": "all_gates",
            "decided_on": datetime.now(UTC).date().isoformat(),
            "waivers": [],
            "remaining_limitations": [],
        }
        _write_json(package_metadata_path, metadata)
        load_model_package(package_metadata_path.parent)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 0.2.5 detector safety-first target workflow"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-dataset-root", type=Path, required=True)
    parser.add_argument("--natural-manifest", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--shift-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dataset-root", type=Path, required=True)
    parser.add_argument("--classifier-package", type=Path, required=True)
    parser.add_argument("--baseline-detector-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-manifest-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detector-image-cache", type=Path)
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-dll-dir", type=Path)
    parser.add_argument("--parity-report", type=Path, action="append")
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--phase", choices=("all",) + PHASES, default="all")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.config = resolve_config_path(args.config)
    config = _load_config(args.config)
    phases = PHASES if args.phase == "all" else (args.phase,)
    phase_functions = {
        "prepare": prepare,
        "train": train,
        "cache": cache,
        "select": select,
        "lock": lock,
        "test": test,
        "export-package": export_package,
        "parity": parity,
        "benchmark": benchmark,
        "finalize": finalize,
    }
    for phase in phases:
        phase_functions[phase](args, config)


if __name__ == "__main__":
    main()
