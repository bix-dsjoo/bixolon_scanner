from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from ..inference import Detection, _box_iou, _nms
from ..package import sha256_file
from .evaluate_detector import _iou, _metrics, _xywh_to_xyxy
from .train_detector import detector_optimizer_recipe
from .train_detector import train as train_detector

SCHEMA_VERSION = "1.0"


def _reject_post_test_mutation(args: argparse.Namespace, operation: str) -> None:
    experiment_path = args.output_dir / "prepared" / "experiment.json"
    if not experiment_path.is_file():
        return
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("test_accessed") is True:
        raise RuntimeError(
            f"post-test output is immutable; {operation} requires a fresh output directory"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_canonical_json(value) + "\n" for value in values), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _records_sha256(records: Iterable[dict[str, Any]]) -> str:
    payload = "".join(_canonical_json(record) + "\n" for record in records)
    return hashlib.sha256(payload.encode()).hexdigest()


def _prediction_metadata_path(predictions_path: Path) -> Path:
    return predictions_path.with_name(predictions_path.name + ".metadata.json")


def _prediction_identity(
    records: list[dict[str, Any]],
    checkpoints: Iterable[Path],
    *,
    source_sha256: str,
    inference_config: dict[str, Any],
) -> dict[str, Any]:
    weights = {
        str(index): sha256_file(_detector_weights_path(checkpoint))
        for index, checkpoint in enumerate(checkpoints)
    }
    identity = {
        "source_sha256": source_sha256,
        "records_sha256": _records_sha256(records),
        "checkpoint_weights_sha256": weights,
        "inference_config_fingerprint": hashlib.sha256(
            _canonical_json(inference_config).encode()
        ).hexdigest(),
        "record_count": len(records),
        "records": records,
    }
    lineage = {key: value for key, value in inference_config.items() if key.endswith("_sha256")}
    if lineage:
        identity["lineage"] = dict(sorted(lineage.items()))
    return identity


def _prediction_artifact_valid(predictions_path: Path, identity: dict[str, Any]) -> bool:
    metadata_path = _prediction_metadata_path(predictions_path)
    if not predictions_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("complete") is not True:
            return False
        if any(metadata.get(key) != value for key, value in identity.items() if key != "records"):
            return False
        if metadata.get("predictions_sha256") != sha256_file(predictions_path):
            return False
        predictions = _read_jsonl(predictions_path)
        sample_keys = [str(item["sample_key"]) for item in predictions]
        expected_keys = [
            f"{record['source']}:{record['image_id']}" for record in identity["records"]
        ]
        return (
            len(predictions) == int(identity["record_count"])
            and len(sample_keys) == len(set(sample_keys))
            and set(sample_keys) == set(expected_keys)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_prediction_artifact(
    predictions_path: Path,
    predictions: list[dict[str, Any]],
    identity: dict[str, Any],
) -> None:
    _write_jsonl(predictions_path, predictions)
    public_identity = {key: value for key, value in identity.items() if key != "records"}
    _write_json(
        _prediction_metadata_path(predictions_path),
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            **public_identity,
            "predictions_sha256": sha256_file(predictions_path),
        },
    )


def checkout_group(filename: str) -> str:
    stem = Path(filename).stem
    token = stem.rsplit("-", 1)[-1]
    if not token.isdigit():
        raise ValueError(f"checkout filename has no terminal group id: {filename}")
    return token


def assign_oof_folds(records: list[dict[str, Any]], fold_count: int) -> dict[str, int]:
    """Balance capture groups by annotation classes and difficulty without leakage."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["capture_session_id"])].append(record)
    fold_totals = [0] * fold_count
    fold_levels = [Counter() for _ in range(fold_count)]
    fold_classes = [Counter() for _ in range(fold_count)]
    assignment: dict[str, int] = {}
    stats: list[tuple[str, int, Counter[int], Counter[str]]] = []
    for group_id, group_records in grouped.items():
        classes: Counter[int] = Counter()
        levels: Counter[str] = Counter()
        for record in group_records:
            levels[str(record["level"])] += 1
            classes.update(int(annotation["category_id"]) for annotation in record["annotations"])
        stats.append((group_id, sum(classes.values()), classes, levels))

    for group_id, total, classes, levels in sorted(stats, key=lambda row: (-row[1], row[0])):

        def cost(fold: int) -> tuple[float, int]:
            class_cost = sum(
                (fold_classes[fold][key] + value) ** 2 for key, value in classes.items()
            )
            level_cost = sum((fold_levels[fold][key] + value) ** 2 for key, value in levels.items())
            return fold_totals[fold] + total + 0.01 * class_cost + 0.1 * level_cost, fold

        selected = min(range(fold_count), key=cost)
        assignment[group_id] = selected
        fold_totals[selected] += total
        fold_classes[selected].update(classes)
        fold_levels[selected].update(levels)
    if set(assignment) != set(grouped):
        raise RuntimeError("failed to assign every capture group")
    return assignment


def build_rpc_detector_manifest(
    dataset_root: Path, detector_dir: Path, fold_count: int
) -> list[dict[str, Any]]:
    payload = json.loads((dataset_root / "instances_val2019.json").read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[int(annotation["image_id"])].append(
            {
                "annotation_id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "bbox_xywh": [float(value) for value in annotation["bbox"]],
                "area": float(annotation["area"]),
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )
    records: list[dict[str, Any]] = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        records.append(
            {
                "record_type": "detection",
                "source": "rpc_val2019",
                "image_path": f"val2019/{image['file_name']}",
                "image_id": image_id,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "capture_session_id": checkout_group(str(image["file_name"])),
                "level": str(image["level"]),
                "split": "development",
                "fold": None,
                "role": None,
                "annotations": sorted(by_image[image_id], key=lambda row: row["annotation_id"]),
            }
        )
    folds = assign_oof_folds(records, fold_count)
    for record in records:
        record["fold"] = folds[str(record["capture_session_id"])]
    lines = [_canonical_json(record) for record in records]
    manifest_path = detector_dir / "manifest" / "manifest.jsonl"
    _write_jsonl(manifest_path, records)
    _write_json(
        manifest_path.parent / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": "rpc-detector-"
            + hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()[:12],
            "fold_count": fold_count,
            "record_count": len(records),
            "source_sha256": sha256_file(dataset_root / "instances_val2019.json"),
            "manifest_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        },
    )
    return records


def assign_validation_roles(
    records: list[dict[str, Any]], category_count: int, seed: int
) -> dict[str, str]:
    """Assign calibration/selection inside every OOF fold while keeping groups intact."""
    result: dict[str, str] = {}
    folds = sorted({int(record["fold"]) for record in records})
    for fold in folds:
        subset = [record for record in records if int(record["fold"]) == fold]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in subset:
            grouped[str(record["capture_session_id"])].append(record)
        dimensions = category_count + 3
        totals = np.zeros(dimensions, dtype=np.float64)
        vectors: dict[str, np.ndarray] = {}
        for group_id, values in grouped.items():
            vector = np.zeros(dimensions, dtype=np.float64)
            for record in values:
                vector[category_count + ("easy", "medium", "hard").index(str(record["level"]))] += 1
                for annotation in record["annotations"]:
                    vector[int(annotation["category_id"]) - 1] += 1
            vectors[group_id] = vector
            totals += vector
        target = totals / 2.0
        normalizer = np.maximum(target, 1.0)
        order = sorted(
            grouped,
            key=lambda group: (
                -len(grouped[group]),
                hashlib.sha256(f"{seed}:{fold}:{group}".encode()).hexdigest(),
            ),
        )
        allocated = {"calibration": np.zeros(dimensions), "selection": np.zeros(dimensions)}
        counts = {"calibration": 0, "selection": 0}
        limits = {"calibration": math.ceil(len(order) / 2), "selection": len(order) // 2}
        for group_id in order:
            scores: dict[str, float] = {}
            for side, other in (("calibration", "selection"), ("selection", "calibration")):
                if counts[side] >= limits[side]:
                    scores[side] = float("inf")
                    continue
                proposed = allocated[side] + vectors[group_id]
                scores[side] = float((((proposed - target) / normalizer) ** 2).mean())
                scores[side] += float((((allocated[other] - target) / normalizer) ** 2).mean())
            selected = min(scores, key=lambda side: (scores[side], side))
            result[group_id] = selected
            allocated[selected] += vectors[group_id]
            counts[selected] += 1
    return result


def _detector_namespace(
    options: dict[str, Any],
    manifest: Path,
    dataset_root: Path,
    output_dir: Path,
    fold: int,
    *,
    resume: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        dataset_root=dataset_root,
        output_dir=output_dir,
        fold=fold,
        final_training=False,
        cache_dir=None,
        pretrained_name=str(options["pretrained_name"]),
        image_size=int(options["image_size"]),
        batch_size=int(options["batch_size"]),
        workers=int(options.get("workers", 0)),
        epochs=int(options["epochs"]),
        patience=int(options["patience"]),
        learning_rate=float(options["learning_rate"]),
        head_lr_multiplier=float(options.get("head_lr_multiplier", 10.0)),
        class_head_prior_probability=float(options.get("class_head_prior_probability", 0.01)),
        warmup_epochs=int(options.get("warmup_epochs", 3)),
        weight_decay=float(options.get("weight_decay", 1e-4)),
        min_score_threshold=float(options["min_score_threshold"]),
        max_score_threshold=float(options["max_score_threshold"]),
        threshold_steps=int(options["threshold_steps"]),
        nms_iou_threshold=float(options["nms_iou_threshold"]),
        match_iou_threshold=float(options["match_iou_threshold"]),
        target_recall=float(options["target_recall"]),
        max_queries=int(options["max_queries"]),
        # The one-class heads are randomly reinitialized by Transformers. Keep
        # that initialization identical across OOF folds so fold assignment is
        # the only intended training difference.
        seed=int(options["seed"]),
        cpu=False,
        resume=resume,
        initial_checkpoint=None,
        initial_checkpoint_sha256=None,
        training_identity=None,
        fixed_epoch_checkpoint=False,
        skip_epoch_validation=bool(options.get("skip_epoch_validation", False)),
    )


def _domain_adaptation_namespace(
    options: dict[str, Any],
    manifest: Path,
    dataset_root: Path,
    output_dir: Path,
    fold: int,
    initial_checkpoint: Path,
    training_identity: dict[str, Any],
    *,
    resume: bool,
) -> argparse.Namespace:
    adaptation = dict(options["domain_adaptation"])
    adapted_options = dict(options)
    adapted_options.update(
        {
            "epochs": int(adaptation["epochs"]),
            "patience": int(adaptation["patience"]),
            "learning_rate": float(adaptation["learning_rate"]),
            "seed": int(adaptation["seed"]),
            "head_lr_multiplier": float(
                adaptation.get("head_lr_multiplier", options.get("head_lr_multiplier", 10.0))
            ),
            "weight_decay": float(adaptation.get("weight_decay", 0.0)),
            "workers": int(adaptation.get("workers", options.get("workers", 0))),
        }
    )
    namespace = _detector_namespace(
        adapted_options,
        manifest,
        dataset_root,
        output_dir,
        fold,
        resume=resume,
    )
    namespace.initial_checkpoint = initial_checkpoint
    namespace.initial_checkpoint_sha256 = sha256_file(_detector_weights_path(initial_checkpoint))
    namespace.training_identity = dict(training_identity)
    namespace.fixed_epoch_checkpoint = True
    namespace.freeze_mode = str(adaptation.get("freeze_mode", "classification_heads_only"))
    namespace.frozen_modules_eval = bool(adaptation.get("frozen_modules_eval", True))
    namespace.skip_epoch_validation = bool(adaptation.get("skip_epoch_validation", False))
    return namespace


def _detector_weights_path(checkpoint: Path) -> Path:
    candidates = [
        path
        for path in (checkpoint / "model.safetensors", checkpoint / "pytorch_model.bin")
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"detector checkpoint weights are missing or ambiguous: {checkpoint}"
        )
    return candidates[0]


def _checkpoint_complete(
    path: Path,
    *,
    expected_seed: int | None = None,
    expected_optimizer_recipe: dict[str, Any] | None = None,
) -> bool:
    marker_path = path / "complete.json"
    history_path = path / "history.json"
    run_path = path / "run.json"
    if (
        not history_path.is_file()
        or not run_path.is_file()
        or not (path / "best" / "config.json").is_file()
    ):
        return False
    if not marker_path.is_file() and not (path / "training_progress.pt").is_file():
        try:
            _mark_checkpoint_complete(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        weights = _detector_weights_path(path / "best")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        fixed_epochs_complete = (
            expected_optimizer_recipe is None
            or not bool(expected_optimizer_recipe.get("fixed_epoch_checkpoint", False))
            or len(history) == int(expected_optimizer_recipe["total_epochs"])
        )
        return (
            marker.get("complete") is True
            and marker.get("weights_sha256") == sha256_file(weights)
            and int(marker.get("history_epochs", -1)) == len(history)
            and fixed_epochs_complete
            and (
                expected_seed is None
                or int(run.get("arguments", {}).get("seed", -1)) == expected_seed
            )
            and (
                expected_optimizer_recipe is None
                or (
                    marker.get("optimizer_recipe") == expected_optimizer_recipe
                    and run.get("optimizer_recipe") == expected_optimizer_recipe
                )
            )
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _mark_checkpoint_complete(
    path: Path, *, optimizer_recipe: dict[str, Any] | None = None
) -> None:
    history = json.loads((path / "history.json").read_text(encoding="utf-8"))
    metric_rows = [row for row in history if isinstance(row.get("detector_quality_key"), list)]
    best_metric_row = (
        history[-1]
        if optimizer_recipe is not None
        and bool(optimizer_recipe.get("fixed_epoch_checkpoint", False))
        and history
        else max(
            metric_rows,
            key=lambda row: (list(row["detector_quality_key"]), -int(row["epoch"])),
            default=None,
        )
    )
    _write_json(
        path / "complete.json",
        {
            "complete": True,
            "completed_at": datetime.now(UTC).isoformat(),
            "history_epochs": len(history),
            "weights_sha256": sha256_file(_detector_weights_path(path / "best")),
            "optimizer_recipe": optimizer_recipe,
            "best_epoch": None if best_metric_row is None else int(best_metric_row["epoch"]),
            "best_detector_metrics": (
                None if best_metric_row is None else best_metric_row.get("detector_metrics")
            ),
            "selected_score_threshold": (
                None if best_metric_row is None else best_metric_row.get("selected_score_threshold")
            ),
            "target_recall_satisfied": (
                None if best_metric_row is None else best_metric_row.get("target_recall_satisfied")
            ),
        },
    )


def _detector_phase_complete(
    detector_dir: Path,
    dataset_root: Path | None = None,
    config: dict[str, Any] | None = None,
    *,
    marker_path: Path | None = None,
) -> bool:
    marker_path = marker_path or detector_dir / "complete.json"
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        artifact_root = _active_detector_artifact_root(detector_dir, marker)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    artifacts = {
        "threshold_sha256": artifact_root / "threshold.json",
        "val_predictions_sha256": artifact_root / "predictions" / "val_oof.jsonl",
        "val_predictions_metadata_sha256": _prediction_metadata_path(
            artifact_root / "predictions" / "val_oof.jsonl"
        ),
        "train_predictions_sha256": artifact_root / "predictions" / "train_assigned.jsonl",
        "train_predictions_metadata_sha256": _prediction_metadata_path(
            artifact_root / "predictions" / "train_assigned.jsonl"
        ),
    }
    if marker.get("adaptation_manifest_sha256") is not None:
        artifacts["adaptation_manifest_sha256"] = artifact_root / "manifest" / "manifest.jsonl"
        artifacts["adaptation_manifest_metadata_sha256"] = (
            artifact_root / "manifest" / "metadata.json"
        )
    if marker.get("target_oof_gate_sha256") is not None:
        artifacts["target_oof_gate_sha256"] = artifact_root / "target_oof_gate.json"
    if marker.get("progressive_fold0_gate_sha256") is not None:
        artifacts["progressive_fold0_gate_sha256"] = (
            artifact_root / "progressive-gate" / "fold0.json"
        )
    try:
        hashes_valid = all(
            path.is_file() and marker.get(key) == sha256_file(path)
            for key, path in artifacts.items()
        )
        if not hashes_valid or dataset_root is None or config is None:
            return hashes_valid
        options = config["detector"]
        if marker.get("checkpoint_set") == "domain_adaptation":
            adaptation = options.get("domain_adaptation")
            if (
                not isinstance(adaptation, dict)
                or marker.get("adaptation_config_sha256")
                != hashlib.sha256(_canonical_json(adaptation).encode()).hexdigest()
            ):
                return False
            baseline_path = detector_dir / "baseline" / "complete.json"
            if (
                not baseline_path.is_file()
                or marker.get("baseline_complete_sha256") != sha256_file(baseline_path)
                or not _detector_phase_complete(
                    detector_dir,
                    dataset_root,
                    config,
                    marker_path=baseline_path,
                )
            ):
                return False
            threshold = json.loads((artifact_root / "threshold.json").read_text(encoding="utf-8"))
            if (
                threshold.get("selection_threshold_policy") != "frozen_calibration_threshold"
                or threshold.get("selection_score_threshold")
                != threshold.get("selected_score_threshold")
                or threshold.get("frozen_threshold_selection_gate") is not True
                or marker.get("selection_threshold_policy")
                != threshold.get("selection_threshold_policy")
                or marker.get("selection_score_threshold")
                != threshold.get("selection_score_threshold")
                or marker.get("selection_raw_bbox_recall")
                != threshold.get("selection_metrics", {}).get("recall")
                or marker.get("selection_target_recall_satisfied") is not True
                or marker.get("frozen_threshold_selection_gate") is not True
                or marker.get("target_oof_gate", {}).get("passes") is not True
                or marker.get("progressive_fold0_gate", {}).get("passes") is not True
            ):
                return False
        fold_count = int(options["fold_count"])
        manifest = detector_dir / "manifest" / "manifest.jsonl"
        if not _manifest_valid(manifest, dataset_root, fold_count=fold_count):
            return False
        checkpoint_set = str(marker.get("checkpoint_set", "baseline"))
        checkpoints = (
            _detector_checkpoint_set(detector_dir, fold_count, checkpoint_set)
            if checkpoint_set == "baseline"
            else [artifact_root / "folds" / f"fold{fold}" / "best" for fold in range(fold_count)]
        )
        inference_config = {
            "batch_size": int(options["inference_batch_size"]),
            "minimum_score": float(options["min_score_threshold"]),
        }
        val_records = _read_jsonl(manifest)
        val_identity = _prediction_identity(
            val_records,
            checkpoints,
            source_sha256=sha256_file(dataset_root / "instances_val2019.json"),
            inference_config={
                **inference_config,
                "partition": str(marker.get("val_prediction_partition", "validation_oof")),
            },
        )
        train_records = _train_records(
            dataset_root,
            fold_count=fold_count,
            fold_assignment=str(marker.get("train_fold_assignment", "sample_hash")),
        )
        train_identity = _prediction_identity(
            train_records,
            checkpoints,
            source_sha256=sha256_file(dataset_root / "instances_train2019.json"),
            inference_config={
                **inference_config,
                "partition": str(marker.get("train_prediction_partition", "train_assigned")),
            },
        )
        return _prediction_artifact_valid(
            artifact_root / "predictions" / "val_oof.jsonl", val_identity
        ) and _prediction_artifact_valid(
            artifact_root / "predictions" / "train_assigned.jsonl", train_identity
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _baseline_detector_complete(
    detector_dir: Path, dataset_root: Path, config: dict[str, Any]
) -> bool:
    """Validate the immutable baseline marker, migrating only a valid legacy baseline."""
    baseline_path = detector_dir / "baseline" / "complete.json"
    if baseline_path.is_file():
        return _detector_phase_complete(
            detector_dir, dataset_root, config, marker_path=baseline_path
        )
    legacy_path = detector_dir / "complete.json"
    if not legacy_path.is_file():
        return False
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if str(legacy.get("checkpoint_set", "baseline")) != "baseline" or not (
        _detector_phase_complete(detector_dir, dataset_root, config)
    ):
        return False
    migrated = {
        **legacy,
        "checkpoint_set": "baseline",
        "artifact_root": ".",
        "contract": "rpc-detector-baseline-complete-v1",
    }
    _write_json(baseline_path, migrated)
    return _detector_phase_complete(detector_dir, dataset_root, config, marker_path=baseline_path)


def _detector_checkpoint_set(
    detector_dir: Path, fold_count: int, checkpoint_set: str
) -> list[Path]:
    if checkpoint_set == "baseline":
        root = detector_dir / "folds"
    elif checkpoint_set == "domain_adaptation":
        root = detector_dir / "domain-adaptation" / "folds"
    else:
        raise ValueError(f"unsupported detector checkpoint set: {checkpoint_set}")
    return [root / f"fold{fold}" / "best" for fold in range(fold_count)]


def _active_detector_artifact_root(
    detector_dir: Path, marker: dict[str, Any] | None = None
) -> Path:
    if marker is None:
        marker = json.loads((detector_dir / "complete.json").read_text(encoding="utf-8"))
    relative = str(marker.get("artifact_root", "."))
    if relative == ".":
        return detector_dir
    path = (detector_dir / relative).resolve()
    if detector_dir.resolve() not in path.parents:
        raise ValueError("detector artifact root escapes detector directory")
    return path


def _manifest_valid(manifest: Path, dataset_root: Path, *, fold_count: int) -> bool:
    metadata_path = manifest.parent / "metadata.json"
    if not manifest.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            int(metadata.get("fold_count", -1)) == fold_count
            and metadata.get("source_sha256")
            == sha256_file(dataset_root / "instances_val2019.json")
            and metadata.get("manifest_sha256") == sha256_file(manifest)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def train_oof_detectors(
    options: dict[str, Any], manifest: Path, dataset_root: Path, detector_dir: Path, *, resume: bool
) -> None:
    for fold in range(int(options["fold_count"])):
        output = detector_dir / "folds" / f"fold{fold}"
        namespace = _detector_namespace(
            options, manifest, dataset_root, output, fold, resume=resume
        )
        recipe = detector_optimizer_recipe(namespace)
        if resume and _checkpoint_complete(
            output,
            expected_seed=int(namespace.seed),
            expected_optimizer_recipe=recipe,
        ):
            print(json.dumps({"skipped_complete_detector_fold": fold}), flush=True)
            continue
        train_detector(namespace)
        _mark_checkpoint_complete(output, optimizer_recipe=recipe)


def _prediction_fold(record: dict[str, Any]) -> int:
    value = record.get("prediction_fold")
    if value is None:
        value = record.get("fold")
    return -1 if value is None else int(value)


def predict_records(
    checkpoint: Path,
    records: list[dict[str, Any]],
    dataset_root: Path,
    *,
    batch_size: int,
    minimum_score: float = 0.05,
    device_name: str = "cuda",
) -> list[dict[str, Any]]:
    from .models import require_torch

    torch = require_torch()
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    device = torch.device(
        device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = RTDetrV2ForObjectDetection.from_pretrained(checkpoint).to(device).eval()
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images: list[Image.Image] = []
        sizes: list[list[int]] = []
        for record in batch:
            with Image.open(dataset_root / record["image_path"]) as source:
                images.append(source.convert("RGB"))
            sizes.append([int(record["height"]), int(record["width"])])
        inputs = {
            key: value.to(device)
            for key, value in processor(images=images, return_tensors="pt").items()
        }
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            outputs = model(**inputs)
        processed = processor.post_process_object_detection(
            outputs,
            threshold=float(minimum_score),
            target_sizes=torch.asarray(sizes, device=device),
        )
        for record, result in zip(batch, processed):
            predictions.append(
                {
                    "sample_key": f"{record['source']}:{record['image_id']}",
                    "image_id": int(record["image_id"]),
                    "fold_model": _prediction_fold(record),
                    "boxes_xyxy": result["boxes"].float().cpu().numpy().tolist(),
                    "scores": result["scores"].float().cpu().numpy().tolist(),
                }
            )
        if start and start % (batch_size * 100) == 0:
            print(json.dumps({"detector_predictions": start, "total": len(records)}), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def _match(detections: list[Detection], annotations: list[dict[str, Any]], threshold: float):
    gt = [_xywh_to_xyxy(annotation["bbox_xywh"]) for annotation in annotations]
    remaining = set(range(len(gt)))
    matches: dict[int, tuple[int, float]] = {}
    for detection_index, detection in sorted(
        enumerate(detections), key=lambda item: item[1].score, reverse=True
    ):
        box = np.asarray([detection.x1, detection.y1, detection.x2, detection.y2], dtype=np.float32)
        candidates = [(index, _iou(box, gt[index])) for index in remaining]
        if candidates:
            index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= threshold:
                remaining.remove(index)
                matches[detection_index] = (index, overlap)
    return matches, remaining


def postprocess_worker_gate(
    record: dict[str, Any], prediction: dict[str, Any], options: dict[str, Any]
) -> dict[str, Any]:
    score_threshold = float(options["score_threshold"])
    width, height = int(record["width"]), int(record["height"])
    raw = [
        Detection(*[float(value) for value in box], float(score))
        for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
    ]
    accepted_raw = [item for item in raw if item.score >= score_threshold]
    capacity = len(accepted_raw) >= int(options["max_queries"])
    detections = _nms(accepted_raw, float(options["nms_iou_threshold"]))
    uncertain = 0
    uncertainty_threshold = options.get("uncertainty_score_threshold")
    if uncertainty_threshold is not None:
        shadow = _nms(
            [item for item in raw if item.score >= float(uncertainty_threshold)],
            float(options["nms_iou_threshold"]),
        )
        for candidate in shadow:
            if candidate.score >= score_threshold:
                continue
            area_ratio = (
                (candidate.x2 - candidate.x1)
                * (candidate.y2 - candidate.y1)
                / float(width * height)
            )
            if area_ratio < float(options["uncertainty_min_area_ratio"]):
                continue
            overlaps = [_box_iou(candidate, accepted) for accepted in detections]
            if not overlaps or max(overlaps) < float(options["uncertainty_match_iou_threshold"]):
                uncertain += 1
    reasons: list[str] = []
    if capacity:
        reasons.append("DETECTOR_CAPACITY_EXCEEDED")
    if not detections:
        reasons.append("DETECTOR_NO_OBJECT")
    image_area = float(width * height)
    if any(
        (item.x2 - item.x1) * (item.y2 - item.y1) / image_area
        < float(options["min_object_area_ratio"])
        for item in detections
    ):
        reasons.append("DETECTOR_OBJECT_TOO_SMALL")
    if uncertain:
        reasons.append("DETECTOR_UNCERTAIN_OBJECT")
    matches, missed = _match(
        detections, record["annotations"], float(options["match_iou_threshold"])
    )
    return {
        "detections": [
            {"bbox_xyxy": [item.x1, item.y1, item.x2, item.y2], "score": item.score}
            for item in detections
        ],
        "matches": {str(key): [value[0], value[1]] for key, value in matches.items()},
        "missed_annotation_indices": sorted(missed),
        "unmatched_detection_indices": sorted(set(range(len(detections))) - set(matches)),
        "recapture_reasons": list(dict.fromkeys(reasons)),
        "uncertain_candidate_count": uncertain,
    }


def select_detector_threshold(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]], options: dict[str, Any]
) -> dict[str, Any]:
    by_key = {str(item["sample_key"]): item for item in predictions}
    calibration = [record for record in records if record["role"] == "calibration"]
    ordered_predictions = [
        by_key[f"{record['source']}:{record['image_id']}"] for record in calibration
    ]
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(
        float(options["min_score_threshold"]),
        float(options["max_score_threshold"]),
        int(options["threshold_steps"]),
    ):
        metrics = _metrics(
            calibration,
            ordered_predictions,
            score_threshold=float(threshold),
            nms_iou_threshold=float(options["nms_iou_threshold"]),
            match_iou_threshold=float(options["match_iou_threshold"]),
            max_queries=int(options["max_queries"]),
        )
        metrics["score_threshold"] = float(threshold)
        candidates.append(metrics)
    eligible = [item for item in candidates if item["recall"] >= float(options["target_recall"])]
    selected = max(
        eligible if eligible else candidates,
        key=(
            (lambda item: (item["count_accuracy"], item["precision"], item["score_threshold"]))
            if eligible
            else (lambda item: (item["recall"], item["count_accuracy"], item["precision"]))
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "threshold_policy": "calibration_oof_only",
        "selected_score_threshold": selected["score_threshold"],
        "target_recall": float(options["target_recall"]),
        "target_recall_satisfied": selected["recall"] >= float(options["target_recall"]),
        "calibration_metrics": selected,
    }


def evaluate_frozen_detector_threshold_selection(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold_report: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate selection recall at the calibration-selected threshold only."""
    by_key = {str(item["sample_key"]): item for item in predictions}
    selection = [record for record in records if record["role"] == "selection"]
    if not selection:
        raise ValueError("detector selection partition is empty")
    selection_predictions = [
        by_key[f"{record['source']}:{record['image_id']}"] for record in selection
    ]
    selected_threshold = float(threshold_report["selected_score_threshold"])
    metrics = _metrics(
        selection,
        selection_predictions,
        score_threshold=selected_threshold,
        nms_iou_threshold=float(options["nms_iou_threshold"]),
        match_iou_threshold=float(options["match_iou_threshold"]),
        max_queries=int(options["max_queries"]),
    )
    metrics["score_threshold"] = selected_threshold
    target_recall = float(options["target_recall"])
    passed = float(metrics["recall"]) >= target_recall
    return {
        **threshold_report,
        "selection_threshold_policy": "frozen_calibration_threshold",
        "selection_score_threshold": selected_threshold,
        "selection_metrics": metrics,
        "selection_target_recall_satisfied": passed,
        "frozen_threshold_selection_gate": passed,
    }


def _train_records(
    dataset_root: Path,
    *,
    fold_count: int = 3,
    fold_assignment: str = "sample_hash",
) -> list[dict[str, Any]]:
    if fold_assignment not in {"sample_hash", "physical_group"}:
        raise ValueError(f"unsupported train detector fold assignment: {fold_assignment}")
    payload = json.loads((dataset_root / "instances_train2019.json").read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    records = []
    for annotation in payload["annotations"]:
        image = images[int(annotation["image_id"])]
        filename = str(image["file_name"])
        digest = hashlib.sha256(f"train:{image['id']}".encode()).digest()
        physical_group = filename.split("_camera", 1)[0].removesuffix("-back")
        record = {
            "record_type": "detection",
            "source": "rpc_train2019",
            "image_path": f"train2019/{filename}",
            "image_id": int(image["id"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "prediction_fold": int.from_bytes(digest[:4], "big") % fold_count,
            "annotations": [
                {
                    "annotation_id": int(annotation["id"]),
                    "category_id": int(annotation["category_id"]),
                    "bbox_xywh": [float(value) for value in annotation["bbox"]],
                    "area": float(annotation["area"]),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            ],
        }
        if fold_assignment == "physical_group":
            record["physical_group"] = physical_group
        records.append(record)
    if fold_assignment == "physical_group":
        group_counts = Counter(str(record["physical_group"]) for record in records)
        fold_load = [0] * fold_count
        group_fold: dict[str, int] = {}
        for group in sorted(
            group_counts,
            key=lambda value: (
                -group_counts[value],
                hashlib.sha256(f"physical-group:{value}".encode()).hexdigest(),
            ),
        ):
            fold = min(range(fold_count), key=lambda value: (fold_load[value], value))
            group_fold[group] = fold
            fold_load[fold] += group_counts[group]
        for record in records:
            record["prediction_fold"] = group_fold[str(record["physical_group"])]
    return records


def _domain_adaptation_train_subset(
    records: list[dict[str, Any]],
    *,
    dataset_root: Path,
    samples_per_surface_camera: int,
    seed: int,
    fold_count: int,
    strategy: str = "view_farthest_first",
) -> list[dict[str, Any]]:
    if samples_per_surface_camera < 1:
        raise ValueError("detector domain adaptation samples_per_surface_camera must be positive")
    if any(not 0 <= int(record["prediction_fold"]) < fold_count for record in records):
        raise ValueError("detector domain adaptation record has an invalid fold")
    if strategy != "view_farthest_first":
        raise ValueError(f"unsupported detector adaptation strategy: {strategy}")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        filename = Path(str(record["image_path"])).name
        prefix, camera_view = filename.rsplit("_camera", 1)
        camera_text, view_text = camera_view.rsplit("-", 1)
        view_id = int(Path(view_text).stem)
        if not 1 <= view_id <= 40:
            raise ValueError("detector adaptation view_id must be in 1..40")
        surface = "back" if prefix.endswith("-back") else "front"
        grouped[(str(record["physical_group"]), surface, int(camera_text))].append(
            dict(record, view_id=view_id)
        )
    hash_owner: dict[str, tuple[tuple[str, str, int], str, str]] = {}
    record_hashes: dict[tuple[tuple[str, str, int], str, str], str] = {}
    for stratum, values in grouped.items():
        for record in values:
            owner = (
                stratum,
                str(record["image_id"]),
                str(record["image_path"]),
            )
            image_hash = sha256_file(dataset_root / str(record["image_path"]))
            record_hashes[owner] = image_hash
            if image_hash not in hash_owner or owner < hash_owner[image_hash]:
                hash_owner[image_hash] = owner
    selected: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        candidates = []
        for record in grouped[stratum]:
            owner = (stratum, str(record["image_id"]), str(record["image_path"]))
            image_hash = record_hashes[owner]
            if hash_owner[image_hash] == owner:
                candidates.append((record, image_hash))
        order: list[tuple[dict[str, Any], str]] = []
        while candidates:
            if not order:
                selected_index = min(
                    range(len(candidates)),
                    key=lambda index: hashlib.sha256(
                        f"{seed}:{stratum}:{candidates[index][0]['view_id']}:"
                        f"{candidates[index][0]['image_id']}".encode()
                    ).hexdigest(),
                )
            else:
                chosen_views = [int(record["view_id"]) for record, _hash in order]

                def rank(index: int) -> tuple[int, str]:
                    view_id = int(candidates[index][0]["view_id"])
                    distance = min(
                        min(abs(view_id - chosen), 40 - abs(view_id - chosen))
                        for chosen in chosen_views
                    )
                    tie = hashlib.sha256(
                        f"{seed}:{stratum}:{view_id}:{candidates[index][0]['image_id']}".encode()
                    ).hexdigest()
                    return -distance, tie

                selected_index = min(range(len(candidates)), key=rank)
            order.append(candidates.pop(selected_index))
        for winner, image_hash in order[:samples_per_surface_camera]:
            adapted = dict(winner)
            adapted["fold"] = int(winner["prediction_fold"])
            adapted["split"] = "development"
            adapted["domain_adaptation"] = True
            adapted["source_image_sha256"] = image_hash
            selected.append(adapted)
    return selected


def _target_oof_gate_report(
    records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    adaptation: dict[str, Any],
    *,
    expected_class_count: int,
    score_threshold: float,
) -> dict[str, Any]:
    if not records or len(results) != len(records) or expected_class_count < 1:
        raise ValueError("target OOF gate inputs are incomplete")

    def is_exact_normal(record: dict[str, Any], result: dict[str, Any]) -> bool:
        return bool(
            not result["recapture_reasons"]
            and len(record["annotations"]) == 1
            and len(result["matches"]) == 1
            and not result["unmatched_detection_indices"]
        )

    exact_normal_pairs = [
        (record, result)
        for record, result in zip(records, results)
        if is_exact_normal(record, result)
    ]
    exact_normal_count = len(exact_normal_pairs)
    class_count = len(
        {
            int(annotation["category_id"])
            for record, _result in exact_normal_pairs
            for annotation in record["annotations"]
        }
    )
    accepted_per_class = Counter(
        int(record["annotations"][0]["category_id"]) for record, _result in exact_normal_pairs
    )
    minimum_accepted_per_class = int(adaptation.get("target_min_accepted_per_class", 1))
    if minimum_accepted_per_class < 1:
        raise ValueError("target_min_accepted_per_class must be positive")
    minimum_observed = min(accepted_per_class.values(), default=0)
    report = {
        "score_threshold": float(score_threshold),
        "bbox_recall": float(metrics["recall"]),
        "target_bbox_recall": float(adaptation.get("target_bbox_recall", 0.99)),
        "exact_normal_rate": exact_normal_count / len(records),
        "target_exact_normal_rate": float(adaptation.get("target_exact_normal_rate", 0.99)),
        "class_count": class_count,
        "expected_class_count": expected_class_count,
        "class_coverage": class_count / expected_class_count,
        "target_class_coverage": float(adaptation.get("target_class_coverage", 1.0)),
        "accepted_per_class": {
            str(category_id): accepted_per_class.get(category_id, 0)
            for category_id in range(1, expected_class_count + 1)
        },
        "minimum_accepted_per_class": minimum_accepted_per_class,
        "minimum_observed_accepted_per_class": minimum_observed,
    }
    gates = (
        ("TARGET_BBOX_RECALL", report["bbox_recall"] >= report["target_bbox_recall"]),
        (
            "TARGET_EXACT_NORMAL_RATE",
            report["exact_normal_rate"] >= report["target_exact_normal_rate"],
        ),
        (
            "TARGET_CLASS_COVERAGE",
            report["class_coverage"] >= report["target_class_coverage"],
        ),
        (
            "TARGET_MIN_ACCEPTED_PER_CLASS",
            minimum_observed >= minimum_accepted_per_class,
        ),
    )
    report["passes"] = all(passed for _name, passed in gates)
    report["failure_reasons"] = [name for name, passed in gates if not passed]
    return report


def _domain_adaptation_source_replay(
    records: list[dict[str, Any]], multiplier: int
) -> list[dict[str, Any]]:
    if multiplier < 1:
        raise ValueError("detector adaptation source_replay_multiplier must be positive")
    return [
        {
            **record,
            "adaptation_replay_only": True,
            "adaptation_replay_index": replay_index,
            "adaptation_replay_key": (
                f"{record['source']}:{record['image_id']}:replay:{replay_index}"
            ),
        }
        for replay_index in range(1, multiplier)
        for record in records
    ]


def _adaptation_progressive_fold_gate(
    *,
    adaptation_dir: Path,
    detector_dir: Path,
    dataset_root: Path,
    checkpoint: Path,
    fold: int,
    source_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    options: dict[str, Any],
    adaptation: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if not source_records or not target_records:
        raise ValueError("progressive adaptation fold gate has an empty partition")
    threshold_path = detector_dir / "threshold.json"
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    if threshold.get("threshold_policy") != "calibration_oof_only":
        raise ValueError("progressive gate requires a calibration-only baseline threshold")
    score_threshold = float(threshold["selected_score_threshold"])
    checkpoint_sha256 = sha256_file(_detector_weights_path(checkpoint))
    threshold_sha256 = sha256_file(threshold_path)
    recipe_sha256 = hashlib.sha256(_canonical_json(adaptation).encode()).hexdigest()
    policy = dict(options.get("train_gate_policy", adaptation))
    policy_sha256 = hashlib.sha256(_canonical_json(policy).encode()).hexdigest()
    inference = {
        "batch_size": int(options["inference_batch_size"]),
        "minimum_score": float(options["min_score_threshold"]),
        "fold": fold,
        "frozen_score_threshold": score_threshold,
        "baseline_threshold_sha256": threshold_sha256,
        "adaptation_config_sha256": recipe_sha256,
    }

    def predictions_for(
        name: str, records: list[dict[str, Any]], source_sha256: str
    ) -> tuple[Path, list[dict[str, Any]]]:
        path = adaptation_dir / "progressive-gate" / f"fold{fold}-{name}.jsonl"
        identity = _prediction_identity(
            records,
            [checkpoint],
            source_sha256=source_sha256,
            inference_config={**inference, "partition": name},
        )
        if _prediction_artifact_valid(path, identity):
            return path, _read_jsonl(path)
        predictions = predict_records(
            checkpoint,
            records,
            dataset_root,
            batch_size=int(options["inference_batch_size"]),
            minimum_score=float(options["min_score_threshold"]),
        )
        _write_prediction_artifact(path, predictions, identity)
        return path, predictions

    source_path, source_predictions = predictions_for(
        "source-oof",
        source_records,
        sha256_file(dataset_root / "instances_val2019.json"),
    )
    target_path, target_predictions = predictions_for(
        "target-physical-group",
        target_records,
        sha256_file(dataset_root / "instances_train2019.json"),
    )
    metric_options = dict(options, score_threshold=score_threshold)
    source_metrics = _metrics(
        source_records,
        source_predictions,
        score_threshold=score_threshold,
        nms_iou_threshold=float(options["nms_iou_threshold"]),
        match_iou_threshold=float(options["match_iou_threshold"]),
        max_queries=int(options["max_queries"]),
    )
    by_target_key = {str(prediction["sample_key"]): prediction for prediction in target_predictions}
    target_results = [
        postprocess_worker_gate(
            record,
            by_target_key[f"{record['source']}:{record['image_id']}"],
            metric_options,
        )
        for record in target_records
    ]
    target_metrics = _metrics(
        target_records,
        target_predictions,
        score_threshold=score_threshold,
        nms_iou_threshold=float(options["nms_iou_threshold"]),
        match_iou_threshold=float(options["match_iou_threshold"]),
        max_queries=int(options["max_queries"]),
    )
    fold_class_count = len(
        {
            int(annotation["category_id"])
            for record in target_records
            for annotation in record["annotations"]
        }
    )
    target_gate = _target_oof_gate_report(
        target_records,
        target_results,
        target_metrics,
        policy,
        expected_class_count=fold_class_count,
        score_threshold=score_threshold,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": "baseline_calibration_frozen_threshold_progressive_fold_gate",
        "fold": fold,
        "score_threshold": score_threshold,
        "baseline_threshold_sha256": threshold_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "adaptation_config_sha256": recipe_sha256,
        "train_gate_policy_sha256": policy_sha256,
        "source_prediction_sha256": sha256_file(source_path),
        "source_prediction_metadata_sha256": sha256_file(_prediction_metadata_path(source_path)),
        "target_prediction_sha256": sha256_file(target_path),
        "target_prediction_metadata_sha256": sha256_file(_prediction_metadata_path(target_path)),
        "source_metrics": source_metrics,
        "source_role": "diagnostic_only",
        "target_gate": target_gate,
        "passes": target_gate["passes"],
    }
    report["failure_reasons"] = list(target_gate["failure_reasons"])
    report_path = adaptation_dir / "progressive-gate" / f"fold{fold}.json"
    _write_json(report_path, report)
    return report


def _train_gate_complete(
    detector_dir: Path, config: dict[str, Any]
) -> tuple[bool, dict[str, Any] | None]:
    marker_path = detector_dir / "train-gate" / "complete.json"
    if not marker_path.is_file():
        return False, None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        adaptation = config["detector"]["domain_adaptation"]
        root = _active_detector_artifact_root(detector_dir, marker)
        core_expected = {
            "baseline_complete_sha256": detector_dir / "baseline" / "complete.json",
            "baseline_threshold_sha256": detector_dir / "threshold.json",
            "adaptation_manifest_sha256": root / "manifest" / "manifest.jsonl",
            "adaptation_manifest_metadata_sha256": root / "manifest" / "metadata.json",
            "train_predictions_sha256": root / "predictions" / "train_assigned.jsonl",
            "train_predictions_metadata_sha256": _prediction_metadata_path(
                root / "predictions" / "train_assigned.jsonl"
            ),
            "target_oof_gate_sha256": root / "target_oof_gate.json",
            "progressive_fold0_gate_sha256": root / "progressive-gate" / "fold0.json",
            "threshold_sha256": root / "threshold.json",
        }
        core_valid = (
            marker.get("role") == "train_gate_only"
            and marker.get("checkpoint_set") == "domain_adaptation"
            and marker.get("adaptation_config_sha256")
            == hashlib.sha256(_canonical_json(adaptation).encode()).hexdigest()
            and marker.get("train_gate_policy_sha256")
            == hashlib.sha256(
                _canonical_json(config["detector"].get("train_gate_policy", adaptation)).encode()
            ).hexdigest()
            and marker.get("target_oof_gate", {}).get("passes") is True
            and marker.get("progressive_fold0_gate", {}).get("passes") is True
            and all(
                path.is_file() and marker.get(key) == sha256_file(path)
                for key, path in core_expected.items()
            )
        )
        selection_path = root / "baseline_frozen_selection_gate.json"
        if (
            core_valid
            and marker.get("baseline_frozen_selection_gate_sha256") is None
            and marker.get("baseline_frozen_selection_gate") is None
            and not selection_path.exists()
        ):
            experiment_path = detector_dir.parent / "prepared" / "experiment.json"
            test_accessed = (
                experiment_path.is_file()
                and json.loads(experiment_path.read_text(encoding="utf-8")).get("test_accessed")
                is True
            )
            if not test_accessed:
                baseline_marker = json.loads(
                    (detector_dir / "baseline" / "complete.json").read_text(encoding="utf-8")
                )
                baseline_predictions_path = detector_dir / "predictions" / "val_oof.jsonl"
                baseline_predictions_metadata_path = _prediction_metadata_path(
                    baseline_predictions_path
                )
                baseline_lineage_valid = (
                    baseline_marker.get("checkpoint_set") == "baseline"
                    and baseline_marker.get("val_predictions_sha256")
                    == sha256_file(baseline_predictions_path)
                    and baseline_marker.get("val_predictions_metadata_sha256")
                    == sha256_file(baseline_predictions_metadata_path)
                )
                if baseline_lineage_valid:
                    threshold = json.loads(
                        (detector_dir / "threshold.json").read_text(encoding="utf-8")
                    )
                    selection_gate = evaluate_frozen_detector_threshold_selection(
                        _read_jsonl(detector_dir / "manifest" / "manifest.jsonl"),
                        _read_jsonl(baseline_predictions_path),
                        threshold,
                        config["detector"],
                    )
                    if selection_gate.get("frozen_threshold_selection_gate") is True:
                        _write_json(selection_path, selection_gate)
                        marker["baseline_frozen_selection_gate_sha256"] = sha256_file(
                            selection_path
                        )
                        marker["baseline_frozen_selection_gate"] = selection_gate
                        _write_json(marker_path, marker)
        selection_valid = (
            selection_path.is_file()
            and marker.get("baseline_frozen_selection_gate_sha256") == sha256_file(selection_path)
            and marker.get("baseline_frozen_selection_gate")
            == json.loads(selection_path.read_text(encoding="utf-8"))
            and marker.get("baseline_frozen_selection_gate", {}).get("selection_threshold_policy")
            == "frozen_calibration_threshold"
            and marker.get("baseline_frozen_selection_gate", {}).get(
                "frozen_threshold_selection_gate"
            )
            is True
        )
        valid = core_valid and selection_valid
        return valid, marker if valid else None
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False, None


def prepare_detector_domain_adaptation(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    _reject_post_test_mutation(args, "detector adaptation")
    options = config["detector"]
    adaptation = options.get("domain_adaptation")
    if not isinstance(adaptation, dict) or not bool(adaptation.get("enabled", False)):
        raise ValueError("RPC detector domain adaptation is not enabled")
    detector_dir = args.output_dir / "detector"
    if not _baseline_detector_complete(detector_dir, args.dataset_root, config):
        raise ValueError("baseline detector phase is incomplete or invalid")
    baseline_path = detector_dir / "baseline" / "complete.json"
    train_gate_valid, train_gate_marker = _train_gate_complete(detector_dir, config)
    if train_gate_valid and train_gate_marker is not None:
        return train_gate_marker

    fold_count = int(options["fold_count"])
    val_records = _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    train_records = _train_records(
        args.dataset_root, fold_count=fold_count, fold_assignment="physical_group"
    )
    adaptation_train = _domain_adaptation_train_subset(
        train_records,
        dataset_root=args.dataset_root,
        samples_per_surface_camera=int(adaptation["samples_per_surface_camera"]),
        seed=int(adaptation["seed"]),
        fold_count=fold_count,
        strategy=str(adaptation.get("strategy", "view_farthest_first")),
    )
    adaptation_config_sha256 = hashlib.sha256(_canonical_json(adaptation).encode()).hexdigest()
    source_replay_multiplier = int(adaptation.get("source_replay_multiplier", 1))
    replay_records = _domain_adaptation_source_replay(val_records, source_replay_multiplier)
    adaptation_dir = detector_dir / "adaptation-attempts" / adaptation_config_sha256
    manifest = adaptation_dir / "manifest" / "manifest.jsonl"
    combined_records = [dict(record) for record in val_records] + replay_records + adaptation_train
    _write_jsonl(manifest, combined_records)
    _write_json(
        manifest.parent / "metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": "rpc-detector-domain-adaptation-" + sha256_file(manifest)[:12],
            "fold_count": fold_count,
            "validation_record_count": len(val_records),
            "adaptation_train_record_count": len(adaptation_train),
            "source_replay_multiplier": source_replay_multiplier,
            "source_replay_record_count": len(replay_records),
            "samples_per_surface_camera": int(adaptation["samples_per_surface_camera"]),
            "selection_strategy": str(adaptation.get("strategy", "view_farthest_first")),
            "train_source_sha256": sha256_file(args.dataset_root / "instances_train2019.json"),
            "validation_source_sha256": sha256_file(args.dataset_root / "instances_val2019.json"),
            "manifest_sha256": sha256_file(manifest),
        },
    )
    threshold_report = json.loads((detector_dir / "threshold.json").read_text(encoding="utf-8"))
    if threshold_report.get("threshold_policy") != "calibration_oof_only":
        raise ValueError("train gate requires immutable baseline calibration threshold")
    baseline_val_predictions = _read_jsonl(detector_dir / "predictions" / "val_oof.jsonl")
    baseline_selection_gate = evaluate_frozen_detector_threshold_selection(
        val_records,
        baseline_val_predictions,
        threshold_report,
        options,
    )
    _write_json(
        adaptation_dir / "baseline_frozen_selection_gate.json",
        baseline_selection_gate,
    )
    if baseline_selection_gate["frozen_threshold_selection_gate"] is not True:
        raise RuntimeError("RPC baseline detector frozen-threshold selection recall is below gate")
    baseline_checkpoints = _detector_checkpoint_set(detector_dir, fold_count, "baseline")
    adaptation_identity = {
        "manifest_sha256": sha256_file(manifest),
        "manifest_metadata_sha256": sha256_file(manifest.parent / "metadata.json"),
        "train_annotation_sha256": sha256_file(args.dataset_root / "instances_train2019.json"),
        "validation_annotation_sha256": sha256_file(args.dataset_root / "instances_val2019.json"),
        "adaptation_config_sha256": adaptation_config_sha256,
        "workers": int(adaptation.get("workers", options.get("workers", 0))),
    }
    for fold in range(fold_count):
        output = adaptation_dir / "folds" / f"fold{fold}"
        namespace = _domain_adaptation_namespace(
            options,
            manifest,
            args.dataset_root,
            output,
            fold,
            baseline_checkpoints[fold],
            adaptation_identity | {"fold": fold},
            resume=args.resume,
        )
        recipe = detector_optimizer_recipe(namespace)
        if _checkpoint_complete(
            output,
            expected_seed=int(namespace.seed),
            expected_optimizer_recipe=recipe,
        ):
            print(json.dumps({"skipped_complete_domain_adaptation_fold": fold}), flush=True)
        else:
            train_detector(namespace)
            _mark_checkpoint_complete(output, optimizer_recipe=recipe)
        if fold == 0:
            progressive_report = _adaptation_progressive_fold_gate(
                adaptation_dir=adaptation_dir,
                detector_dir=detector_dir,
                dataset_root=args.dataset_root,
                checkpoint=output / "best",
                fold=fold,
                source_records=[record for record in val_records if int(record["fold"]) == fold],
                target_records=[
                    record for record in adaptation_train if int(record["fold"]) == fold
                ],
                options=options,
                adaptation=adaptation,
                resume=args.resume,
            )
            if progressive_report["passes"] is not True:
                raise RuntimeError("domain adaptation progressive fold0 gate failed before fold1")

    checkpoints = [adaptation_dir / "folds" / f"fold{fold}" / "best" for fold in range(fold_count)]
    inference_config = {
        "batch_size": int(options["inference_batch_size"]),
        "minimum_score": float(options["min_score_threshold"]),
    }
    _write_json(adaptation_dir / "threshold.json", threshold_report)

    train_predictions_path = adaptation_dir / "predictions" / "train_assigned.jsonl"
    train_identity = _prediction_identity(
        train_records,
        checkpoints,
        source_sha256=sha256_file(args.dataset_root / "instances_train2019.json"),
        inference_config={**inference_config, "partition": "train_domain_adapted_oof"},
    )
    if _prediction_artifact_valid(train_predictions_path, train_identity):
        train_predictions = _read_jsonl(train_predictions_path)
    else:
        train_predictions = _predict_partitioned_train(
            train_records,
            adaptation_dir,
            args.dataset_root,
            int(options["inference_batch_size"]),
            float(options["min_score_threshold"]),
        )
        _write_prediction_artifact(train_predictions_path, train_predictions, train_identity)
    by_train_key = {str(item["sample_key"]): item for item in train_predictions}
    frozen_options = dict(options)
    frozen_options["score_threshold"] = float(threshold_report["selected_score_threshold"])
    target_metrics = _metrics(
        train_records,
        [by_train_key[f"{record['source']}:{record['image_id']}"] for record in train_records],
        score_threshold=float(frozen_options["score_threshold"]),
        nms_iou_threshold=float(options["nms_iou_threshold"]),
        match_iou_threshold=float(options["match_iou_threshold"]),
        max_queries=int(options["max_queries"]),
    )
    target_results = [
        postprocess_worker_gate(
            record,
            by_train_key[f"{record['source']}:{record['image_id']}"],
            frozen_options,
        )
        for record in train_records
    ]
    expected_class_count = int(config["experiment"]["expected_num_classes"])
    target_oof_gate = _target_oof_gate_report(
        train_records,
        target_results,
        target_metrics,
        options.get("train_gate_policy", adaptation),
        expected_class_count=expected_class_count,
        score_threshold=float(frozen_options["score_threshold"]),
    )
    _write_json(adaptation_dir / "target_oof_gate.json", target_oof_gate)
    if not target_oof_gate["passes"]:
        raise RuntimeError("RPC detector offline train-gate OOF viability gate failed")
    completed = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": datetime.now(UTC).isoformat(),
        "fold_count": fold_count,
        "role": "train_gate_only",
        "checkpoint_set": "domain_adaptation",
        "artifact_root": adaptation_dir.relative_to(detector_dir).as_posix(),
        "train_fold_assignment": "physical_group",
        "train_prediction_partition": "train_domain_adapted_oof",
        "validation_images": len(val_records),
        "train_images": len(train_records),
        "adaptation_manifest_sha256": sha256_file(manifest),
        "adaptation_manifest_metadata_sha256": sha256_file(manifest.parent / "metadata.json"),
        "adaptation_config_sha256": adaptation_identity["adaptation_config_sha256"],
        "train_gate_policy_sha256": hashlib.sha256(
            _canonical_json(options.get("train_gate_policy", adaptation)).encode()
        ).hexdigest(),
        "baseline_complete_sha256": sha256_file(baseline_path),
        "threshold_role": "immutable_baseline_calibration",
        "baseline_threshold_sha256": sha256_file(detector_dir / "threshold.json"),
        "baseline_frozen_selection_gate_sha256": sha256_file(
            adaptation_dir / "baseline_frozen_selection_gate.json"
        ),
        "baseline_frozen_selection_gate": baseline_selection_gate,
        "target_oof_gate_sha256": sha256_file(adaptation_dir / "target_oof_gate.json"),
        "target_oof_gate": target_oof_gate,
        "progressive_fold0_gate_sha256": sha256_file(
            adaptation_dir / "progressive-gate" / "fold0.json"
        ),
        "progressive_fold0_gate": json.loads(
            (adaptation_dir / "progressive-gate" / "fold0.json").read_text(encoding="utf-8")
        ),
        "threshold_sha256": sha256_file(adaptation_dir / "threshold.json"),
        "train_predictions_sha256": sha256_file(train_predictions_path),
        "train_predictions_metadata_sha256": sha256_file(
            _prediction_metadata_path(train_predictions_path)
        ),
    }
    _write_json(detector_dir / "train-gate" / "complete.json", completed)
    return completed


def _predict_partitioned_train(
    records: list[dict[str, Any]],
    detector_dir: Path,
    dataset_root: Path,
    batch_size: int,
    minimum_score: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for fold in sorted({int(record["prediction_fold"]) for record in records}):
        subset = [record for record in records if int(record["prediction_fold"]) == fold]
        predictions.extend(
            predict_records(
                detector_dir / "folds" / f"fold{fold}" / "best",
                subset,
                dataset_root,
                batch_size=batch_size,
                minimum_score=minimum_score,
            )
        )
    return predictions


def prepare_detector_phase(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    _reject_post_test_mutation(args, "detector training")
    options = config["detector"]
    detector_dir = args.output_dir / "detector"
    baseline_path = detector_dir / "baseline" / "complete.json"
    active_path = detector_dir / "complete.json"
    if baseline_path.is_file():
        if not _baseline_detector_complete(detector_dir, args.dataset_root, config):
            raise ValueError("immutable baseline detector completion is invalid")
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    if active_path.is_file():
        try:
            active_marker = json.loads(active_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("active detector completion is invalid") from exc
        if str(active_marker.get("checkpoint_set", "baseline")) != "baseline":
            raise ValueError("cannot reconstruct an immutable baseline from an adapted marker")
        if not _baseline_detector_complete(detector_dir, args.dataset_root, config):
            raise ValueError("legacy baseline detector completion is invalid")
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    manifest = detector_dir / "manifest" / "manifest.jsonl"
    fold_count = int(options["fold_count"])
    if args.resume and _manifest_valid(manifest, args.dataset_root, fold_count=fold_count):
        records = _read_jsonl(manifest)
    else:
        records = build_rpc_detector_manifest(args.dataset_root, detector_dir, fold_count)
    roles = assign_validation_roles(
        records,
        int(config["experiment"]["expected_num_classes"]),
        int(config["experiment"]["validation_split_seed"]),
    )
    for record in records:
        record["role"] = roles[str(record["capture_session_id"])]
    _write_jsonl(manifest, records)
    manifest_metadata_path = manifest.parent / "metadata.json"
    manifest_metadata = json.loads(manifest_metadata_path.read_text(encoding="utf-8"))
    manifest_metadata.update(
        {
            "fold_count": fold_count,
            "record_count": len(records),
            "source_sha256": sha256_file(args.dataset_root / "instances_val2019.json"),
            "manifest_sha256": sha256_file(manifest),
        }
    )
    _write_json(manifest_metadata_path, manifest_metadata)
    train_oof_detectors(options, manifest, args.dataset_root, detector_dir, resume=args.resume)

    val_predictions_path = detector_dir / "predictions" / "val_oof.jsonl"
    inference_config = {
        "batch_size": int(options["inference_batch_size"]),
        "minimum_score": float(options["min_score_threshold"]),
    }
    val_identity = _prediction_identity(
        records,
        [detector_dir / "folds" / f"fold{fold}" / "best" for fold in range(fold_count)],
        source_sha256=sha256_file(args.dataset_root / "instances_val2019.json"),
        inference_config={**inference_config, "partition": "validation_oof"},
    )
    if args.resume and _prediction_artifact_valid(val_predictions_path, val_identity):
        val_predictions = _read_jsonl(val_predictions_path)
    else:
        val_predictions = []
        for fold in range(fold_count):
            subset = [record for record in records if int(record["fold"]) == fold]
            val_predictions.extend(
                predict_records(
                    detector_dir / "folds" / f"fold{fold}" / "best",
                    subset,
                    args.dataset_root,
                    batch_size=int(options["inference_batch_size"]),
                    minimum_score=float(options["min_score_threshold"]),
                )
            )
        _write_prediction_artifact(val_predictions_path, val_predictions, val_identity)
    threshold_report = select_detector_threshold(records, val_predictions, options)
    _write_json(detector_dir / "threshold.json", threshold_report)
    if not threshold_report["target_recall_satisfied"]:
        raise RuntimeError("RPC detector calibration recall is below the 99% gate")

    train_records = _train_records(args.dataset_root)
    train_predictions_path = detector_dir / "predictions" / "train_assigned.jsonl"
    train_identity = _prediction_identity(
        train_records,
        [detector_dir / "folds" / f"fold{fold}" / "best" for fold in range(fold_count)],
        source_sha256=sha256_file(args.dataset_root / "instances_train2019.json"),
        inference_config={**inference_config, "partition": "train_assigned"},
    )
    if args.resume and _prediction_artifact_valid(train_predictions_path, train_identity):
        train_predictions = _read_jsonl(train_predictions_path)
    else:
        train_predictions = _predict_partitioned_train(
            train_records,
            detector_dir,
            args.dataset_root,
            int(options["inference_batch_size"]),
            float(options["min_score_threshold"]),
        )
        _write_prediction_artifact(train_predictions_path, train_predictions, train_identity)
    completed = {
        "schema_version": SCHEMA_VERSION,
        "contract": "rpc-detector-baseline-complete-v1",
        "checkpoint_set": "baseline",
        "artifact_root": ".",
        "completed_at": datetime.now(UTC).isoformat(),
        "fold_count": int(options["fold_count"]),
        "validation_images": len(records),
        "train_images": len(train_records),
        "threshold_sha256": sha256_file(detector_dir / "threshold.json"),
        "val_predictions_sha256": sha256_file(val_predictions_path),
        "val_predictions_metadata_sha256": sha256_file(
            _prediction_metadata_path(val_predictions_path)
        ),
        "train_predictions_sha256": sha256_file(train_predictions_path),
        "train_predictions_metadata_sha256": sha256_file(
            _prediction_metadata_path(train_predictions_path)
        ),
    }
    if baseline_path.is_file():
        existing = json.loads(baseline_path.read_text(encoding="utf-8"))
        immutable = {key: value for key, value in completed.items() if key != "completed_at"}
        existing_immutable = {
            key: value for key, value in existing.items() if key != "completed_at"
        }
        if existing_immutable != immutable:
            raise ValueError("immutable baseline detector completion already exists")
    else:
        _write_json(baseline_path, completed)
    _write_json(detector_dir / "complete.json", completed)
    return completed


def load_worker_gated_records(
    dataset_root: Path, output_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detector_dir = output_dir / "detector"
    if not _baseline_detector_complete(detector_dir, dataset_root, config):
        raise ValueError("immutable baseline detector marker is invalid")
    train_gate_valid, train_gate_marker = _train_gate_complete(detector_dir, config)
    if not train_gate_valid or train_gate_marker is None:
        raise ValueError("detector train-gate completion is invalid")
    baseline_marker = json.loads(
        (detector_dir / "baseline" / "complete.json").read_text(encoding="utf-8")
    )
    baseline_root = _active_detector_artifact_root(detector_dir, baseline_marker)
    train_gate_root = _active_detector_artifact_root(detector_dir, train_gate_marker)
    threshold = json.loads((baseline_root / "threshold.json").read_text(encoding="utf-8"))
    options = dict(config["detector"])
    options["score_threshold"] = float(threshold["selected_score_threshold"])
    val_base = _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    train_base = _train_records(
        dataset_root,
        fold_count=int(config["detector"]["fold_count"]),
        fold_assignment="physical_group",
    )
    val_predictions = {
        str(item["sample_key"]): item
        for item in _read_jsonl(baseline_root / "predictions" / "val_oof.jsonl")
    }
    train_predictions = {
        str(item["sample_key"]): item
        for item in _read_jsonl(train_gate_root / "predictions" / "train_assigned.jsonl")
    }
    train_rows: list[dict[str, Any]] = []
    recapture_positive_count = 0
    hard_negative_candidates: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    train_rejected: Counter[str] = Counter()
    val_image_outcomes: list[dict[str, Any]] = []

    for record in train_base:
        result = postprocess_worker_gate(
            record, train_predictions[f"{record['source']}:{record['image_id']}"], options
        )
        hard_negative_candidates.extend(
            _hard_negative_rows(record, result, config.get("training", {}))
        )
        reasons = list(result["recapture_reasons"])
        if (
            len(result["detections"]) != 1
            or len(result["matches"]) != 1
            or result["missed_annotation_indices"]
        ):
            reasons.append("DATA_ALIGNMENT_REJECT")
        worker_recaptured = bool(reasons)
        if worker_recaptured:
            _update_unique_reason_counts(train_rejected, reasons)
        train_row = _train_product_row(
            record,
            result,
            reasons=reasons,
            training=config.get("training", {}),
        )
        if train_row is None:
            continue
        train_rows.append(train_row)
        recapture_positive_count += int(worker_recaptured)

    positive_train_count = len(train_rows)
    selected_hard_negatives = _select_hard_negative_rows(
        hard_negative_candidates,
        positive_count=positive_train_count,
        training=config.get("training", {}),
    )
    train_rows.extend(selected_hard_negatives)

    for record in val_base:
        result = postprocess_worker_gate(
            record, val_predictions[f"{record['source']}:{record['image_id']}"], options
        )
        reason_counts.update(result["recapture_reasons"])
        outcome = {
            "image_id": int(record["image_id"]),
            "image_path": record["image_path"],
            "fold": int(record["fold"]),
            "role": record["role"],
            "level": record["level"],
            "recapture_reasons": result["recapture_reasons"],
            "ground_truth_count": len(record["annotations"]),
            "detection_count": len(result["detections"]),
            "matched_count": len(result["matches"]),
            "missed_count": len(result["missed_annotation_indices"]),
            "unmatched_count": len(result["unmatched_detection_indices"]),
        }
        val_image_outcomes.append(outcome)
        if result["recapture_reasons"]:
            continue
        for detection_index, detection in enumerate(result["detections"]):
            match = result["matches"].get(str(detection_index))
            annotation = record["annotations"][int(match[0])] if match is not None else None
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            margin = float(options["border_margin_ratio"])
            touches_border = (
                x1 <= int(record["width"]) * margin
                or y1 <= int(record["height"]) * margin
                or x2 >= int(record["width"]) * (1.0 - margin)
                or y2 >= int(record["height"]) * (1.0 - margin)
            )
            val_rows.append(
                {
                    "sample_id": f"val:{record['image_id']}:det{detection_index}",
                    "split": "val",
                    "image_id": int(record["image_id"]),
                    "annotation_id": None
                    if annotation is None
                    else int(annotation["annotation_id"]),
                    "image_path": record["image_path"],
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                    "bbox_xyxy": detection["bbox_xyxy"],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "detector_score": detection["score"],
                    "category_id": None if annotation is None else int(annotation["category_id"]),
                    "target": -1 if annotation is None else int(annotation["category_id"]) - 1,
                    "level": record["level"],
                    "group_id": record["capture_session_id"],
                    "fold": int(record["fold"]),
                    "role": record["role"],
                    "match_iou": None if match is None else float(match[1]),
                    "touches_border": bool(touches_border),
                }
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "score_threshold": options["score_threshold"],
        "threshold_source": "calibration_oof_only",
        "validation_detector_role": "immutable_baseline_oof",
        "training_detector_role": "train_gate_only",
        "train_gate_complete_sha256": sha256_file(detector_dir / "train-gate" / "complete.json"),
        "train_candidates": positive_train_count,
        "train_normal_positive_count": positive_train_count - recapture_positive_count,
        "train_recapture_positive_count": recapture_positive_count,
        "hard_negative_candidates": len(hard_negative_candidates),
        "hard_negative_selected": len(selected_hard_negatives),
        "train_rejected": dict(sorted(train_rejected.items())),
        "validation_images": len(val_base),
        "validation_normal_images": sum(not row["recapture_reasons"] for row in val_image_outcomes),
        "validation_recapture_images": sum(
            bool(row["recapture_reasons"]) for row in val_image_outcomes
        ),
        "validation_recapture_reasons": dict(sorted(reason_counts.items())),
        "validation_missed_boxes": sum(row["missed_count"] for row in val_image_outcomes),
        "validation_unmatched_boxes": sum(row["unmatched_count"] for row in val_image_outcomes),
        "validation_image_outcomes": val_image_outcomes,
    }
    return train_rows, val_rows, report


def _train_product_row(
    record: dict[str, Any],
    result: dict[str, Any],
    *,
    reasons: list[str],
    training: dict[str, Any],
) -> dict[str, Any] | None:
    """Create a product ROI without changing an image-level RECAPTURE verdict."""
    worker_recaptured = bool(reasons)
    if worker_recaptured:
        if not bool(training.get("recapture_positive_enabled", False)):
            return None
        if result["missed_annotation_indices"] or len(result["matches"]) != 1:
            return None
        detection_index, match = next(iter(result["matches"].items()))
        min_iou = float(training.get("recapture_positive_min_iou", 0.5))
        if not 0.5 <= min_iou <= 1.0:
            raise ValueError("recapture-positive IoU must be between 0.5 and 1.0")
        if float(match[1]) < min_iou:
            return None
        detection = result["detections"][int(detection_index)]
    else:
        if len(result["detections"]) != 1 or len(result["matches"]) != 1:
            raise ValueError("normal train product row requires one matched detection")
        detection = result["detections"][0]
    annotation = record["annotations"][0]
    filename = Path(record["image_path"]).name
    prefix, camera_view = filename.rsplit("_camera", 1)
    camera_text, view_text = camera_view.rsplit("-", 1)
    x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
    return {
        "sample_id": f"train:{record['image_id']}:{annotation['annotation_id']}",
        "split": "train",
        "image_id": int(record["image_id"]),
        "annotation_id": int(annotation["annotation_id"]),
        "image_path": record["image_path"],
        "width": int(record["width"]),
        "height": int(record["height"]),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "detector_score": float(detection["score"]),
        "category_id": int(annotation["category_id"]),
        "target": int(annotation["category_id"]) - 1,
        "barcode": prefix,
        "surface": "back" if prefix.endswith("-back") else "front",
        "camera": int(camera_text),
        "view_id": int(Path(view_text).stem),
        "prediction_fold": int(record["prediction_fold"]),
        "worker_gate_role": ("recapture_positive" if worker_recaptured else "normal_positive"),
        "worker_recapture_reasons": list(dict.fromkeys(reasons)),
    }


def _hard_negative_rows(
    record: dict[str, Any],
    result: dict[str, Any],
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return OOF detections that are confidently separated from every train GT."""
    if not bool(training.get("hard_negative_enabled", False)):
        return []
    max_iou = float(training.get("hard_negative_max_gt_iou", 0.1))
    min_score = float(training.get("hard_negative_min_score", 0.1))
    min_area_ratio = float(training.get("hard_negative_min_area_ratio", 0.005))
    max_per_image = int(training.get("hard_negative_max_per_image", 1))
    if not 0.0 <= max_iou < 0.5 or not 0.0 <= min_score <= 1.0:
        raise ValueError("invalid classifier hard-negative overlap/score policy")
    if min_area_ratio < 0.0 or max_per_image < 1:
        raise ValueError("invalid classifier hard-negative area/count policy")
    ground_truth = [_xywh_to_xyxy(annotation["bbox_xywh"]) for annotation in record["annotations"]]
    filename = Path(str(record["image_path"])).name
    prefix, camera_view = filename.rsplit("_camera", 1)
    camera_text, view_text = camera_view.rsplit("-", 1)
    width = int(record["width"])
    height = int(record["height"])
    image_area = float(width * height)
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for detection_index in result["unmatched_detection_indices"]:
        detection = result["detections"][int(detection_index)]
        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        overlap = max((_iou([x1, y1, x2, y2], box) for box in ground_truth), default=0.0)
        score = float(detection["score"])
        if score < min_score or area_ratio < min_area_ratio or overlap > max_iou:
            continue
        candidates.append(
            (
                -score,
                int(detection_index),
                {
                    "sample_id": f"train-hard-negative:{record['image_id']}:det{detection_index}",
                    "split": "train",
                    "role": "hard_negative",
                    "image_id": int(record["image_id"]),
                    "annotation_id": None,
                    "image_path": record["image_path"],
                    "width": width,
                    "height": height,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "detector_score": score,
                    "max_gt_iou": float(overlap),
                    "category_id": None,
                    "target": -1,
                    "physical_group": str(record["physical_group"]),
                    "barcode": prefix,
                    "surface": "back" if prefix.endswith("-back") else "front",
                    "camera": int(camera_text),
                    "view_id": int(Path(view_text).stem),
                    "prediction_fold": int(record["prediction_fold"]),
                },
            )
        )
    return [value[2] for value in sorted(candidates)[:max_per_image]]


def _select_hard_negative_rows(
    rows: list[dict[str, Any]],
    *,
    positive_count: int,
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select repeat-resistant, view-diverse hard negatives from train2019 only."""
    if not rows or not bool(training.get("hard_negative_enabled", False)):
        return []
    views_per_stratum = int(training.get("hard_negative_views_per_surface_camera", 8))
    max_ratio = float(training.get("hard_negative_max_ratio", 1.0))
    seed = int(training.get("hard_negative_seed", 20260810))
    if views_per_stratum < 1 or max_ratio <= 0.0:
        raise ValueError("invalid classifier hard-negative selection policy")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stratum = (
            str(row["physical_group"]),
            str(row["surface"]),
            int(row["camera"]),
        )
        grouped[stratum].append(row)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        candidates = sorted(
            grouped[stratum],
            key=lambda row: (-float(row["detector_score"]), str(row["sample_id"])),
        )
        ordered: list[dict[str, Any]] = []
        while candidates and len(ordered) < views_per_stratum:
            if not ordered:
                chosen_index = 0
            else:
                used_views = [int(row["view_id"]) for row in ordered]

                def rank(index: int) -> tuple[int, float, str]:
                    view_id = int(candidates[index]["view_id"])
                    distance = min(
                        min(abs(view_id - used), 40 - abs(view_id - used)) for used in used_views
                    )
                    tie = hashlib.sha256(
                        f"{seed}:{stratum}:{candidates[index]['sample_id']}".encode()
                    ).hexdigest()
                    return (
                        -distance,
                        -float(candidates[index]["detector_score"]),
                        tie,
                    )

                chosen_index = min(range(len(candidates)), key=rank)
            ordered.append(candidates.pop(chosen_index))
        selected.extend(ordered)
    limit = max(1, int(round(positive_count * max_ratio)))
    return sorted(
        selected,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest(),
            str(row["sample_id"]),
        ),
    )[:limit]


def _update_unique_reason_counts(counter: Counter[str], reasons: list[str]) -> None:
    """Count each rejection reason at most once for a single image."""
    counter.update(dict.fromkeys(reasons, 1))


def _best_detector_epoch(detector_dir: Path, fold_count: int) -> int:
    """Select the fixed base-training length from baseline OOF folds only.

    Domain-adaptation history is downstream evidence and must never be allowed
    to shorten the independently selected val-all base training run.
    """
    fold_root = detector_dir / "folds"
    epochs: list[int] = []
    for fold in range(fold_count):
        history = json.loads(
            (fold_root / f"fold{fold}" / "history.json").read_text(encoding="utf-8")
        )
        eligible = [
            row
            for row in history
            if row.get("validation_loss") is not None
            and isinstance(row.get("detector_quality_key"), list)
        ]
        if not eligible:
            raise ValueError(f"detector fold {fold} has no metric-aware validation history")
        best = max(
            eligible,
            key=lambda row: (list(row["detector_quality_key"]), -int(row["epoch"])),
        )
        epochs.append(int(best["epoch"]))
    return max(1, int(round(float(np.median(epochs)))))


def _config_sha256(args: argparse.Namespace, config: dict[str, Any]) -> str:
    config_path = getattr(args, "config", None)
    if config_path is not None and Path(config_path).is_file():
        return sha256_file(Path(config_path))
    return hashlib.sha256(_canonical_json(config).encode()).hexdigest()


def _final_detector_paths(detector_dir: Path) -> dict[str, Path]:
    return {
        "baseline_manifest": detector_dir / "manifest" / "manifest.jsonl",
        "baseline_manifest_metadata": detector_dir / "manifest" / "metadata.json",
        "stage_a_complete": detector_dir / "final" / "stage-a-base" / "complete.json",
        "stage_a_training": detector_dir / "final" / "stage-a-base" / "final_training.json",
        "stage_a_checkpoint": detector_dir / "final" / "stage-a-base" / "best",
        "final_complete": detector_dir / "final" / "complete.json",
    }


def _final_detector_artifacts(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Validate the checkout-domain val-all operational detector."""
    detector_dir = args.output_dir / "detector"
    paths = _final_detector_paths(detector_dir)
    complete = json.loads(paths["final_complete"].read_text(encoding="utf-8"))
    if (
        complete.get("complete") is not True
        or complete.get("contract") != "rpc-final-detector-baseline-val-all-v1"
    ):
        raise ValueError("final detector completion marker is invalid")
    baseline_complete = detector_dir / "baseline" / "complete.json"
    active_threshold = detector_dir / "threshold.json"
    train_gate_complete = detector_dir / "train-gate" / "complete.json"
    expected = {
        "config_sha256": _config_sha256(args, config),
        "active_detector_complete_sha256": sha256_file(baseline_complete),
        "active_threshold_sha256": sha256_file(active_threshold),
        "baseline_manifest_sha256": sha256_file(paths["baseline_manifest"]),
        "baseline_manifest_metadata_sha256": sha256_file(paths["baseline_manifest_metadata"]),
        "train_gate_complete_sha256": sha256_file(train_gate_complete),
        "validation_annotation_sha256": sha256_file(args.dataset_root / "instances_val2019.json"),
        "stage_a_complete_sha256": sha256_file(paths["stage_a_complete"]),
        "stage_a_checkpoint_sha256": sha256_file(
            _detector_weights_path(paths["stage_a_checkpoint"])
        ),
    }
    if any(complete.get(key) != value for key, value in expected.items()):
        raise ValueError("final detector lineage checksum mismatch")
    options = config["detector"]
    expected_base_epochs = _best_detector_epoch(detector_dir, int(options["fold_count"]))
    stage_a_identity = {
        "stage": "base",
        "epoch_policy": "median_baseline_oof_metric_best_epoch",
        "baseline_manifest_sha256": expected["baseline_manifest_sha256"],
        "baseline_manifest_metadata_sha256": expected["baseline_manifest_metadata_sha256"],
        "config_sha256": expected["config_sha256"],
        "validation_annotation_sha256": expected["validation_annotation_sha256"],
    }
    stage_a_namespace = _detector_namespace(
        options,
        paths["baseline_manifest"],
        args.dataset_root,
        paths["stage_a_complete"].parent,
        0,
        resume=True,
    )
    stage_a_namespace.final_training = True
    stage_a_namespace.epochs = expected_base_epochs
    stage_a_namespace.seed = int(options["seed"])
    stage_a_namespace.fixed_epoch_checkpoint = True
    stage_a_namespace.training_identity = stage_a_identity
    stage_a_recipe = detector_optimizer_recipe(stage_a_namespace)
    if not _checkpoint_complete(
        paths["stage_a_complete"].parent,
        expected_seed=int(stage_a_namespace.seed),
        expected_optimizer_recipe=stage_a_recipe,
    ):
        raise ValueError("final detector stage-A recipe/checkpoint is invalid")
    stage_a = json.loads(paths["stage_a_complete"].read_text(encoding="utf-8"))
    expected_stage_a_marker = {
        "parent_reference": str(options["pretrained_name"]),
        "parent_reference_sha256": hashlib.sha256(
            str(options["pretrained_name"]).encode()
        ).hexdigest(),
        "manifest_sha256": expected["baseline_manifest_sha256"],
        "manifest_metadata_sha256": expected["baseline_manifest_metadata_sha256"],
    }
    if any(stage_a.get(key) != value for key, value in expected_stage_a_marker.items()):
        raise ValueError("final detector stage-A lineage is invalid")
    stage_a_training = json.loads(paths["stage_a_training"].read_text(encoding="utf-8"))
    expected_stage_a_training = {
        "stage": "base",
        "epoch_policy": "median_baseline_oof_metric_best_epoch",
        "epochs": expected_base_epochs,
        "trained_on": "val2019_all_groups",
        "parent": {"kind": "pretrained", "reference": options["pretrained_name"]},
        **stage_a_identity,
    }
    if any(stage_a_training.get(key) != value for key, value in expected_stage_a_training.items()):
        raise ValueError("final detector stage-A training policy is invalid")

    expected_aggregate_policy = {
        "base_epoch_policy": "median_baseline_oof_metric_best_epoch",
        "base_epochs": expected_base_epochs,
        "operational_training_domain": "checkout_val2019_all_groups",
        "operational_detector_role": "checkout_baseline_val_all_operational",
        "train_gate_role": "offline_roi_train_gate_only",
        "target_adaptation_stage": "disabled_train_gate_only",
    }
    if any(complete.get(key) != value for key, value in expected_aggregate_policy.items()):
        raise ValueError("final detector aggregate training policy is invalid")
    return paths["stage_a_checkpoint"], complete


def train_final_detector(args: argparse.Namespace, config: dict[str, Any], *, resume: bool) -> Path:
    options = config["detector"]
    detector_dir = args.output_dir / "detector"
    _reject_post_test_mutation(args, "final operational detector training")
    if not _baseline_detector_complete(detector_dir, args.dataset_root, config):
        raise ValueError("immutable baseline detector phase is incomplete or invalid")
    train_gate_valid, _train_gate_marker = _train_gate_complete(detector_dir, config)
    if not train_gate_valid:
        raise ValueError("train-gate detector phase is incomplete or invalid")
    paths = _final_detector_paths(detector_dir)
    if resume and paths["final_complete"].is_file():
        try:
            checkpoint, _complete = _final_detector_artifacts(args, config)
            return checkpoint
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    epochs = _best_detector_epoch(detector_dir, int(options["fold_count"]))
    stage_a = detector_dir / "final" / "stage-a-base"
    namespace = _detector_namespace(
        options,
        detector_dir / "manifest" / "manifest.jsonl",
        args.dataset_root,
        stage_a,
        0,
        resume=resume,
    )
    namespace.final_training = True
    namespace.epochs = epochs
    namespace.seed = int(options["seed"])
    namespace.fixed_epoch_checkpoint = True
    namespace.training_identity = {
        "stage": "base",
        "epoch_policy": "median_baseline_oof_metric_best_epoch",
        "baseline_manifest_sha256": sha256_file(paths["baseline_manifest"]),
        "baseline_manifest_metadata_sha256": sha256_file(paths["baseline_manifest_metadata"]),
        "config_sha256": _config_sha256(args, config),
        "validation_annotation_sha256": sha256_file(args.dataset_root / "instances_val2019.json"),
    }
    recipe = detector_optimizer_recipe(namespace)
    if not (
        resume
        and _checkpoint_complete(
            stage_a,
            expected_seed=int(namespace.seed),
            expected_optimizer_recipe=recipe,
        )
    ):
        train_detector(namespace)
        _mark_checkpoint_complete(stage_a, optimizer_recipe=recipe)
    stage_a_checkpoint = stage_a / "best"
    stage_a_sha256 = sha256_file(_detector_weights_path(stage_a_checkpoint))
    stage_a_complete = json.loads((stage_a / "complete.json").read_text(encoding="utf-8"))
    stage_a_complete["parent_reference"] = str(options["pretrained_name"])
    stage_a_complete["parent_reference_sha256"] = hashlib.sha256(
        str(options["pretrained_name"]).encode()
    ).hexdigest()
    stage_a_complete["manifest_sha256"] = namespace.training_identity["baseline_manifest_sha256"]
    stage_a_complete["manifest_metadata_sha256"] = namespace.training_identity[
        "baseline_manifest_metadata_sha256"
    ]
    _write_json(stage_a / "complete.json", stage_a_complete)
    _write_json(
        stage_a / "final_training.json",
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "base",
            "epoch_policy": "median_baseline_oof_metric_best_epoch",
            "epochs": epochs,
            "trained_on": "val2019_all_groups",
            "parent": {"kind": "pretrained", "reference": options["pretrained_name"]},
            **namespace.training_identity,
        },
    )

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "contract": "rpc-final-detector-baseline-val-all-v1",
        "complete": True,
        "completed_at": datetime.now(UTC).isoformat(),
        "base_epoch_policy": "median_baseline_oof_metric_best_epoch",
        "base_epochs": epochs,
        "operational_training_domain": "checkout_val2019_all_groups",
        "operational_detector_role": "checkout_baseline_val_all_operational",
        "train_gate_role": "offline_roi_train_gate_only",
        "target_adaptation_stage": "disabled_train_gate_only",
        "config_sha256": _config_sha256(args, config),
        "active_detector_complete_sha256": sha256_file(detector_dir / "baseline" / "complete.json"),
        "active_threshold_sha256": sha256_file(detector_dir / "threshold.json"),
        "train_gate_complete_sha256": sha256_file(detector_dir / "train-gate" / "complete.json"),
        "baseline_manifest_sha256": sha256_file(paths["baseline_manifest"]),
        "baseline_manifest_metadata_sha256": sha256_file(paths["baseline_manifest_metadata"]),
        "validation_annotation_sha256": sha256_file(args.dataset_root / "instances_val2019.json"),
        "stage_a_complete_sha256": sha256_file(stage_a / "complete.json"),
        "stage_a_checkpoint_sha256": stage_a_sha256,
    }
    _write_json(paths["final_complete"], aggregate)
    checkpoint, _complete = _final_detector_artifacts(args, config)
    return checkpoint


def _load_checkout_split_records(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads((dataset_root / f"instances_{split}2019.json").read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[int(annotation["image_id"])].append(
            {
                "annotation_id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "bbox_xywh": [float(value) for value in annotation["bbox"]],
                "area": float(annotation["area"]),
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )
    records = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        records.append(
            {
                "record_type": "detection",
                "source": f"rpc_{split}2019",
                "image_path": f"{split}2019/{image['file_name']}",
                "image_id": int(image["id"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "capture_session_id": checkout_group(str(image["file_name"])),
                "level": str(image["level"]),
                "annotations": sorted(
                    by_image[int(image["id"])], key=lambda row: row["annotation_id"]
                ),
            }
        )
    return records


def prepare_final_test_records(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    resume: bool,
    model_lock_path: Path,
    before_test_access: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate an already locked final detector without any training path."""
    experiment_path = args.output_dir / "prepared" / "experiment.json"
    if (
        experiment_path.is_file()
        and json.loads(experiment_path.read_text(encoding="utf-8")).get("test_accessed") is True
    ):
        raise RuntimeError("post-test output is immutable; use the sealed final-test report")
    detector_dir = args.output_dir / "detector"
    checkpoint, final_complete = _final_detector_artifacts(args, config)
    lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    lock_sha256 = sha256_file(model_lock_path)
    required_lock_hashes = {
        "rpc_config_sha256": final_complete["config_sha256"],
        "active_detector_complete_sha256": final_complete["active_detector_complete_sha256"],
        "active_detector_threshold_sha256": final_complete["active_threshold_sha256"],
        "detector_train_gate_complete_sha256": final_complete["train_gate_complete_sha256"],
        "final_detector_complete_sha256": sha256_file(detector_dir / "final" / "complete.json"),
        "final_detector_checkpoint_sha256": final_complete["stage_a_checkpoint_sha256"],
    }
    if any(lock.get(key) != value for key, value in required_lock_hashes.items()):
        raise ValueError("model lock final detector lineage checksum mismatch")
    if (
        lock.get("operational_detector_role") != "checkout_baseline_val_all_operational"
        or lock.get("train_gate_role") != "offline_roi_train_gate_only"
        or final_complete.get("target_adaptation_stage") != "disabled_train_gate_only"
    ):
        raise ValueError("model lock detector role separation mismatch")
    if before_test_access is not None:
        before_test_access()
    test_records = _load_checkout_split_records(args.dataset_root, "test")
    val_groups = {
        str(record["capture_session_id"])
        for record in _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    }
    test_groups = {str(record["capture_session_id"]) for record in test_records}
    overlap = val_groups & test_groups
    if overlap:
        raise ValueError(f"validation/test checkout groups overlap: {sorted(overlap)[:5]}")
    predictions_path = detector_dir / "predictions" / "test_final.jsonl"
    prediction_identity = _prediction_identity(
        test_records,
        [checkpoint],
        source_sha256=sha256_file(args.dataset_root / "instances_test2019.json"),
        inference_config={
            "batch_size": int(config["detector"]["inference_batch_size"]),
            "minimum_score": float(config["detector"]["min_score_threshold"]),
            "partition": "test_final",
            "model_lock_sha256": lock_sha256,
            "final_detector_complete_sha256": required_lock_hashes[
                "final_detector_complete_sha256"
            ],
            "final_detector_checkpoint_sha256": required_lock_hashes[
                "final_detector_checkpoint_sha256"
            ],
        },
    )
    if resume and _prediction_artifact_valid(predictions_path, prediction_identity):
        predictions = _read_jsonl(predictions_path)
    else:
        predictions = predict_records(
            checkpoint,
            test_records,
            args.dataset_root,
            batch_size=int(config["detector"]["inference_batch_size"]),
            minimum_score=float(config["detector"]["min_score_threshold"]),
        )
        _write_prediction_artifact(predictions_path, predictions, prediction_identity)
    by_key = {str(item["sample_key"]): item for item in predictions}
    threshold = json.loads((detector_dir / "threshold.json").read_text(encoding="utf-8"))
    options = dict(config["detector"])
    options["score_threshold"] = float(threshold["selected_score_threshold"])
    rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in test_records:
        result = postprocess_worker_gate(
            record, by_key[f"{record['source']}:{record['image_id']}"], options
        )
        reasons.update(result["recapture_reasons"])
        outcomes.append(
            {
                "image_id": int(record["image_id"]),
                "level": record["level"],
                "ground_truth_count": len(record["annotations"]),
                "detection_count": len(result["detections"]),
                "matched_count": len(result["matches"]),
                "missed_count": len(result["missed_annotation_indices"]),
                "unmatched_count": len(result["unmatched_detection_indices"]),
                "recapture_reasons": result["recapture_reasons"],
            }
        )
        if result["recapture_reasons"]:
            continue
        for detection_index, detection in enumerate(result["detections"]):
            match = result["matches"].get(str(detection_index))
            annotation = record["annotations"][int(match[0])] if match is not None else None
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            margin = float(options["border_margin_ratio"])
            rows.append(
                {
                    "sample_id": f"test:{record['image_id']}:det{detection_index}",
                    "split": "test",
                    "image_id": int(record["image_id"]),
                    "annotation_id": None
                    if annotation is None
                    else int(annotation["annotation_id"]),
                    "image_path": record["image_path"],
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "detector_score": detection["score"],
                    "category_id": None if annotation is None else int(annotation["category_id"]),
                    "target": -1 if annotation is None else int(annotation["category_id"]) - 1,
                    "level": record["level"],
                    "group_id": record["capture_session_id"],
                    "role": "test",
                    "match_iou": None if match is None else float(match[1]),
                    "touches_border": bool(
                        x1 <= int(record["width"]) * margin
                        or y1 <= int(record["height"]) * margin
                        or x2 >= int(record["width"]) * (1.0 - margin)
                        or y2 >= int(record["height"]) * (1.0 - margin)
                    ),
                }
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "operational_detector_role": "checkout_baseline_val_all_operational",
        "train_gate_role": "offline_roi_train_gate_only",
        "train_gate_complete_sha256": required_lock_hashes["detector_train_gate_complete_sha256"],
        "detector_checkpoint_sha256": sha256_file(_detector_weights_path(checkpoint)),
        "final_detector_complete_sha256": required_lock_hashes["final_detector_complete_sha256"],
        "model_lock_sha256": lock_sha256,
        "active_detector_threshold_sha256": required_lock_hashes[
            "active_detector_threshold_sha256"
        ],
        "test_annotation_sha256": sha256_file(args.dataset_root / "instances_test2019.json"),
        "image_count": len(test_records),
        "normal_image_count": sum(not row["recapture_reasons"] for row in outcomes),
        "recapture_image_count": sum(bool(row["recapture_reasons"]) for row in outcomes),
        "recapture_reasons": dict(sorted(reasons.items())),
        "ground_truth_count": sum(row["ground_truth_count"] for row in outcomes),
        "matched_count": sum(row["matched_count"] for row in outcomes),
        "missed_count": sum(row["missed_count"] for row in outcomes),
        "unmatched_count": sum(row["unmatched_count"] for row in outcomes),
        "outcomes": outcomes,
    }
    _write_json(args.output_dir / "test" / "detector_report.json", report)
    return rows, report
