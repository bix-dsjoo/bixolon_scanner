from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from ..inference import Detection, _box_iou, _nms
from ..package import sha256_file
from .evaluate_detector import _iou, _metrics, _xywh_to_xyxy
from .train_detector import train as train_detector


SCHEMA_VERSION = "1.0"


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
            class_cost = sum((fold_classes[fold][key] + value) ** 2 for key, value in classes.items())
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


def build_rpc_detector_manifest(dataset_root: Path, detector_dir: Path, fold_count: int) -> list[dict[str, Any]]:
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
            "dataset_version": "rpc-detector-" + hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()[:12],
            "fold_count": fold_count,
            "record_count": len(records),
            "source_sha256": sha256_file(dataset_root / "instances_val2019.json"),
            "manifest_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        },
    )
    return records


def assign_validation_roles(records: list[dict[str, Any]], category_count: int, seed: int) -> dict[str, str]:
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
        workers=int(options["workers"]),
        epochs=int(options["epochs"]),
        patience=int(options["patience"]),
        learning_rate=float(options["learning_rate"]),
        seed=int(options["seed"]) + fold,
        cpu=False,
        resume=resume,
    )


def _detector_weights_path(checkpoint: Path) -> Path:
    candidates = [
        path for path in (checkpoint / "model.safetensors", checkpoint / "pytorch_model.bin")
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"detector checkpoint weights are missing or ambiguous: {checkpoint}")
    return candidates[0]


def _checkpoint_complete(path: Path) -> bool:
    marker_path = path / "complete.json"
    history_path = path / "history.json"
    if not history_path.is_file() or not (path / "best" / "config.json").is_file():
        return False
    if not marker_path.is_file() and not (path / "training_progress.pt").is_file():
        try:
            _mark_checkpoint_complete(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        weights = _detector_weights_path(path / "best")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        return (
            marker.get("complete") is True
            and marker.get("weights_sha256") == sha256_file(weights)
            and int(marker.get("history_epochs", -1)) == len(history)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _mark_checkpoint_complete(path: Path) -> None:
    history = json.loads((path / "history.json").read_text(encoding="utf-8"))
    _write_json(
        path / "complete.json",
        {
            "complete": True,
            "completed_at": datetime.now(UTC).isoformat(),
            "history_epochs": len(history),
            "weights_sha256": sha256_file(_detector_weights_path(path / "best")),
        },
    )


def train_oof_detectors(
    options: dict[str, Any], manifest: Path, dataset_root: Path, detector_dir: Path, *, resume: bool
) -> None:
    for fold in range(int(options["fold_count"])):
        output = detector_dir / "folds" / f"fold{fold}"
        if resume and _checkpoint_complete(output):
            print(json.dumps({"skipped_complete_detector_fold": fold}), flush=True)
            continue
        train_detector(
            _detector_namespace(
                options, manifest, dataset_root, output, fold, resume=resume
            )
        )
        _mark_checkpoint_complete(output)


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

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
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
        inputs = {key: value.to(device) for key, value in processor(images=images, return_tensors="pt").items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
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
                    "fold_model": int(record.get("prediction_fold", record.get("fold", -1))),
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
                (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1) / float(width * height)
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
    matches, missed = _match(detections, record["annotations"], float(options["match_iou_threshold"]))
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
    ordered_predictions = [by_key[f"{record['source']}:{record['image_id']}"] for record in calibration]
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


def _train_records(dataset_root: Path) -> list[dict[str, Any]]:
    payload = json.loads((dataset_root / "instances_train2019.json").read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    records = []
    for annotation in payload["annotations"]:
        image = images[int(annotation["image_id"])]
        filename = str(image["file_name"])
        digest = hashlib.sha256(f"train:{image['id']}".encode()).digest()
        records.append(
            {
                "record_type": "detection",
                "source": "rpc_train2019",
                "image_path": f"train2019/{filename}",
                "image_id": int(image["id"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "prediction_fold": int.from_bytes(digest[:4], "big") % 3,
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
        )
    return records


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
    options = config["detector"]
    detector_dir = args.output_dir / "detector"
    manifest = detector_dir / "manifest" / "manifest.jsonl"
    if args.resume and manifest.is_file():
        records = _read_jsonl(manifest)
    else:
        records = build_rpc_detector_manifest(args.dataset_root, detector_dir, int(options["fold_count"]))
    roles = assign_validation_roles(
        records, int(config["experiment"]["expected_num_classes"]), int(config["experiment"]["validation_split_seed"])
    )
    for record in records:
        record["role"] = roles[str(record["capture_session_id"])]
    _write_jsonl(manifest, records)
    train_oof_detectors(options, manifest, args.dataset_root, detector_dir, resume=args.resume)

    val_predictions_path = detector_dir / "predictions" / "val_oof.jsonl"
    if args.resume and val_predictions_path.is_file():
        val_predictions = _read_jsonl(val_predictions_path)
    else:
        val_predictions = []
        for fold in range(int(options["fold_count"])):
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
        _write_jsonl(val_predictions_path, val_predictions)
    threshold_report = select_detector_threshold(records, val_predictions, options)
    _write_json(detector_dir / "threshold.json", threshold_report)
    if not threshold_report["target_recall_satisfied"]:
        raise RuntimeError("RPC detector calibration recall is below the 99% gate")

    train_records = _train_records(args.dataset_root)
    train_predictions_path = detector_dir / "predictions" / "train_assigned.jsonl"
    if args.resume and train_predictions_path.is_file():
        train_predictions = _read_jsonl(train_predictions_path)
    else:
        train_predictions = _predict_partitioned_train(
            train_records,
            detector_dir,
            args.dataset_root,
            int(options["inference_batch_size"]),
            float(options["min_score_threshold"]),
        )
        _write_jsonl(train_predictions_path, train_predictions)
    completed = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": datetime.now(UTC).isoformat(),
        "fold_count": int(options["fold_count"]),
        "validation_images": len(records),
        "train_images": len(train_records),
        "threshold_sha256": sha256_file(detector_dir / "threshold.json"),
        "val_predictions_sha256": sha256_file(val_predictions_path),
        "train_predictions_sha256": sha256_file(train_predictions_path),
    }
    _write_json(detector_dir / "complete.json", completed)
    return completed


def load_worker_gated_records(
    dataset_root: Path, output_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detector_dir = output_dir / "detector"
    threshold = json.loads((detector_dir / "threshold.json").read_text(encoding="utf-8"))
    options = dict(config["detector"])
    options["score_threshold"] = float(threshold["selected_score_threshold"])
    val_base = _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    train_base = _train_records(dataset_root)
    val_predictions = {
        str(item["sample_key"]): item
        for item in _read_jsonl(detector_dir / "predictions" / "val_oof.jsonl")
    }
    train_predictions = {
        str(item["sample_key"]): item
        for item in _read_jsonl(detector_dir / "predictions" / "train_assigned.jsonl")
    }
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    train_rejected: Counter[str] = Counter()
    val_image_outcomes: list[dict[str, Any]] = []

    for record in train_base:
        result = postprocess_worker_gate(
            record, train_predictions[f"{record['source']}:{record['image_id']}"], options
        )
        reasons = list(result["recapture_reasons"])
        if len(result["detections"]) != 1 or len(result["matches"]) != 1 or result["missed_annotation_indices"]:
            reasons.append("DATA_ALIGNMENT_REJECT")
        if reasons:
            train_rejected.update(dict.fromkeys(reasons))
            continue
        detection = result["detections"][0]
        annotation = record["annotations"][0]
        filename = Path(record["image_path"]).name
        prefix, camera_view = filename.rsplit("_camera", 1)
        camera_text, view_text = camera_view.rsplit("-", 1)
        train_rows.append(
            {
                "sample_id": f"train:{record['image_id']}:{annotation['annotation_id']}",
                "split": "train",
                "image_id": int(record["image_id"]),
                "annotation_id": int(annotation["annotation_id"]),
                "image_path": record["image_path"],
                "width": int(record["width"]),
                "height": int(record["height"]),
                "bbox_xyxy": detection["bbox_xyxy"],
                "bbox_xywh": [
                    detection["bbox_xyxy"][0],
                    detection["bbox_xyxy"][1],
                    detection["bbox_xyxy"][2] - detection["bbox_xyxy"][0],
                    detection["bbox_xyxy"][3] - detection["bbox_xyxy"][1],
                ],
                "detector_score": detection["score"],
                "category_id": int(annotation["category_id"]),
                "target": int(annotation["category_id"]) - 1,
                "barcode": prefix,
                "surface": "back" if prefix.endswith("-back") else "front",
                "camera": int(camera_text),
                "view_id": int(Path(view_text).stem),
                "prediction_fold": int(record["prediction_fold"]),
            }
        )

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
                    "annotation_id": None if annotation is None else int(annotation["annotation_id"]),
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
        "train_candidates": len(train_rows),
        "train_rejected": dict(sorted(train_rejected.items())),
        "validation_images": len(val_base),
        "validation_normal_images": sum(not row["recapture_reasons"] for row in val_image_outcomes),
        "validation_recapture_images": sum(bool(row["recapture_reasons"]) for row in val_image_outcomes),
        "validation_recapture_reasons": dict(sorted(reason_counts.items())),
        "validation_missed_boxes": sum(row["missed_count"] for row in val_image_outcomes),
        "validation_unmatched_boxes": sum(row["unmatched_count"] for row in val_image_outcomes),
        "validation_image_outcomes": val_image_outcomes,
    }
    return train_rows, val_rows, report


def _best_detector_epoch(detector_dir: Path, fold_count: int) -> int:
    epochs: list[int] = []
    for fold in range(fold_count):
        history = json.loads(
            (detector_dir / "folds" / f"fold{fold}" / "history.json").read_text(encoding="utf-8")
        )
        eligible = [row for row in history if row.get("validation_loss") is not None]
        if not eligible:
            raise ValueError(f"detector fold {fold} has no validation loss")
        best = min(eligible, key=lambda row: (float(row["validation_loss"]), int(row["epoch"])))
        epochs.append(int(best["epoch"]))
    return max(1, int(round(float(np.median(epochs)))))


def train_final_detector(
    args: argparse.Namespace, config: dict[str, Any], *, resume: bool
) -> Path:
    options = config["detector"]
    detector_dir = args.output_dir / "detector"
    output = detector_dir / "final"
    if resume and _checkpoint_complete(output):
        return output / "best"
    epochs = _best_detector_epoch(detector_dir, int(options["fold_count"]))
    namespace = _detector_namespace(
        options,
        detector_dir / "manifest" / "manifest.jsonl",
        args.dataset_root,
        output,
        0,
        resume=resume,
    )
    namespace.final_training = True
    namespace.epochs = epochs
    namespace.seed = int(options["seed"])
    train_detector(namespace)
    _mark_checkpoint_complete(output)
    _write_json(
        output / "final_training.json",
        {
            "schema_version": SCHEMA_VERSION,
            "epoch_policy": "median_oof_best_epoch",
            "epochs": epochs,
            "trained_on": "val2019_all_groups",
        },
    )
    return output / "best"


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
    args: argparse.Namespace, config: dict[str, Any], *, resume: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train the final detector first, then open and gate test2019 exactly once."""
    checkpoint = train_final_detector(args, config, resume=resume)
    detector_dir = args.output_dir / "detector"
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
    if resume and predictions_path.is_file():
        predictions = _read_jsonl(predictions_path)
    else:
        predictions = predict_records(
            checkpoint,
            test_records,
            args.dataset_root,
            batch_size=int(config["detector"]["inference_batch_size"]),
            minimum_score=float(config["detector"]["min_score_threshold"]),
        )
        _write_jsonl(predictions_path, predictions)
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
                    "annotation_id": None if annotation is None else int(annotation["annotation_id"]),
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
        "detector_checkpoint_sha256": sha256_file(_detector_weights_path(checkpoint)),
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
