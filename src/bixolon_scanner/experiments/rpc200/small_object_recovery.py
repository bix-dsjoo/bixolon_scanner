from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ...configuration import load_json_config
from ...contracts.model_package import load_model_package, sha256_file
from ...training.models import require_torch
from .context_rejector import (
    _detector_features,
    _feature_matrix,
    _onnx_parity,
)
from .data_scale import (
    RpcCachedDataset,
    _build_cache,
    _infer,
    _load_checkpoint_model,
    evaluate_worker_taxonomy,
)
from .worker_gate import postprocess_worker_gate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {key: archive[key] for key in archive.files}


def _validation_row(
    record: dict[str, Any], result: dict[str, Any], detection_index: int, options: dict[str, Any]
) -> dict[str, Any]:
    detection = result["detections"][detection_index]
    match = result["matches"].get(str(detection_index))
    annotation = record["annotations"][int(match[0])] if match is not None else None
    x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
    margin = float(options["border_margin_ratio"])
    touches_border = (
        x1 <= int(record["width"]) * margin
        or y1 <= int(record["height"]) * margin
        or x2 >= int(record["width"]) * (1.0 - margin)
        or y2 >= int(record["height"]) * (1.0 - margin)
    )
    return {
        "sample_id": f"val:{record['image_id']}:det{detection_index}",
        "split": "val",
        "image_id": int(record["image_id"]),
        "annotation_id": None if annotation is None else int(annotation["annotation_id"]),
        "image_path": str(record["image_path"]),
        "width": int(record["width"]),
        "height": int(record["height"]),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "detector_score": float(detection["score"]),
        "category_id": None if annotation is None else int(annotation["category_id"]),
        "target": -1 if annotation is None else int(annotation["category_id"]) - 1,
        "level": str(record["level"]),
        "group_id": str(record["capture_session_id"]),
        "fold": int(record["fold"]),
        "role": str(record["role"]),
        "match_iou": None if match is None else float(match[1]),
        "touches_border": bool(touches_border),
    }


def _worker_records(
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    options: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        result = postprocess_worker_gate(
            record, predictions[f"{record['source']}:{record['image_id']}"], options
        )
        reasons = list(result["recapture_reasons"])
        reason_counts.update(reasons)
        outcomes.append(
            {
                "image_id": int(record["image_id"]),
                "image_path": str(record["image_path"]),
                "fold": int(record["fold"]),
                "role": str(record["role"]),
                "level": str(record["level"]),
                "recapture_reasons": reasons,
                "ground_truth_count": len(record["annotations"]),
                "detection_count": len(result["detections"]),
                "matched_count": len(result["matches"]),
                "missed_count": len(result["missed_annotation_indices"]),
                "unmatched_count": len(result["unmatched_detection_indices"]),
            }
        )
        if reasons:
            continue
        rows.extend(
            _validation_row(record, result, index, options)
            for index in range(len(result["detections"]))
        )
    report = {
        "schema_version": "1.0",
        "score_threshold": float(options["score_threshold"]),
        "min_object_area_ratio": float(options["min_object_area_ratio"]),
        "validation_images": len(records),
        "validation_normal_images": sum(not row["recapture_reasons"] for row in outcomes),
        "validation_recapture_images": sum(bool(row["recapture_reasons"]) for row in outcomes),
        "validation_recapture_reasons": dict(sorted(reason_counts.items())),
        "validation_image_outcomes": outcomes,
    }
    return rows, report


def _merge_archive(
    existing: dict[str, np.ndarray], additions: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    existing_ids = set(existing["sample_ids"].astype(str).tolist())
    addition_mask = np.asarray(
        [str(value) not in existing_ids for value in additions["sample_ids"]],
        dtype=bool,
    )
    merged = {
        key: np.concatenate([existing[key], additions[key][addition_mask]]) for key in existing
    }
    order = np.argsort(merged["sample_ids"].astype(str), kind="stable")
    return {key: value[order] for key, value in merged.items()}


def _infer_missing(
    missing: list[dict[str, Any]],
    *,
    dataset_root: Path,
    cache_dir: Path,
    experiment: dict[str, Any],
    config: dict[str, Any],
    checkpoint: Path,
) -> dict[str, np.ndarray]:
    torch = require_torch()
    if not missing:
        return {
            "logits": np.empty(
                (0, int(config["experiment"]["expected_num_classes"])), dtype=np.float32
            ),
            "targets": np.empty(0, dtype=np.int64),
            "levels": np.empty(0, dtype="<U6"),
            "groups": np.empty(0, dtype="<U16"),
            "sample_ids": np.empty(0, dtype="<U32"),
            "image_ids": np.empty(0, dtype=np.int64),
            "touches_border": np.empty(0, dtype=bool),
        }
    cached = _build_cache(
        dataset_root,
        cache_dir,
        missing,
        experiment,
        config["training"],
        resume=True,
    )
    dataset = RpcCachedDataset(
        cached,
        cache_dir / "images.npy",
        image_size=int(config["training"]["image_size"]),
        training=False,
        include_metadata=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("small-object recovery inference requires CUDA")
    device = torch.device("cuda")
    model, _ = _load_checkpoint_model(checkpoint, config, device)
    result = _infer(model, loader, device)
    del model
    torch.cuda.empty_cache()
    return result


def _build_validation_package(root: Path, min_object_area_ratio: float) -> Path:
    source = root / "validation-candidate-package"
    destination = root / "validation-candidate-package-small-object-v5"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    metadata_path = destination / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["package_version"] = "0.0.0-rpc-small-object-v5"
    metadata["quality"]["min_object_area_ratio"] = float(min_object_area_ratio)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    load_model_package(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-object-area-ratio", type=float, default=0.002)
    args = parser.parse_args()
    if not 0.0 <= args.min_object_area_ratio < 0.005:
        raise ValueError("recovery minimum area ratio must be in [0, 0.005)")

    config = load_json_config(args.config)
    root = args.output_dir
    run_dir = root / "runs" / "full" / f"seed{args.seed}"
    output_dir = run_dir / "context-small-object-v5"
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_dir = root / "detector"
    records = _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    raw_predictions = {
        str(row["sample_key"]): row
        for row in _read_jsonl(detector_dir / "predictions" / "val_oof.jsonl")
    }
    threshold = json.loads((detector_dir / "threshold.json").read_text(encoding="utf-8"))[
        "selected_score_threshold"
    ]
    options = dict(
        config["detector"],
        score_threshold=float(threshold),
        min_object_area_ratio=float(args.min_object_area_ratio),
    )
    all_rows, detector_report = _worker_records(records, raw_predictions, options)
    existing = {
        "calibration": _load_archive(run_dir / "partial_calibration_predictions.npz"),
        "selection": _load_archive(run_dir / "selection_predictions.npz"),
    }
    for name in ("calibration", "selection"):
        recovered_path = output_dir / f"{name}_predictions.npz"
        if recovered_path.exists():
            existing[name] = _merge_archive(existing[name], _load_archive(recovered_path))
    desired = {
        "calibration": [],
        "selection": [row for row in all_rows if row["role"] == "selection"],
    }
    existing_ids = {
        name: set(archive["sample_ids"].astype(str).tolist()) for name, archive in existing.items()
    }
    missing = {
        name: [row for row in rows if row["sample_id"] not in existing_ids[name]]
        for name, rows in desired.items()
    }
    experiment = json.loads((root / "prepared" / "experiment.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "partial.pt"
    merged: dict[str, dict[str, np.ndarray]] = {}
    for name in ("calibration", "selection"):
        additions = _infer_missing(
            missing[name],
            dataset_root=args.dataset_root,
            cache_dir=output_dir / "cache" / name,
            experiment=experiment,
            config=config,
            checkpoint=checkpoint,
        )
        merged[name] = _merge_archive(existing[name], additions)
        desired_ids = {row["sample_id"] for row in desired[name]}
        desired_mask = np.asarray(
            [str(value) in desired_ids for value in merged[name]["sample_ids"]],
            dtype=bool,
        )
        merged[name] = {key: value[desired_mask] for key, value in merged[name].items()}
        np.savez_compressed(output_dir / f"{name}_predictions.npz", **merged[name])

    baseline_context = json.loads(
        (run_dir / "context-rejector" / "report.json").read_text(encoding="utf-8")
    )
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    detector_features = _detector_features(records, raw_predictions, options)
    selection_features = _feature_matrix(merged["selection"], detector_features, temperature)
    model = joblib.load(run_dir / "context-rejector" / "logistic.joblib")
    selection_quality = model.predict_proba(selection_features)[:, 1]
    policy = baseline_context["models"]["logistic"]["policy"]
    policy_calibration = dict(
        calibration,
        approval_threshold=float(policy["classifier_threshold"]),
        risk_control_satisfied=True,
    )
    selection = evaluate_worker_taxonomy(
        merged["selection"],
        policy_calibration,
        detector_report,
        role="selection",
        segment_quality_scores=selection_quality,
        segment_quality_threshold=float(policy["quality_threshold"]),
    )
    joblib.dump(model, output_dir / "logistic.joblib")
    onnx_path = output_dir / "logistic.onnx"
    shutil.copy2(run_dir / "context-rejector" / "logistic.onnx", onnx_path)
    parity = _onnx_parity(
        onnx_path,
        selection_features,
        selection_quality,
        float(policy["quality_threshold"]),
    )
    report = {
        "contract": "rpc-context-small-object-v5",
        "min_object_area_ratio": float(args.min_object_area_ratio),
        "checkpoint_sha256": sha256_file(checkpoint),
        "calibration_existing_count": len(existing["calibration"]["targets"]),
        "calibration_added_count": len(missing["calibration"]),
        "selection_existing_count": len(existing["selection"]["targets"]),
        "selection_added_count": len(missing["selection"]),
        "policy": policy,
        "policy_source": "immutable_context_logistic_v4",
        "selection": selection,
        "onnx_cpu_parity": parity,
        "detector_report": detector_report,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    package_dir = _build_validation_package(root, float(args.min_object_area_ratio))
    print(
        json.dumps(
            {
                "contract": report["contract"],
                "policy": report["policy"],
                "selection": report["selection"],
                "report_path": str(output_dir / "report.json"),
                "package_dir": str(package_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
