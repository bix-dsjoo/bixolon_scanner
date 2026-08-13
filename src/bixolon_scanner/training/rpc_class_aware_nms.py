from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ..inference import Detection
from .calibration import softmax
from .rpc_context_rejector import _feature_matrix, _geometry_features, _read_jsonl
from .rpc_data_scale import evaluate_worker_taxonomy
from .rpc_worker_gate import _iou, _match, postprocess_worker_gate


LEVELS = ("easy", "medium", "hard")


def _keep_indices(
    detections: list[dict[str, Any]],
    predicted_classes: list[int],
    threshold: float,
) -> list[int]:
    kept: list[int] = []
    for index, detection in enumerate(detections):
        if any(
            predicted_classes[index] == predicted_classes[accepted]
            and _iou(
                detection["bbox_xyxy"], detections[accepted]["bbox_xyxy"]
            )
            > threshold
            for accepted in kept
        ):
            continue
        kept.append(index)
    return kept


def _evaluate_role(
    *,
    role: str,
    archive: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    detector_options: dict[str, Any],
    detector_report: dict[str, Any],
    calibration: dict[str, Any],
    context_session: ort.InferenceSession,
    context_threshold: float,
    nms_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample_index = {
        str(sample_id): index
        for index, sample_id in enumerate(archive["sample_ids"])
    }
    selected_indices: list[int] = []
    new_targets: dict[int, int] = {}
    geometry: dict[str, list[float]] = {}
    outcomes_by_id = {
        int(row["image_id"]): row
        for row in detector_report["validation_image_outcomes"]
        if row["role"] == role
    }
    suppressed = {level: {"matched": 0, "unmatched": 0} for level in LEVELS}
    for record in (row for row in records if row["role"] == role):
        outcome = outcomes_by_id[int(record["image_id"])]
        if outcome["recapture_reasons"]:
            continue
        result = postprocess_worker_gate(
            record,
            predictions[f"{record['source']}:{record['image_id']}"],
            detector_options,
        )
        detections = result["detections"]
        expected_ids = [
            f"val:{record['image_id']}:det{index}"
            for index in range(len(detections))
        ]
        if not all(sample_id in sample_index for sample_id in expected_ids):
            continue
        original_indices = [
            sample_index[sample_id] for sample_id in expected_ids
        ]
        predicted = [
            int(archive["logits"][index].argmax()) for index in original_indices
        ]
        kept_detection_indices = _keep_indices(detections, predicted, nms_threshold)
        kept_detections = [
            detections[index] for index in kept_detection_indices
        ]
        old_matches = result["matches"]
        for index in set(range(len(detections))) - set(kept_detection_indices):
            kind = "matched" if str(index) in old_matches else "unmatched"
            suppressed[str(record["level"])][kind] += 1
        matches, missed = _match(
            [
                Detection(
                    *[float(value) for value in item["bbox_xyxy"]],
                    float(item["score"]),
                )
                for item in kept_detections
            ],
            record["annotations"],
            float(detector_options["match_iou_threshold"]),
        )
        for local_index, original_detection_index in enumerate(kept_detection_indices):
            archive_index = original_indices[original_detection_index]
            selected_indices.append(archive_index)
            match = matches.get(local_index)
            new_targets[archive_index] = (
                -1
                if match is None
                else int(record["annotations"][int(match[0])]["category_id"]) - 1
            )
            sample_id = str(archive["sample_ids"][archive_index])
            geometry[sample_id] = _geometry_features(
                kept_detections,
                float(record["width"]),
                float(record["height"]),
                local_index,
            )
        outcome["detection_count"] = len(kept_detections)
        outcome["matched_count"] = len(matches)
        outcome["missed_count"] = len(missed)
        outcome["unmatched_count"] = len(kept_detections) - len(matches)
    selected = np.asarray(sorted(selected_indices), dtype=np.int64)
    filtered = {key: value[selected] for key, value in archive.items()}
    filtered["targets"] = np.asarray(
        [new_targets[int(index)] for index in selected], dtype=np.int64
    )
    features = _feature_matrix(
        filtered, geometry, float(calibration["temperature"])
    )
    quality = context_session.run(
        ["quality_score"], {"features": features.astype(np.float32)}
    )[0].reshape(-1)
    metrics = evaluate_worker_taxonomy(
        filtered,
        calibration,
        detector_report,
        role=role,
        segment_quality_scores=quality,
        segment_quality_threshold=context_threshold,
    )
    probabilities = softmax(filtered["logits"], float(calibration["temperature"]))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    attribution: dict[str, Any] = {}
    for level in LEVELS:
        level_outcomes = [
            row
            for row in detector_report["validation_image_outcomes"]
            if row["role"] == role and row["level"] == level
        ]
        level_ids = {int(row["image_id"]) for row in level_outcomes}
        recapture_ids = {
            int(row["image_id"])
            for row in level_outcomes
            if row["recapture_reasons"]
        }
        level_mask = np.asarray(
            [int(value) in level_ids for value in filtered["image_ids"]]
        )
        normal = np.asarray(
            [int(value) not in recapture_ids for value in filtered["image_ids"]]
        )
        segment_recapture = level_mask & normal & (
            (
                filtered["touches_border"].astype(bool)
                & (confidence < float(calibration["approval_threshold"]))
            )
            | (quality < context_threshold)
        )
        approved = (
            level_mask
            & normal
            & ~segment_recapture
            & (confidence >= float(calibration["approval_threshold"]))
        )
        matched = filtered["targets"] >= 0
        attribution[level] = {
            "approved_unmatched_count": int((approved & ~matched).sum()),
            "approved_misclassification_count": int(
                (approved & matched & (predicted != filtered["targets"])).sum()
            ),
        }
    return metrics, suppressed, attribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--class-aware-nms-threshold", type=float, default=0.55)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.output_dir
    run_dir = root / "runs" / "full" / f"seed{args.seed}"
    detector_dir = root / "detector"
    records = _read_jsonl(detector_dir / "manifest" / "manifest.jsonl")
    predictions = {
        str(row["sample_key"]): row
        for row in _read_jsonl(detector_dir / "predictions" / "val_oof.jsonl")
    }
    score_threshold = float(
        json.loads((detector_dir / "threshold.json").read_text(encoding="utf-8"))[
            "selected_score_threshold"
        ]
    )
    detector_options = dict(config["detector"], score_threshold=score_threshold)
    detector_report = json.loads(
        (root / "prepared" / "worker_gate_report.json").read_text(encoding="utf-8")
    )
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    context_report = json.loads(
        (run_dir / "context-rejector" / "report.json").read_text(encoding="utf-8")
    )["models"]["logistic"]["policy"]
    policy_calibration = dict(
        calibration,
        approval_threshold=float(context_report["classifier_threshold"]),
        risk_control_satisfied=True,
    )
    context_session = ort.InferenceSession(
        str(run_dir / "context-rejector" / "logistic.onnx"),
        providers=["CPUExecutionProvider"],
    )
    output: dict[str, Any] = {
        "contract": "rpc-class-aware-nms-v8-diagnostic",
        "class_aware_nms_threshold": float(args.class_aware_nms_threshold),
        "policy_source": "calibration_only",
    }
    for role, filename in (
        ("calibration", "partial_calibration_predictions.npz"),
        ("selection", "selection_predictions.npz"),
    ):
        loaded = np.load(run_dir / filename)
        archive = {key: loaded[key] for key in loaded.files}
        metrics, suppressed, attribution = _evaluate_role(
            role=role,
            archive=archive,
            records=records,
            predictions=predictions,
            detector_options=detector_options,
            detector_report=copy.deepcopy(detector_report),
            calibration=policy_calibration,
            context_session=context_session,
            context_threshold=float(context_report["quality_threshold"]),
            nms_threshold=float(args.class_aware_nms_threshold),
        )
        output[role] = {
            "metrics": metrics,
            "suppressed": suppressed,
            "wrong_approval_attribution": attribution,
        }
    output_dir = run_dir / "class-aware-nms-v8"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
