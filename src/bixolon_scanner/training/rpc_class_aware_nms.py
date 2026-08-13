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


def _containment(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = max((left[2] - left[0]) * (left[3] - left[1]), 1e-9)
    right_area = max((right[2] - right[0]) * (right[3] - right[1]), 1e-9)
    return intersection / min(left_area, right_area)


def _duplicate_ambiguity_features(
    detections: list[dict[str, Any]], predicted_classes: list[int]
) -> tuple[list[float], list[bool]]:
    maximum_containment: list[float] = []
    has_higher_score_duplicate: list[bool] = []
    for index, detection in enumerate(detections):
        candidates = [
            other
            for other in range(len(detections))
            if other != index
            and predicted_classes[other] == predicted_classes[index]
            and float(detections[other]["score"]) > float(detection["score"])
        ]
        maximum_containment.append(
            max(
                (
                    _containment(
                        detection["bbox_xyxy"], detections[other]["bbox_xyxy"]
                    )
                    for other in candidates
                ),
                default=0.0,
            )
        )
        has_higher_score_duplicate.append(bool(candidates))
    return maximum_containment, has_higher_score_duplicate


def _duplicate_ambiguity_mask(
    containment: list[float] | np.ndarray,
    repeated: list[bool] | np.ndarray,
    ranks: list[float] | np.ndarray,
    scores: list[float] | np.ndarray,
    quality: list[float] | np.ndarray,
    *,
    context_threshold: float,
    overlap_threshold: float | None,
    overlap_max_score: float | None,
    low_quality_multiplier: float | None,
    minimum_rank: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    containment_values = np.asarray(containment, dtype=np.float64)
    repeated_values = np.asarray(repeated, dtype=bool)
    rank_values = np.asarray(ranks, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64)
    quality_values = np.asarray(quality, dtype=np.float64)
    rank_eligible = rank_values >= minimum_rank
    overlap = np.zeros(len(rank_values), dtype=bool)
    if overlap_threshold is not None:
        overlap = rank_eligible & (containment_values >= overlap_threshold)
        if overlap_max_score is not None:
            overlap &= score_values <= overlap_max_score
    low_quality = np.zeros(len(rank_values), dtype=bool)
    if low_quality_multiplier is not None:
        low_quality = (
            rank_eligible
            & repeated_values
            & (quality_values < context_threshold * low_quality_multiplier)
        )
    return overlap | low_quality, overlap, low_quality


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
    duplicate_overlap_threshold: float | None = None,
    duplicate_overlap_max_score: float | None = None,
    duplicate_low_quality_multiplier: float | None = None,
    duplicate_min_rank: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample_index = {
        str(sample_id): index
        for index, sample_id in enumerate(archive["sample_ids"])
    }
    selected_indices: list[int] = []
    new_targets: dict[int, int] = {}
    geometry: dict[str, list[float]] = {}
    duplicate_containment: dict[int, float] = {}
    higher_score_duplicate: dict[int, bool] = {}
    detector_rank: dict[int, float] = {}
    detector_score: dict[int, float] = {}
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
        kept_classes = [predicted[index] for index in kept_detection_indices]
        containment, repeated = _duplicate_ambiguity_features(
            kept_detections, kept_classes
        )
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
            duplicate_containment[archive_index] = containment[local_index]
            higher_score_duplicate[archive_index] = repeated[local_index]
            detector_rank[archive_index] = local_index / max(
                len(kept_detection_indices) - 1, 1
            )
            detector_score[archive_index] = float(
                kept_detections[local_index]["score"]
            )
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
    probabilities = softmax(filtered["logits"], float(calibration["temperature"]))
    predicted = probabilities.argmax(axis=1)
    top2 = np.argsort(-probabilities, axis=1)[:, :2]
    extra_recapture, overlap_recapture, low_quality_recapture = (
        _duplicate_ambiguity_mask(
            [duplicate_containment[int(index)] for index in selected],
            [higher_score_duplicate[int(index)] for index in selected],
            [detector_rank[int(index)] for index in selected],
            [detector_score[int(index)] for index in selected],
            quality,
            context_threshold=context_threshold,
            overlap_threshold=duplicate_overlap_threshold,
            overlap_max_score=duplicate_overlap_max_score,
            low_quality_multiplier=duplicate_low_quality_multiplier,
            minimum_rank=duplicate_min_rank,
        )
    )
    effective_quality = quality.copy()
    effective_quality[extra_recapture] = -np.inf
    metrics = evaluate_worker_taxonomy(
        filtered,
        calibration,
        detector_report,
        role=role,
        segment_quality_scores=effective_quality,
        segment_quality_threshold=context_threshold,
    )
    confidence = probabilities.max(axis=1)
    attribution: dict[str, Any] = {
        "duplicate_ambiguity_recapture_count": int(extra_recapture.sum()),
        "duplicate_overlap_recapture_count": int(overlap_recapture.sum()),
        "duplicate_low_quality_recapture_count": int(low_quality_recapture.sum()),
        "examples": [],
    }
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
            | (effective_quality < context_threshold)
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
        for index in np.flatnonzero(
            approved & matched & (predicted != filtered["targets"])
        ):
            attribution["examples"].append(
                {
                    "level": level,
                    "sample_id": str(filtered["sample_ids"][index]),
                    "target": int(filtered["targets"][index]),
                    "predicted": int(predicted[index]),
                    "top2": [int(value) for value in top2[index]],
                    "confidence": float(confidence[index]),
                    "quality_score": float(quality[index]),
                    "duplicate_containment": float(
                        duplicate_containment[int(selected[index])]
                    ),
                    "has_higher_score_duplicate": bool(
                        higher_score_duplicate[int(selected[index])]
                    ),
                }
            )
    return metrics, suppressed, attribution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--class-aware-nms-threshold", type=float, default=0.55)
    parser.add_argument("--duplicate-overlap-threshold", type=float)
    parser.add_argument("--duplicate-overlap-max-score", type=float)
    parser.add_argument("--duplicate-low-quality-multiplier", type=float)
    parser.add_argument("--duplicate-min-rank", type=float, default=0.0)
    parser.add_argument("--report-version", default="v8")
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
        "contract": f"rpc-class-aware-nms-{args.report_version}-diagnostic",
        "class_aware_nms_threshold": float(args.class_aware_nms_threshold),
        "policy_source": (
            "validation_failure_analysis"
            if args.duplicate_overlap_threshold is not None
            else "calibration_only"
        ),
        "duplicate_ambiguity_policy": {
            "overlap_threshold": args.duplicate_overlap_threshold,
            "overlap_max_detector_score": args.duplicate_overlap_max_score,
            "low_quality_multiplier": args.duplicate_low_quality_multiplier,
            "minimum_detector_rank": float(args.duplicate_min_rank),
        },
    }
    archives: dict[str, dict[str, np.ndarray]] = {}
    for role, filename in (
        ("calibration", "partial_calibration_predictions.npz"),
        ("selection", "selection_predictions.npz"),
    ):
        loaded = np.load(run_dir / filename)
        archives[role] = {key: loaded[key] for key in loaded.files}

    for role in ("calibration", "selection"):
        metrics, suppressed, attribution = _evaluate_role(
            role=role,
            archive=archives[role],
            records=records,
            predictions=predictions,
            detector_options=detector_options,
            detector_report=copy.deepcopy(detector_report),
            calibration=policy_calibration,
            context_session=context_session,
            context_threshold=float(context_report["quality_threshold"]),
            nms_threshold=float(args.class_aware_nms_threshold),
            duplicate_overlap_threshold=args.duplicate_overlap_threshold,
            duplicate_overlap_max_score=args.duplicate_overlap_max_score,
            duplicate_low_quality_multiplier=(
                args.duplicate_low_quality_multiplier
            ),
            duplicate_min_rank=float(args.duplicate_min_rank),
        )
        output[role] = {
            "metrics": metrics,
            "suppressed": suppressed,
            "wrong_approval_attribution": attribution,
        }
    output_dir = run_dir / f"class-aware-nms-{args.report_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
