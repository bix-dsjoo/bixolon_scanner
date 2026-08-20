from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ...configuration import load_json_config
from ...pipeline.ports import Detection
from ...training.calibration import softmax
from .context_rejector import _feature_matrix, _geometry_features, _read_jsonl
from .data_scale import evaluate_worker_taxonomy
from .worker_gate import _iou, _match, postprocess_worker_gate

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
            and _iou(detection["bbox_xyxy"], detections[accepted]["bbox_xyxy"]) > threshold
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
                    _containment(detection["bbox_xyxy"], detections[other]["bbox_xyxy"])
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
    overlap_max_quality: float | None,
    low_quality_multiplier: float | None,
    low_quality_max_quality: float | None,
    low_quality_min_score: float | None,
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
        if overlap_max_quality is not None:
            overlap &= quality_values <= overlap_max_quality
    low_quality = np.zeros(len(rank_values), dtype=bool)
    if low_quality_multiplier is not None or low_quality_max_quality is not None:
        maximum_quality = (
            context_threshold * low_quality_multiplier
            if low_quality_max_quality is None
            else low_quality_max_quality
        )
        low_quality = rank_eligible & repeated_values & (quality_values < maximum_quality)
        if low_quality_min_score is not None:
            low_quality &= score_values >= low_quality_min_score
    return overlap | low_quality, overlap, low_quality


def _assignment_conflict_mask(
    image_ids: list[int] | np.ndarray,
    top_classes: list[list[int]] | np.ndarray,
    detector_scores: list[float] | np.ndarray,
    detector_ranks: list[float] | np.ndarray,
    confidence: list[float] | np.ndarray,
    quality: list[float] | np.ndarray,
    *,
    minimum_duplicate_rank: float,
    minimum_mutual_confidence: float = 0.0,
    minimum_mutual_quality: float = 0.0,
    enable_duplicate_alternative: bool = True,
    mutual_class_pairs: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find label-assignment conflicts without consulting ground truth.

    ``mutual_swap`` identifies two ROIs whose Top-1 labels differ while each
    ROI contains the other's Top-1 in its remaining candidate list.
    ``duplicate_alternative`` identifies a lower detector-score duplicate
    whose alternative candidate is already the Top-1 of another ROI in the
    same image.  The latter is rank-gated so that a strong primary detection
    is never rejected merely because the checkout image contains variants of
    the same product family.
    """

    images = np.asarray(image_ids, dtype=np.int64)
    candidates = np.asarray(top_classes, dtype=np.int64)
    scores = np.asarray(detector_scores, dtype=np.float64)
    ranks = np.asarray(detector_ranks, dtype=np.float64)
    confidence_values = np.asarray(confidence, dtype=np.float64)
    quality_values = np.asarray(quality, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] < 2:
        raise ValueError("assignment conflict requires at least Top-2 classes")
    if not (
        len(images)
        == len(candidates)
        == len(scores)
        == len(ranks)
        == len(confidence_values)
        == len(quality_values)
    ):
        raise ValueError("assignment conflict inputs must have equal lengths")

    mutual_swap = np.zeros(len(images), dtype=bool)
    duplicate_alternative = np.zeros(len(images), dtype=bool)
    for image_id in np.unique(images):
        group = np.flatnonzero(images == image_id)
        for index in group:
            top1 = int(candidates[index, 0])
            alternatives = set(int(value) for value in candidates[index, 1:])
            for other in group:
                if other == index:
                    continue
                other_top1 = int(candidates[other, 0])
                pair = tuple(sorted((top1, other_top1)))
                if (
                    other_top1 != top1
                    and (mutual_class_pairs is None or pair in mutual_class_pairs)
                    and other_top1 in alternatives
                    and top1 in set(int(value) for value in candidates[other, 1:])
                    and confidence_values[index] >= minimum_mutual_confidence
                    and confidence_values[other] >= minimum_mutual_confidence
                    and quality_values[index] >= minimum_mutual_quality
                    and quality_values[other] >= minimum_mutual_quality
                ):
                    mutual_swap[index] = True
            has_higher_duplicate = any(
                other != index
                and int(candidates[other, 0]) == top1
                and scores[other] > scores[index]
                for other in group
            )
            alternative_is_present = any(
                other != index and int(candidates[other, 0]) in alternatives for other in group
            )
            duplicate_alternative[index] = (
                enable_duplicate_alternative
                and ranks[index] >= minimum_duplicate_rank
                and has_higher_duplicate
                and alternative_is_present
            )
    return (
        mutual_swap | duplicate_alternative,
        mutual_swap,
        duplicate_alternative,
    )


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
    duplicate_overlap_max_quality: float | None = None,
    duplicate_low_quality_multiplier: float | None = None,
    duplicate_low_quality_max_quality: float | None = None,
    duplicate_low_quality_min_score: float | None = None,
    duplicate_min_rank: float = 0.0,
    assignment_conflict_top_k: int | None = None,
    assignment_mutual_min_confidence: float = 0.0,
    assignment_mutual_min_quality: float = 0.0,
    assignment_mutual_only: bool = False,
    assignment_mutual_class_pairs: set[tuple[int, int]] | None = None,
    ambiguity_outcome: str = "segment_recapture",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if ambiguity_outcome not in {"segment_recapture", "unknown"}:
        raise ValueError("ambiguity outcome must be segment_recapture or unknown")
    sample_index = {str(sample_id): index for index, sample_id in enumerate(archive["sample_ids"])}
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
        expected_ids = [f"val:{record['image_id']}:det{index}" for index in range(len(detections))]
        if not all(sample_id in sample_index for sample_id in expected_ids):
            continue
        original_indices = [sample_index[sample_id] for sample_id in expected_ids]
        predicted = [int(archive["logits"][index].argmax()) for index in original_indices]
        kept_detection_indices = _keep_indices(detections, predicted, nms_threshold)
        kept_detections = [detections[index] for index in kept_detection_indices]
        kept_classes = [predicted[index] for index in kept_detection_indices]
        containment, repeated = _duplicate_ambiguity_features(kept_detections, kept_classes)
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
            detector_rank[archive_index] = local_index / max(len(kept_detection_indices) - 1, 1)
            detector_score[archive_index] = float(kept_detections[local_index]["score"])
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
    features = _feature_matrix(filtered, geometry, float(calibration["temperature"]))
    quality = context_session.run(["quality_score"], {"features": features.astype(np.float32)})[
        0
    ].reshape(-1)
    probabilities = softmax(filtered["logits"], float(calibration["temperature"]))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    top2 = np.argsort(-probabilities, axis=1)[:, :2]
    extra_recapture, overlap_recapture, low_quality_recapture = _duplicate_ambiguity_mask(
        [duplicate_containment[int(index)] for index in selected],
        [higher_score_duplicate[int(index)] for index in selected],
        [detector_rank[int(index)] for index in selected],
        [detector_score[int(index)] for index in selected],
        quality,
        context_threshold=context_threshold,
        overlap_threshold=duplicate_overlap_threshold,
        overlap_max_score=duplicate_overlap_max_score,
        overlap_max_quality=duplicate_overlap_max_quality,
        low_quality_multiplier=duplicate_low_quality_multiplier,
        low_quality_max_quality=duplicate_low_quality_max_quality,
        low_quality_min_score=duplicate_low_quality_min_score,
        minimum_rank=duplicate_min_rank,
    )
    assignment_recapture = np.zeros(len(filtered["targets"]), dtype=bool)
    mutual_swap_recapture = np.zeros(len(filtered["targets"]), dtype=bool)
    duplicate_alternative_recapture = np.zeros(len(filtered["targets"]), dtype=bool)
    if assignment_conflict_top_k is not None:
        top_classes = np.argsort(-probabilities, axis=1)[:, :assignment_conflict_top_k]
        (
            assignment_recapture,
            mutual_swap_recapture,
            duplicate_alternative_recapture,
        ) = _assignment_conflict_mask(
            filtered["image_ids"],
            top_classes,
            [detector_score[int(index)] for index in selected],
            [detector_rank[int(index)] for index in selected],
            confidence,
            quality,
            minimum_duplicate_rank=duplicate_min_rank,
            minimum_mutual_confidence=assignment_mutual_min_confidence,
            minimum_mutual_quality=assignment_mutual_min_quality,
            enable_duplicate_alternative=not assignment_mutual_only,
            mutual_class_pairs=assignment_mutual_class_pairs,
        )
    extra_recapture |= assignment_recapture
    effective_quality = quality.copy()
    force_unknown = np.zeros(len(filtered["targets"]), dtype=bool)
    if ambiguity_outcome == "segment_recapture":
        effective_quality[extra_recapture] = -np.inf
    else:
        force_unknown = extra_recapture
    metrics = evaluate_worker_taxonomy(
        filtered,
        calibration,
        detector_report,
        role=role,
        segment_quality_scores=effective_quality,
        segment_quality_threshold=context_threshold,
        force_unknown_mask=force_unknown,
    )
    attribution: dict[str, Any] = {
        "ambiguity_action_count": int(extra_recapture.sum()),
        "duplicate_ambiguity_recapture_count": (
            int(extra_recapture.sum()) if ambiguity_outcome == "segment_recapture" else 0
        ),
        "ambiguity_unknown_count": (
            int(extra_recapture.sum()) if ambiguity_outcome == "unknown" else 0
        ),
        "duplicate_overlap_recapture_count": int(overlap_recapture.sum()),
        "duplicate_low_quality_recapture_count": int(low_quality_recapture.sum()),
        "assignment_conflict_recapture_count": int(assignment_recapture.sum()),
        "mutual_swap_recapture_count": int(mutual_swap_recapture.sum()),
        "duplicate_alternative_recapture_count": int(duplicate_alternative_recapture.sum()),
        "ambiguity_outcome": ambiguity_outcome,
        "examples": [],
    }
    for level in LEVELS:
        level_outcomes = [
            row
            for row in detector_report["validation_image_outcomes"]
            if row["role"] == role and row["level"] == level
        ]
        level_ids = {int(row["image_id"]) for row in level_outcomes}
        recapture_ids = {int(row["image_id"]) for row in level_outcomes if row["recapture_reasons"]}
        level_mask = np.asarray([int(value) in level_ids for value in filtered["image_ids"]])
        normal = np.asarray([int(value) not in recapture_ids for value in filtered["image_ids"]])
        segment_recapture = (
            level_mask
            & normal
            & (
                (
                    filtered["touches_border"].astype(bool)
                    & (confidence < float(calibration["approval_threshold"]))
                )
                | (effective_quality < context_threshold)
            )
        )
        approved = (
            level_mask
            & normal
            & ~segment_recapture
            & ~force_unknown
            & (confidence >= float(calibration["approval_threshold"]))
        )
        matched = filtered["targets"] >= 0
        attribution[level] = {
            "approved_unmatched_count": int((approved & ~matched).sum()),
            "approved_misclassification_count": int(
                (approved & matched & (predicted != filtered["targets"])).sum()
            ),
        }
        for index in np.flatnonzero(approved & matched & (predicted != filtered["targets"])):
            attribution["examples"].append(
                {
                    "level": level,
                    "sample_id": str(filtered["sample_ids"][index]),
                    "target": int(filtered["targets"][index]),
                    "predicted": int(predicted[index]),
                    "top2": [int(value) for value in top2[index]],
                    "confidence": float(confidence[index]),
                    "quality_score": float(quality[index]),
                    "duplicate_containment": float(duplicate_containment[int(selected[index])]),
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
    parser.add_argument("--duplicate-overlap-max-quality", type=float)
    parser.add_argument("--duplicate-low-quality-multiplier", type=float)
    parser.add_argument("--duplicate-low-quality-max-quality", type=float)
    parser.add_argument("--duplicate-low-quality-min-score", type=float)
    parser.add_argument("--duplicate-min-rank", type=float, default=0.0)
    parser.add_argument("--context-threshold-override", type=float)
    parser.add_argument("--assignment-conflict-top-k", type=int)
    parser.add_argument("--assignment-mutual-min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--ambiguity-outcome",
        choices=("segment_recapture", "unknown"),
        default="segment_recapture",
    )
    parser.add_argument("--assignment-mutual-min-quality", type=float, default=0.0)
    parser.add_argument("--assignment-mutual-only", action="store_true")
    parser.add_argument(
        "--assignment-mutual-pair",
        action="append",
        default=[],
        metavar="CLASS_A:CLASS_B",
    )
    parser.add_argument("--report-version", default="v8")
    args = parser.parse_args()
    mutual_pairs: set[tuple[int, int]] | None = None
    if args.assignment_mutual_pair:
        try:
            mutual_pairs = {
                tuple(sorted(int(value) for value in raw.split(":")))
                for raw in args.assignment_mutual_pair
            }
        except ValueError as error:
            parser.error(f"invalid --assignment-mutual-pair: {error}")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in mutual_pairs):
            parser.error("--assignment-mutual-pair requires two different classes")
    config = load_json_config(args.config)
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
    context_threshold = (
        float(context_report["quality_threshold"])
        if args.context_threshold_override is None
        else float(args.context_threshold_override)
    )
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
            "overlap_max_context_quality": args.duplicate_overlap_max_quality,
            "low_quality_multiplier": args.duplicate_low_quality_multiplier,
            "low_quality_max_context_quality": (args.duplicate_low_quality_max_quality),
            "low_quality_min_detector_score": args.duplicate_low_quality_min_score,
            "minimum_detector_rank": float(args.duplicate_min_rank),
            "assignment_conflict_top_k": args.assignment_conflict_top_k,
            "assignment_mutual_min_confidence": (args.assignment_mutual_min_confidence),
            "assignment_mutual_min_quality": args.assignment_mutual_min_quality,
            "assignment_mutual_only": bool(args.assignment_mutual_only),
            "assignment_mutual_class_pairs": (
                None if mutual_pairs is None else sorted(mutual_pairs)
            ),
            "ambiguity_outcome": args.ambiguity_outcome,
            "context_quality_threshold": context_threshold,
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
            context_threshold=context_threshold,
            nms_threshold=float(args.class_aware_nms_threshold),
            duplicate_overlap_threshold=args.duplicate_overlap_threshold,
            duplicate_overlap_max_score=args.duplicate_overlap_max_score,
            duplicate_overlap_max_quality=args.duplicate_overlap_max_quality,
            duplicate_low_quality_multiplier=(args.duplicate_low_quality_multiplier),
            duplicate_low_quality_max_quality=(args.duplicate_low_quality_max_quality),
            duplicate_low_quality_min_score=args.duplicate_low_quality_min_score,
            duplicate_min_rank=float(args.duplicate_min_rank),
            assignment_conflict_top_k=args.assignment_conflict_top_k,
            assignment_mutual_min_confidence=float(args.assignment_mutual_min_confidence),
            assignment_mutual_min_quality=float(args.assignment_mutual_min_quality),
            assignment_mutual_only=bool(args.assignment_mutual_only),
            assignment_mutual_class_pairs=mutual_pairs,
            ambiguity_outcome=args.ambiguity_outcome,
        )
        output[role] = {
            "metrics": metrics,
            "suppressed": suppressed,
            "wrong_approval_attribution": attribution,
        }
    output_dir = run_dir / f"class-aware-nms-{args.report_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
