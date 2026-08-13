from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..inference import Detection, _box_iou
from .calibration import binomial_rate_upper_bound
from .evaluate_detector import _iou, _xywh_to_xyxy


TARGET_MODE_VERSION = "0.2.5"


@dataclass(frozen=True)
class DetectorPolicy:
    score_threshold: float
    nms_iou_threshold: float
    uncertainty_score_threshold: float | None
    uncertainty_min_area_ratio: float
    uncertainty_match_iou_threshold: float = 0.5
    min_object_area_ratio: float = 0.005
    max_queries: int = 300

    def validate(self) -> None:
        rates = (
            self.score_threshold,
            self.nms_iou_threshold,
            self.uncertainty_min_area_ratio,
            self.uncertainty_match_iou_threshold,
            self.min_object_area_ratio,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("detector policy rates must be within [0, 1]")
        if self.max_queries < 1:
            raise ValueError("max_queries must be positive")
        if self.uncertainty_score_threshold is not None:
            if not 0.0 <= self.uncertainty_score_threshold < self.score_threshold:
                raise ValueError(
                    "uncertainty_score_threshold must be below score_threshold"
                )

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class IndexedDetection:
    index: int
    detection: Detection


def policy_grid(config: dict[str, Any]) -> list[DetectorPolicy]:
    policies: list[DetectorPolicy] = []
    uncertainty_scores: Sequence[float | None] = config[
        "uncertainty_score_thresholds"
    ]
    for score in config["score_thresholds"]:
        for nms in config["nms_iou_thresholds"]:
            for uncertainty_score in uncertainty_scores:
                areas = (
                    [0.0]
                    if uncertainty_score is None
                    else config["uncertainty_min_area_ratios"]
                )
                for area in areas:
                    policy = DetectorPolicy(
                        score_threshold=float(score),
                        nms_iou_threshold=float(nms),
                        uncertainty_score_threshold=(
                            None
                            if uncertainty_score is None
                            else float(uncertainty_score)
                        ),
                        uncertainty_min_area_ratio=float(area),
                        uncertainty_match_iou_threshold=float(
                            config["uncertainty_match_iou_threshold"]
                        ),
                        min_object_area_ratio=float(config["min_object_area_ratio"]),
                        max_queries=int(config["max_queries"]),
                    )
                    if (
                        policy.uncertainty_score_threshold is not None
                        and policy.uncertainty_score_threshold
                        >= policy.score_threshold
                    ):
                        continue
                    policy.validate()
                    policies.append(policy)
    return sorted({policy.key: policy for policy in policies}.values(), key=lambda p: p.key)


def _indexed_nms(
    detections: list[IndexedDetection], threshold: float
) -> list[IndexedDetection]:
    ordered = sorted(detections, key=lambda item: item.detection.score, reverse=True)
    selected: list[IndexedDetection] = []
    for candidate in ordered:
        if all(
            _box_iou(candidate.detection, accepted.detection) <= threshold
            for accepted in selected
        ):
            selected.append(candidate)
    return selected


def _raw_detections(prediction: dict[str, Any]) -> list[IndexedDetection]:
    return [
        IndexedDetection(
            index=index,
            detection=Detection(
                *[float(value) for value in box], score=float(score)
            ),
        )
        for index, (box, score) in enumerate(
            zip(prediction["boxes_xyxy"], prediction["scores"])
        )
    ]


def _apply_policy_to_raw(
    record: dict[str, Any],
    prediction: dict[str, Any],
    policy: DetectorPolicy,
    raw: list[IndexedDetection],
    nms_for_score: Any,
) -> dict[str, Any]:
    accepted_raw = [
        item for item in raw if item.detection.score >= policy.score_threshold
    ]
    detections = nms_for_score(policy.score_threshold, policy.nms_iou_threshold)
    reasons = list(prediction.get("fixed_hard_reason_codes", ()))
    if len(accepted_raw) >= policy.max_queries:
        reasons.append("DETECTOR_CAPACITY_EXCEEDED")
    if not detections:
        reasons.append("DETECTOR_NO_OBJECT")
    image_area = float(int(record["width"]) * int(record["height"]))
    if any(
        (item.detection.x2 - item.detection.x1)
        * (item.detection.y2 - item.detection.y1)
        / image_area
        < policy.min_object_area_ratio
        for item in detections
    ):
        reasons.append("DETECTOR_OBJECT_TOO_SMALL")
    uncertain_indices: list[int] = []
    if policy.uncertainty_score_threshold is not None:
        shadow = nms_for_score(
            policy.uncertainty_score_threshold, policy.nms_iou_threshold
        )
        for candidate in shadow:
            if candidate.detection.score >= policy.score_threshold:
                continue
            area_ratio = (
                (candidate.detection.x2 - candidate.detection.x1)
                * (candidate.detection.y2 - candidate.detection.y1)
                / image_area
            )
            if area_ratio < policy.uncertainty_min_area_ratio:
                continue
            overlaps = [
                _box_iou(candidate.detection, accepted.detection)
                for accepted in detections
            ]
            if (
                not overlaps
                or max(overlaps) < policy.uncertainty_match_iou_threshold
            ):
                uncertain_indices.append(candidate.index)
        if uncertain_indices:
            reasons.append("DETECTOR_UNCERTAIN_OBJECT")
    return {
        "detections": detections,
        "pass": not reasons,
        "reason_codes": list(dict.fromkeys(reasons)),
        "uncertain_indices": uncertain_indices,
    }


class PolicyEvaluationCache:
    """Reuse model/image post-processing that is invariant across policy families."""

    def __init__(self, predictions: Sequence[dict[str, Any]]) -> None:
        self._raw = [_raw_detections(prediction) for prediction in predictions]
        self._nms: dict[tuple[int, float, float], list[IndexedDetection]] = {}
        self._diagnostics: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._raw)

    def _nms_for_score(
        self, image_index: int, score: float, nms_iou: float
    ) -> list[IndexedDetection]:
        key = (image_index, float(score), float(nms_iou))
        cached = self._nms.get(key)
        if cached is None:
            cached = _indexed_nms(
                [
                    item
                    for item in self._raw[image_index]
                    if item.detection.score >= score
                ],
                nms_iou,
            )
            self._nms[key] = cached
        return cached

    def apply(
        self,
        image_index: int,
        record: dict[str, Any],
        prediction: dict[str, Any],
        policy: DetectorPolicy,
    ) -> dict[str, Any]:
        return _apply_policy_to_raw(
            record,
            prediction,
            policy,
            self._raw[image_index],
            lambda score, nms_iou: self._nms_for_score(
                image_index, score, nms_iou
            ),
        )

    def diagnostics(
        self,
        image_index: int,
        detections: Sequence[IndexedDetection],
        annotations: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        key = (image_index, tuple(item.index for item in detections))
        cached = self._diagnostics.get(key)
        if cached is None:
            cached = detector_image_diagnostics(detections, annotations)
            self._diagnostics[key] = cached
        return cached


def apply_policy(
    record: dict[str, Any], prediction: dict[str, Any], policy: DetectorPolicy
) -> dict[str, Any]:
    policy.validate()
    raw = _raw_detections(prediction)
    return _apply_policy_to_raw(
        record,
        prediction,
        policy,
        raw,
        lambda score, nms_iou: _indexed_nms(
            [item for item in raw if item.detection.score >= score], nms_iou
        ),
    )


def _annotation_box(annotation: dict[str, Any]) -> np.ndarray:
    value = annotation.get("bbox_xywh", annotation.get("bbox"))
    if value is None:
        raise ValueError("annotation is missing bbox_xywh/bbox")
    return _xywh_to_xyxy([float(item) for item in value])


def match_boxes(
    detections: Sequence[IndexedDetection],
    annotations: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[int, tuple[int, float]]:
    """Return a maximum-cardinality, maximum-IoU threshold-valid assignment."""
    if not detections or not annotations:
        return {}
    from scipy.optimize import linear_sum_assignment

    overlaps = np.asarray(
        [
            [
                _iou(
                    np.asarray(
                        [
                            item.detection.x1,
                            item.detection.y1,
                            item.detection.x2,
                            item.detection.y2,
                        ],
                        dtype=np.float32,
                    ),
                    _annotation_box(annotation),
                )
                for annotation in annotations
            ]
            for item in detections
        ],
        dtype=np.float64,
    )
    # One valid edge is more valuable than every possible IoU tie combined.
    reward = (overlaps >= threshold).astype(np.float64) * (
        min(len(detections), len(annotations)) + 1.0
    ) + overlaps
    detection_indices, annotation_indices = linear_sum_assignment(-reward)
    return {
        int(detection_index): (int(annotation_index), float(overlaps[detection_index, annotation_index]))
        for detection_index, annotation_index in zip(
            detection_indices.tolist(), annotation_indices.tolist()
        )
        if overlaps[detection_index, annotation_index] >= threshold
    }


def detector_image_diagnostics(
    detections: Sequence[IndexedDetection], annotations: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    matches_50 = match_boxes(detections, annotations, 0.5)
    matches_75 = match_boxes(detections, annotations, 0.75)
    predicted_count = len(detections)
    ground_truth_count = len(annotations)
    tp = len(matches_50)
    fp = predicted_count - tp
    fn = ground_truth_count - tp
    exact_50 = predicted_count == ground_truth_count == tp
    exact_75 = predicted_count == ground_truth_count == len(matches_75)
    matched_detection_indices = set(matches_50)
    duplicate = 0
    background = 0
    gt_boxes = [_annotation_box(annotation) for annotation in annotations]
    for index, item in enumerate(detections):
        if index in matched_detection_indices:
            continue
        box = np.asarray(
            [
                item.detection.x1,
                item.detection.y1,
                item.detection.x2,
                item.detection.y2,
            ],
            dtype=np.float32,
        )
        maximum = max((_iou(box, gt) for gt in gt_boxes), default=0.0)
        if maximum >= 0.5:
            duplicate += 1
        else:
            background += 1
    localization = sum(overlap < 0.75 for _, overlap in matches_50.values())
    localization_error = sum(
        (1.0 - overlap) / 0.5 for _, overlap in matches_50.values()
    )
    lrp_denominator = tp + fp + fn
    lrp = (
        (localization_error + fp + fn) / lrp_denominator
        if lrp_denominator
        else 0.0
    )
    return {
        "ground_truth_count": ground_truth_count,
        "prediction_count": predicted_count,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "count_correct": predicted_count == ground_truth_count,
        "exact_iou_50": exact_50,
        "exact_iou_75": exact_75,
        "matches_iou_50": matches_50,
        "error_types": {
            "localization": localization,
            "duplicate": duplicate,
            "background_fp": background,
            "missed_gt": fn,
            "count_mismatch": int(predicted_count != ground_truth_count),
        },
        "lrp": lrp,
        "lrp_localization_error": localization_error,
    }


def _expected_class_id(annotation: dict[str, Any]) -> str:
    if annotation.get("class_id") is not None:
        return str(annotation["class_id"])
    return f"bread_{int(annotation['category_id']):02d}"


def evaluate_image(
    record: dict[str, Any],
    prediction: dict[str, Any],
    policy: DetectorPolicy,
    *,
    approval_threshold: float,
    applied: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    applied = applied or apply_policy(record, prediction, policy)
    diagnostics = diagnostics or detector_image_diagnostics(
        applied["detections"], record["annotations"]
    )
    detector_correct = bool(diagnostics["exact_iou_50"])
    error_types = dict(diagnostics["error_types"])
    groups = record.get("groups", {})
    error_types["border_related"] = int(
        not detector_correct and bool(groups.get("border_contact", False))
    )
    error_types["size_related"] = int(
        not detector_correct
        and (
            bool(groups.get("small_object", False))
            or "DETECTOR_OBJECT_TOO_SMALL" in applied["reason_codes"]
        )
    )
    diagnostics = {**diagnostics, "error_types": error_types}
    detector_pass = bool(applied["pass"])
    approved = False
    e2e_correct = False
    unknown_top3_count = 0
    unknown_top3_correct = 0
    classifier_executed = detector_pass
    classifications = prediction.get("classifications", {})
    if detector_pass:
        selected_classifications: list[dict[str, Any]] = []
        for item in applied["detections"]:
            classification = classifications.get(
                str(item.index), classifications.get(item.index)
            )
            if classification is None:
                raise ValueError(
                    f"missing cached classifier output for raw detection {item.index}"
                )
            selected_classifications.append(classification)
        classifier_recapture = any(
            bool(item.get("recapture", False))
            or (
                bool(item.get("touches_border", False))
                and float(item["confidence"]) < approval_threshold
            )
            for item in selected_classifications
        )
        approved = bool(selected_classifications) and not classifier_recapture and all(
            float(item["confidence"]) >= approval_threshold
            for item in selected_classifications
        )
        matches = diagnostics["matches_iou_50"]
        all_classes_correct = len(matches) == len(selected_classifications)
        for selected_index, classification in enumerate(selected_classifications):
            match = matches.get(selected_index)
            if match is None:
                all_classes_correct = False
                continue
            annotation_index, _ = match
            expected = _expected_class_id(record["annotations"][annotation_index])
            predicted = str(classification["top1_class_id"])
            all_classes_correct = all_classes_correct and predicted == expected
            if (
                not classifier_recapture
                and float(classification["confidence"]) < approval_threshold
            ):
                unknown_top3_count += 1
                unknown_top3_correct += int(
                    expected in [str(value) for value in classification["top3_class_ids"]]
                )
        e2e_correct = bool(approved and detector_correct and all_classes_correct)
    ranking_score = min(
        (item.detection.score for item in applied["detections"]), default=0.0
    )
    return {
        "image_id": record.get("image_id"),
        "detector_correct": detector_correct,
        "detector_exact_iou_75": bool(diagnostics["exact_iou_75"]),
        "detector_pass": detector_pass,
        "reason_codes": applied["reason_codes"],
        "classifier_executed": classifier_executed,
        "approved": approved,
        "e2e_correct": e2e_correct,
        "unknown_top3_count": unknown_top3_count,
        "unknown_top3_correct": unknown_top3_correct,
        "ranking_score": ranking_score,
        "group_values": dict(record.get("groups", {})),
        "diagnostics": diagnostics,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _gate_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "useful_reject": 0,
        "wasted_reject": 0,
        "silent_failure": 0,
        "safe_pass": 0,
    }
    for row in rows:
        if row["detector_correct"] and row["detector_pass"]:
            counts["safe_pass"] += 1
        elif row["detector_correct"]:
            counts["wasted_reject"] += 1
        elif row["detector_pass"]:
            counts["silent_failure"] += 1
        else:
            counts["useful_reject"] += 1
    return counts


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    pairs = sorted(
        (float(score), bool(label)) for label, score in zip(labels, scores)
    )
    positive_count = sum(label for _, label in pairs)
    negative_count = len(pairs) - positive_count
    if not positive_count or not negative_count:
        return None
    # Mann-Whitney U, accumulated by equal-score blocks. This is exactly the
    # pairwise definition below (win=1, tie=0.5) without its O(P*N) loop.
    wins = 0.0
    negatives_before = 0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        block_positive = sum(label for _, label in pairs[start:end])
        block_negative = end - start - block_positive
        wins += block_positive * negatives_before
        wins += 0.5 * block_positive * block_negative
        negatives_before += block_negative
        start = end
    return wins / (positive_count * negative_count)


def summarize_rows(
    rows: Sequence[dict[str, Any]], *, confidence_level: float = 0.95
) -> dict[str, Any]:
    counts = _gate_counts(rows)
    detector_pass_count = counts["silent_failure"] + counts["safe_pass"]
    approved_count = sum(bool(row["approved"]) for row in rows)
    approved_errors = sum(
        bool(row["approved"] and not row["e2e_correct"]) for row in rows
    )
    safe_approved = approved_count - approved_errors
    unknown_count = sum(int(row["unknown_top3_count"]) for row in rows)
    unknown_correct = sum(int(row["unknown_top3_correct"]) for row in rows)
    detector_errors = counts["useful_reject"] + counts["silent_failure"]
    diagnostics = [row["diagnostics"] for row in rows]
    gt = sum(int(value["ground_truth_count"]) for value in diagnostics)
    predictions = sum(int(value["prediction_count"]) for value in diagnostics)
    tp = sum(int(value["true_positive"]) for value in diagnostics)
    error_types = dict(
        sum((Counter(value["error_types"]) for value in diagnostics), Counter())
    )
    error_types.setdefault("border_related", 0)
    error_types.setdefault("size_related", 0)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for name, value in row["group_values"].items():
            groups[f"{name}={value}"].append(row)
    group_reports = {}
    for name, values in sorted(groups.items()):
        group_counts = _gate_counts(values)
        group_detector_pass = (
            group_counts["silent_failure"] + group_counts["safe_pass"]
        )
        group_approved = sum(bool(value["approved"]) for value in values)
        group_safe = sum(bool(value["e2e_correct"]) for value in values)
        group_approved_errors = group_approved - group_safe
        group_reports[name] = {
            "sample_count": len(values),
            "gate_table": group_counts,
            "detector_pass_count": group_detector_pass,
            "detector_pass_risk_upper_95": binomial_rate_upper_bound(
                group_counts["silent_failure"],
                group_detector_pass,
                confidence_level,
            ),
            "detector_coverage": _safe_rate(
                group_detector_pass,
                len(values),
            ),
            "approved_count": group_approved,
            "approved_error_count": group_approved_errors,
            "e2e_approved_risk_upper_95": binomial_rate_upper_bound(
                group_approved_errors,
                group_approved,
                confidence_level,
            ),
            "approval_coverage": _safe_rate(group_approved, len(values)),
            "safe_auto_pass_rate": _safe_rate(group_safe, len(values)),
        }
    detector_risk_upper = binomial_rate_upper_bound(
        counts["silent_failure"], detector_pass_count, confidence_level
    )
    e2e_risk_upper = binomial_rate_upper_bound(
        approved_errors, approved_count, confidence_level
    )
    return {
        "sample_count": len(rows),
        "gate_table": counts,
        "detector_pass_count": detector_pass_count,
        "detector_pass_risk": _safe_rate(
            counts["silent_failure"], detector_pass_count
        ),
        "detector_pass_risk_upper_95": detector_risk_upper,
        "detector_coverage": _safe_rate(detector_pass_count, len(rows)),
        "error_catch_recall": _safe_rate(counts["useful_reject"], detector_errors),
        "rejection_precision": _safe_rate(
            counts["useful_reject"],
            counts["useful_reject"] + counts["wasted_reject"],
        ),
        "false_recapture_rate": _safe_rate(
            counts["wasted_reject"],
            counts["wasted_reject"] + counts["safe_pass"],
        ),
        "approved_count": approved_count,
        "approved_error_count": approved_errors,
        "e2e_approved_risk": _safe_rate(approved_errors, approved_count),
        "e2e_approved_risk_upper_95": e2e_risk_upper,
        "approval_coverage": _safe_rate(approved_count, len(rows)),
        "safe_auto_pass_count": safe_approved,
        "safe_auto_pass_rate": _safe_rate(safe_approved, len(rows)),
        "unknown_top3_count": unknown_count,
        "unknown_top3_accuracy": _safe_rate(unknown_correct, unknown_count),
        "risk_certification": {
            "detector_pass_feasible": detector_risk_upper <= 0.005,
            "e2e_approved_feasible": e2e_risk_upper <= 0.005,
        },
        "object_diagnostics": {
            "ground_truth_count": gt,
            "prediction_count": predictions,
            "true_positive": tp,
            "false_positive": predictions - tp,
            "false_negative": gt - tp,
            "precision": _safe_rate(tp, predictions),
            "recall": _safe_rate(tp, gt),
            "exact_count_accuracy": _safe_rate(
                sum(bool(value["count_correct"]) for value in diagnostics), len(rows)
            ),
            "exact_image_iou_50": _safe_rate(
                sum(bool(value["exact_iou_50"]) for value in diagnostics), len(rows)
            ),
            "exact_image_iou_75": _safe_rate(
                sum(bool(value["exact_iou_75"]) for value in diagnostics), len(rows)
            ),
            "lrp": (
                (
                    sum(float(value["lrp_localization_error"]) for value in diagnostics)
                    + (predictions - tp)
                    + (gt - tp)
                )
                / (tp + (predictions - tp) + (gt - tp))
                if tp + (predictions - tp) + (gt - tp)
                else 0.0
            ),
            "error_types": error_types,
        },
        "failure_auroc": _auc(
            [not bool(row["detector_correct"]) for row in rows],
            [-float(row["ranking_score"]) for row in rows],
        ),
        "groups": group_reports,
    }


def curve_metrics(points: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"risk_coverage": [], "aurc": None, "augrc": None}
    ordered = sorted(
        points,
        key=lambda value: (
            float(value["detector_coverage"] or 0.0),
            float(value["detector_pass_risk"] or 0.0),
        ),
    )
    coverage = np.asarray(
        [float(value["detector_coverage"] or 0.0) for value in ordered]
    )
    selective_risk = np.asarray(
        [float(value["detector_pass_risk"] or 0.0) for value in ordered]
    )
    generalized_risk = np.asarray(
        [
            int(value["gate_table"]["silent_failure"])
            / max(1, int(value["sample_count"]))
            for value in ordered
        ]
    )
    return {
        "risk_coverage": [
            {
                "score_threshold": value.get("score_threshold"),
                "coverage": float(x),
                "risk": float(risk),
                "generalized_risk": float(generalized),
            }
            for value, x, risk, generalized in zip(
                ordered, coverage, selective_risk, generalized_risk
            )
        ],
        "aurc": float(np.trapezoid(selective_risk, coverage))
        if len(ordered) > 1
        else 0.0,
        "augrc": float(np.trapezoid(generalized_risk, coverage))
        if len(ordered) > 1
        else 0.0,
    }


def evaluate_policy(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    policy: DetectorPolicy,
    *,
    approval_threshold: float,
    cache: PolicyEvaluationCache | None = None,
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    if cache is not None and len(cache) != len(records):
        raise ValueError("policy evaluation cache length does not match records")
    rows = []
    for image_index, (record, prediction) in enumerate(zip(records, predictions)):
        applied = (
            cache.apply(image_index, record, prediction, policy)
            if cache is not None
            else None
        )
        diagnostics = (
            cache.diagnostics(
                image_index, applied["detections"], record["annotations"]
            )
            if cache is not None
            else None
        )
        rows.append(
            evaluate_image(
                record,
                prediction,
                policy,
                approval_threshold=approval_threshold,
                applied=applied,
                diagnostics=diagnostics,
            )
        )
    summary = summarize_rows(rows)
    return {
        "schema_version": "1.0",
        "model_version": TARGET_MODE_VERSION,
        "policy": asdict(policy),
        "policy_key": policy.key,
        "metrics": summary,
        "rows": rows,
    }


def select_candidate(
    candidates: Sequence[dict[str, Any]],
    *,
    maximum_risk_upper: float = 0.005,
    minimum_error_catch_recall: float = 0.99,
    minimum_group_sample_count: int = 30,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("detector target selection requires candidates")

    def worst_group(candidate: dict[str, Any]) -> float:
        groups = candidate["natural"]["metrics"].get("groups", {})
        eligible = [
            float(value["approval_coverage"] or 0.0)
            for value in groups.values()
            if int(value["sample_count"]) >= minimum_group_sample_count
        ]
        return min(eligible) if eligible else 0.0

    eligible = []
    for candidate in candidates:
        natural = candidate["natural"]["metrics"]
        hard = candidate["hard"]["metrics"]
        error_catch = hard.get("error_catch_recall")
        if (
            float(natural["detector_pass_risk_upper_95"]) <= maximum_risk_upper
            and float(natural["e2e_approved_risk_upper_95"]) <= maximum_risk_upper
            and error_catch is not None
            and float(error_catch) >= minimum_error_catch_recall
        ):
            eligible.append(candidate)
    if eligible:
        selected = min(
            eligible,
            key=lambda value: (
                -float(value["natural"]["metrics"]["safe_auto_pass_rate"] or 0.0),
                -worst_group(value),
                float(value.get("augrc", 1.0)),
                int(value.get("seed", 0)),
                str(value.get("model_id", "")),
                value["natural"]["policy_key"],
            ),
        )
        status = "locked_candidate"
    else:
        selected = min(
            candidates,
            key=lambda value: (
                int(value["natural"]["metrics"]["gate_table"]["silent_failure"]),
                float(value["natural"]["metrics"]["detector_pass_risk_upper_95"]),
                float(value["natural"]["metrics"]["e2e_approved_risk_upper_95"]),
                -float(value["natural"]["metrics"]["safe_auto_pass_rate"] or 0.0),
                int(value.get("seed", 0)),
                str(value.get("model_id", "")),
                value["natural"]["policy_key"],
            ),
        )
        status = "experiment_only"
    return {
        "schema_version": "1.0",
        "selection_rule": (
            "risk_u95_constraints_then_safe_auto_pass_then_worst_group_coverage_"
            "then_augrc"
        ),
        "promotion_status": status,
        "eligible_candidate_count": len(eligible),
        "candidate_count": len(candidates),
        "selected": selected,
    }


def assert_no_split_leakage(records: Iterable[dict[str, Any]]) -> None:
    fields = (
        "image_sha256",
        "perceptual_group_id",
        "capture_session_id",
        "physical_target_group_id",
    )
    seen: dict[tuple[str, str], str] = {}
    folds: dict[tuple[str, str], int] = {}
    for record in records:
        split = str(record["split"])
        for field in fields:
            value = record.get(field)
            if value in (None, ""):
                continue
            key = (field, str(value))
            previous = seen.setdefault(key, split)
            if previous != split:
                raise ValueError(
                    f"split leakage for {field}={value}: {previous} vs {split}"
                )
            if split == "development" and record.get("fold") is not None:
                fold = int(record["fold"])
                previous_fold = folds.setdefault(key, fold)
                if previous_fold != fold:
                    raise ValueError(
                        f"fold leakage for {field}={value}: {previous_fold} vs {fold}"
                    )


def sha256_paths(paths: dict[str, Path]) -> dict[str, str]:
    values = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        values[name] = digest.hexdigest()
    return values
