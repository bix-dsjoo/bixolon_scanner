from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..pipeline.ports import Detection
from ..runtime.onnx import nms

_GEOMETRY_SCORE_THRESHOLDS = (0.05, 0.1, 0.145, 0.25, 0.485, 0.65)


def _sharpness(rgb: np.ndarray) -> float:
    gray = rgb.astype(np.float32).mean(axis=2)
    if min(gray.shape) < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def _allowed_box(box: list[float], maximum_aspect_ratio: float) -> bool:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width > 0 and height > 0 and max(width / height, height / width) <= maximum_aspect_ratio


def _selected_detections(
    prediction: dict[str, Any],
    score_threshold: float,
    *,
    nms_iou_threshold: float,
    maximum_aspect_ratio: float,
) -> list[Detection]:
    candidates = [
        Detection(*box, float(score))
        for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
        if score >= score_threshold and _allowed_box(box, maximum_aspect_ratio)
    ]
    return nms(candidates, nms_iou_threshold)


def _box_iou(left: Detection, right: Detection) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _box_geometry_features(
    detections: list[Detection],
    *,
    width: int,
    height: int,
    prefix: str,
) -> dict[str, float]:
    """Summarize label-free scene geometry for a post-NMS detection set."""
    image_area = float(width * height)
    if not detections:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_area_sum_ratio": 0.0,
            f"{prefix}_area_min_ratio": 0.0,
            f"{prefix}_area_median_ratio": 0.0,
            f"{prefix}_area_max_ratio": 0.0,
            f"{prefix}_border_fraction": 0.0,
            f"{prefix}_maximum_pair_iou": 0.0,
            f"{prefix}_maximum_pair_ioa": 0.0,
            f"{prefix}_minimum_normalized_center_distance": 10.0,
            f"{prefix}_crowding_index": 0.0,
        }

    areas = np.asarray(
        [(item.x2 - item.x1) * (item.y2 - item.y1) for item in detections],
        dtype=np.float64,
    )
    border_margin = 0.01 * min(width, height)
    border_count = sum(
        item.x1 <= border_margin
        or item.y1 <= border_margin
        or item.x2 >= width - border_margin
        or item.y2 >= height - border_margin
        for item in detections
    )
    pair_ious: list[float] = []
    pair_ioas: list[float] = []
    normalized_distances: list[float] = []
    for index, left in enumerate(detections):
        left_area = areas[index]
        left_center = ((left.x1 + left.x2) / 2.0, (left.y1 + left.y2) / 2.0)
        for right_index in range(index + 1, len(detections)):
            right = detections[right_index]
            right_area = areas[right_index]
            intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
            intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
            intersection = intersection_width * intersection_height
            pair_ious.append(_box_iou(left, right))
            pair_ioas.append(intersection / min(left_area, right_area))
            right_center = ((right.x1 + right.x2) / 2.0, (right.y1 + right.y2) / 2.0)
            center_distance = math.hypot(
                left_center[0] - right_center[0], left_center[1] - right_center[1]
            )
            normalized_distances.append(
                center_distance / max(math.sqrt((left_area + right_area) / 2.0), 1e-9)
            )

    maximum_iou = max(pair_ious, default=0.0)
    maximum_ioa = max(pair_ioas, default=0.0)
    minimum_center_distance = min(normalized_distances, default=10.0)
    crowding_index = max(maximum_ioa, 1.0 / (1.0 + minimum_center_distance))
    return {
        f"{prefix}_count": float(len(detections)),
        f"{prefix}_area_sum_ratio": float(areas.sum() / image_area),
        f"{prefix}_area_min_ratio": float(areas.min() / image_area),
        f"{prefix}_area_median_ratio": float(np.median(areas) / image_area),
        f"{prefix}_area_max_ratio": float(areas.max() / image_area),
        f"{prefix}_border_fraction": border_count / len(detections),
        f"{prefix}_maximum_pair_iou": maximum_iou,
        f"{prefix}_maximum_pair_ioa": maximum_ioa,
        f"{prefix}_minimum_normalized_center_distance": minimum_center_distance,
        f"{prefix}_crowding_index": crowding_index,
    }


def _query_cluster_features(
    prediction: dict[str, Any],
    *,
    score_threshold: float,
    maximum_aspect_ratio: float,
) -> dict[str, float]:
    """Measure duplicate transformer queries before NMS without using labels."""
    candidates = [
        Detection(*box, float(score))
        for box, score in zip(prediction["boxes_xyxy"], prediction["scores"])
        if score >= score_threshold and _allowed_box(box, maximum_aspect_ratio)
    ]
    anchors = nms(candidates, 0.7)
    prefix = f"query_{str(score_threshold).replace('.', '_')}"
    if not anchors:
        return {
            f"{prefix}_maximum_cluster_size": 0.0,
            f"{prefix}_mean_cluster_size": 0.0,
            f"{prefix}_duplicate_fraction": 0.0,
        }
    cluster_sizes = [
        sum(_box_iou(anchor, item) >= 0.7 for item in candidates) for anchor in anchors
    ]
    duplicate_count = sum(max(size - 1, 0) for size in cluster_sizes)
    return {
        f"{prefix}_maximum_cluster_size": float(max(cluster_sizes)),
        f"{prefix}_mean_cluster_size": float(np.mean(cluster_sizes)),
        f"{prefix}_duplicate_fraction": duplicate_count / max(len(candidates), 1),
    }


def detection_geometry_features(
    prediction: dict[str, Any],
    *,
    width: int,
    height: int,
    nms_iou_threshold: float,
    maximum_aspect_ratio: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for threshold in _GEOMETRY_SCORE_THRESHOLDS:
        prefix = f"nms_{str(threshold).replace('.', '_')}"
        selected = _selected_detections(
            prediction,
            threshold,
            nms_iou_threshold=nms_iou_threshold,
            maximum_aspect_ratio=maximum_aspect_ratio,
        )
        features.update(_box_geometry_features(selected, width=width, height=height, prefix=prefix))
        if threshold in (0.05, 0.145, 0.485):
            features.update(
                _query_cluster_features(
                    prediction,
                    score_threshold=threshold,
                    maximum_aspect_ratio=maximum_aspect_ratio,
                )
            )
    return features


def _counts(flagged: list[bool], rows: list[dict[str, Any]]) -> dict[str, Any]:
    recapture_total = sum(row["is_recapture"] for row in rows)
    normal_total = len(rows) - recapture_total
    true_positive = int(sum(bool(flag) and row["is_recapture"] for flag, row in zip(flagged, rows)))
    false_positive = int(
        sum(bool(flag) and not row["is_recapture"] for flag, row in zip(flagged, rows))
    )
    expected_by_reason = Counter(
        reason for row in rows if row["is_recapture"] for reason in row["reason_codes"]
    )
    caught_by_reason = Counter(
        reason
        for flag, row in zip(flagged, rows)
        if flag and row["is_recapture"]
        for reason in row["reason_codes"]
    )
    return {
        "recapture_sample_count": recapture_total,
        "normal_sample_count": normal_total,
        "true_recapture_count": true_positive,
        "recapture_recall": true_positive / recapture_total if recapture_total else None,
        "false_recapture_count": false_positive,
        "false_recapture_rate": false_positive / normal_total if normal_total else None,
        "caught_by_reason": {
            reason: {
                "caught": caught_by_reason[reason],
                "total": total,
                "recall": caught_by_reason[reason] / total,
            }
            for reason, total in sorted(expected_by_reason.items())
        },
    }


def best_single_feature_rules(
    rows: list[dict[str, Any]], *, maximum_false_recapture_rate: float
) -> list[dict[str, Any]]:
    normal_count = sum(not row["is_recapture"] for row in rows)
    maximum_false_count = math.floor(normal_count * maximum_false_recapture_rate + 1e-12)
    feature_names = sorted(rows[0]["features"]) if rows else []
    results: list[dict[str, Any]] = []
    for feature_name in feature_names:
        values = sorted({float(row["features"][feature_name]) for row in rows})
        candidates: list[dict[str, Any]] = []
        for direction in ("at_or_below", "at_or_above"):
            for threshold in values:
                if direction == "at_or_below":
                    flagged = [row["features"][feature_name] <= threshold for row in rows]
                else:
                    flagged = [row["features"][feature_name] >= threshold for row in rows]
                metrics = _counts(flagged, rows)
                if metrics["false_recapture_count"] <= maximum_false_count:
                    candidates.append(
                        {
                            "feature": feature_name,
                            "direction": direction,
                            "threshold": threshold,
                            **metrics,
                        }
                    )
        if candidates:
            results.append(
                max(
                    candidates,
                    key=lambda row: (
                        row["true_recapture_count"],
                        -row["false_recapture_count"],
                    ),
                )
            )
    return sorted(
        results,
        key=lambda row: (row["true_recapture_count"], -row["false_recapture_count"]),
        reverse=True,
    )


def best_monotonic_or_policy(
    rows: list[dict[str, Any]],
    *,
    maximum_false_recapture_rate: float,
    maximum_rule_count: int = 6,
) -> dict[str, Any]:
    """Greedily combine explainable threshold rules under a false-recapture cap."""
    normal_count = sum(not row["is_recapture"] for row in rows)
    maximum_false_count = math.floor(normal_count * maximum_false_recapture_rate + 1e-12)
    candidates: list[tuple[dict[str, Any], list[bool]]] = []
    feature_names = sorted(rows[0]["features"]) if rows else []
    for feature_name in feature_names:
        values = sorted({float(row["features"][feature_name]) for row in rows})
        for direction in ("at_or_below", "at_or_above"):
            for threshold in values:
                flagged = [
                    row["features"][feature_name] <= threshold
                    if direction == "at_or_below"
                    else row["features"][feature_name] >= threshold
                    for row in rows
                ]
                metrics = _counts(flagged, rows)
                if metrics["false_recapture_count"] <= maximum_false_count:
                    candidates.append(
                        (
                            {
                                "feature": feature_name,
                                "direction": direction,
                                "threshold": threshold,
                            },
                            flagged,
                        )
                    )

    selected_rules: list[dict[str, Any]] = []
    combined = [False] * len(rows)
    while len(selected_rules) < maximum_rule_count:
        best: tuple[dict[str, Any], list[bool], dict[str, Any]] | None = None
        for rule, flagged in candidates:
            proposed = [left or right for left, right in zip(combined, flagged)]
            metrics = _counts(proposed, rows)
            if metrics["false_recapture_count"] > maximum_false_count:
                continue
            if metrics["true_recapture_count"] <= _counts(combined, rows)["true_recapture_count"]:
                continue
            if best is None or (
                metrics["true_recapture_count"],
                -metrics["false_recapture_count"],
            ) > (best[2]["true_recapture_count"], -best[2]["false_recapture_count"]):
                best = (rule, proposed, metrics)
        if best is None:
            break
        selected_rules.append(best[0])
        combined = best[1]
        candidates = [item for item in candidates if item[0] != best[0]]
    return {
        "policy": "any_monotonic_threshold_rule",
        "optimistic_upper_bound_only": True,
        "rules": selected_rules,
        **_counts(combined, rows),
    }


def _matches_rule(features: dict[str, float], rule: dict[str, Any]) -> bool:
    value = features[rule["feature"]]
    if rule["direction"] == "at_or_below":
        return value <= rule["threshold"]
    return value >= rule["threshold"]


def normal_envelope_outlier_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an intentionally optimistic, label-inspected zero-false-recapture bound."""
    feature_names = sorted(rows[0]["features"]) if rows else []
    normal_rows = [row for row in rows if not row["is_recapture"]]
    bounds = {
        feature_name: {
            "minimum": min(float(row["features"][feature_name]) for row in normal_rows),
            "maximum": max(float(row["features"][feature_name]) for row in normal_rows),
        }
        for feature_name in feature_names
    }
    flagged = [
        any(
            row["features"][feature_name] < bounds[feature_name]["minimum"]
            or row["features"][feature_name] > bounds[feature_name]["maximum"]
            for feature_name in feature_names
        )
        for row in rows
    ]
    return {
        "policy": "outside_any_normal_feature_range",
        "optimistic_upper_bound_only": True,
        "normal_feature_bounds": bounds,
        **_counts(flagged, rows),
    }


def best_detector_policy(
    rows: list[dict[str, Any]],
    selected_by_threshold: dict[float, list[list[Detection]]],
    *,
    main_thresholds: list[float],
    shadow_thresholds: list[float],
    minimum_area_ratios: list[float],
    match_iou_thresholds: list[float],
    maximum_false_recapture_rate: float,
) -> dict[str, Any] | None:
    normal_count = sum(not row["is_recapture"] for row in rows)
    maximum_false_count = math.floor(normal_count * maximum_false_recapture_rate + 1e-12)
    best: dict[str, Any] | None = None
    for main_threshold in main_thresholds:
        accepted_by_image = selected_by_threshold[main_threshold]
        for shadow_threshold in shadow_thresholds:
            if shadow_threshold >= main_threshold:
                continue
            shadow_by_image = selected_by_threshold[shadow_threshold]
            for minimum_area_ratio in minimum_area_ratios:
                for match_iou_threshold in match_iou_thresholds:
                    flagged: list[bool] = []
                    for row, accepted, shadow in zip(rows, accepted_by_image, shadow_by_image):
                        uncertain = False
                        for candidate in shadow:
                            if candidate.score >= main_threshold:
                                continue
                            area_ratio = (
                                (candidate.x2 - candidate.x1)
                                * (candidate.y2 - candidate.y1)
                                / row["image_area"]
                            )
                            if area_ratio < minimum_area_ratio:
                                continue
                            if (
                                not accepted
                                or max(_box_iou(candidate, item) for item in accepted)
                                < match_iou_threshold
                            ):
                                uncertain = True
                                break
                        flagged.append(not accepted or uncertain)
                    metrics = _counts(flagged, rows)
                    if metrics["false_recapture_count"] > maximum_false_count:
                        continue
                    candidate = {
                        "main_score_threshold": main_threshold,
                        "uncertainty_score_threshold": shadow_threshold,
                        "uncertainty_min_area_ratio": minimum_area_ratio,
                        "uncertainty_match_iou_threshold": match_iou_threshold,
                        **metrics,
                    }
                    if best is None or (
                        candidate["true_recapture_count"],
                        -candidate["false_recapture_count"],
                    ) > (best["true_recapture_count"], -best["false_recapture_count"]):
                        best = candidate
    return best


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.dataset_root.resolve() / "annotations" / args.annotation
    payload = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    predictions = {
        int(row["image_id"]): row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    rows: list[dict[str, Any]] = []
    for image_row in sorted(payload["images"], key=lambda row: int(row["id"])):
        image_id = int(image_row["id"])
        prediction = predictions[image_id]
        image_path = (annotation_path.parent / image_row["file_name"]).resolve()
        image_path.relative_to(args.dataset_root.resolve())
        with Image.open(image_path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        scores = sorted((float(value) for value in prediction["scores"]), reverse=True)
        geometry_features = detection_geometry_features(
            prediction,
            width=width,
            height=height,
            nms_iou_threshold=args.nms_threshold,
            maximum_aspect_ratio=args.maximum_aspect_ratio,
        )
        rows.append(
            {
                "image_id": image_id,
                "is_recapture": image_row.get("status") == "RECAPTURE",
                "reason_codes": list(image_row.get("reason_codes", [])),
                "image_area": float(width * height),
                "features": {
                    "mean_luminance": float(rgb.mean()),
                    "luminance_std": float(rgb.std()),
                    "sharpness": _sharpness(rgb),
                    "maximum_detector_score": scores[0] if scores else 0.0,
                    "second_detector_score": scores[1] if len(scores) > 1 else 0.0,
                    "detector_score_gap": scores[0] - scores[1] if len(scores) > 1 else 0.0,
                    "raw_count_at_0_145": sum(value >= 0.145 for value in scores),
                    "raw_count_at_0_485": sum(value >= 0.485 for value in scores),
                    **geometry_features,
                },
            }
        )

    main_thresholds = [0.145, 0.25, 0.35, 0.485, 0.55, 0.65, 0.75, 0.85, 0.9]
    shadow_thresholds = [0.05, 0.1, 0.145, 0.2, 0.25, 0.3, 0.35, 0.4]
    all_thresholds = sorted(set(main_thresholds + shadow_thresholds))
    selected_by_threshold = {
        threshold: [
            _selected_detections(
                predictions[row["image_id"]],
                threshold,
                nms_iou_threshold=args.nms_threshold,
                maximum_aspect_ratio=args.maximum_aspect_ratio,
            )
            for row in rows
        ]
        for threshold in all_thresholds
    }
    single_feature_rules = best_single_feature_rules(
        rows, maximum_false_recapture_rate=args.maximum_false_recapture_rate
    )
    monotonic_or_policy = best_monotonic_or_policy(
        rows,
        maximum_false_recapture_rate=args.maximum_false_recapture_rate,
    )
    normal_envelope = normal_envelope_outlier_diagnostic(rows)
    report = {
        "evaluation": "recapture_separability_diagnostic",
        "selection_use_prohibited": True,
        "annotation": args.annotation,
        "prediction_source": args.predictions.name,
        "maximum_false_recapture_rate": args.maximum_false_recapture_rate,
        "maximum_false_recapture_count": math.floor(
            sum(not row["is_recapture"] for row in rows) * args.maximum_false_recapture_rate + 1e-12
        ),
        "single_feature_rules": single_feature_rules,
        "best_monotonic_or_policy": monotonic_or_policy,
        "normal_envelope_outlier_diagnostic": normal_envelope,
        "recapture_case_diagnostics": [
            {
                "image_id": row["image_id"],
                "reason_codes": row["reason_codes"],
                "caught_by_monotonic_or_policy": any(
                    _matches_rule(row["features"], rule) for rule in monotonic_or_policy["rules"]
                ),
                "normal_envelope_outlier_features": [
                    feature_name
                    for feature_name, bounds in normal_envelope["normal_feature_bounds"].items()
                    if row["features"][feature_name] < bounds["minimum"]
                    or row["features"][feature_name] > bounds["maximum"]
                ],
            }
            for row in rows
            if row["is_recapture"]
        ],
        "best_detector_policy_grid": best_detector_policy(
            rows,
            selected_by_threshold,
            main_thresholds=main_thresholds,
            shadow_thresholds=shadow_thresholds,
            minimum_area_ratios=[0.0, 0.0025, 0.005, 0.01],
            match_iou_thresholds=[0.3, 0.5, 0.7],
            maximum_false_recapture_rate=args.maximum_false_recapture_rate,
        ),
        "limitations": [
            "The development evaluation labels were inspected, so these rules cannot be promoted.",
            "The diagnostic estimates separability only and is not an independent test result.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose recapture separability without promotion"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-false-recapture-rate", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--maximum-aspect-ratio", type=float, default=5.0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
