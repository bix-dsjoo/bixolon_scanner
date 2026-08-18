from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier

from ...evaluation.detector import _metrics, _xywh_to_xyxy, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import box_iou, nms
from ...training.data import read_manifest


def _intersection_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    return np.prod(np.maximum(0.0, bottom_right - top_left), axis=2)


def _iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = _intersection_matrix(left, right)
    left_area = np.prod(np.maximum(0.0, left[:, 2:] - left[:, :2]), axis=1)
    right_area = np.prod(np.maximum(0.0, right[:, 2:] - right[:, :2]), axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0.0,
    )


def proposal_intersection_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Public experiment helper for pairwise proposal intersection areas."""
    return _intersection_matrix(left, right)


def proposal_iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Public experiment helper for pairwise proposal IoU."""
    return _iou_matrix(left, right)


def proposal_features(
    record: dict[str, Any],
    primary: dict[str, Any],
    recovery: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return boxes, original scores/classes, and GT-independent ranker features."""
    rows = (primary, recovery)
    boxes = np.asarray([box for row in rows for box in row["boxes_xyxy"]], dtype=np.float32)
    scores = np.asarray([score for row in rows for score in row["scores"]], dtype=np.float32)
    class_ids = np.asarray(
        [class_id for row in rows for class_id in row["class_ids"]], dtype=np.int64
    )
    top3 = np.asarray(
        [classes for row in rows for classes in row["top3_class_ids"]], dtype=np.int64
    )
    source = np.concatenate(
        [
            np.full(len(primary["scores"]), 0, dtype=np.int64),
            np.full(len(recovery["scores"]), 1, dtype=np.int64),
        ]
    )
    width = float(record["width"])
    height = float(record["height"])
    normalized = boxes / np.asarray([width, height, width, height], dtype=np.float32)
    sizes = np.maximum(0.0, normalized[:, 2:] - normalized[:, :2])
    centers = (normalized[:, :2] + normalized[:, 2:]) * 0.5
    areas = sizes[:, 0] * sizes[:, 1]
    aspects = np.divide(
        sizes[:, 0],
        sizes[:, 1],
        out=np.zeros(len(sizes), dtype=np.float32),
        where=sizes[:, 1] > 0.0,
    )
    edge_distance = np.min(
        np.column_stack(
            (normalized[:, 0], normalized[:, 1], 1.0 - normalized[:, 2], 1.0 - normalized[:, 3])
        ),
        axis=1,
    )

    overlap = _iou_matrix(boxes, boxes)
    np.fill_diagonal(overlap, 0.0)
    absolute_areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    intersection = _intersection_matrix(boxes, boxes)
    np.fill_diagonal(intersection, 0.0)
    contains_other = np.divide(
        intersection,
        absolute_areas[None, :],
        out=np.zeros_like(intersection, dtype=np.float32),
        where=absolute_areas[None, :] > 0.0,
    )
    inside_other = np.divide(
        intersection,
        absolute_areas[:, None],
        out=np.zeros_like(intersection, dtype=np.float32),
        where=absolute_areas[:, None] > 0.0,
    )
    same_source = source[:, None] == source[None, :]
    cross_source = ~same_source
    higher_score = scores[None, :] > scores[:, None]
    within_overlap = np.where(same_source, overlap, 0.0)
    cross_overlap = np.where(cross_source, overlap, 0.0)
    cross_best_index = np.argmax(cross_overlap, axis=1)
    cross_best_iou = cross_overlap[np.arange(len(boxes)), cross_best_index]
    cross_best_score = scores[cross_best_index]
    cross_class_match = (class_ids == class_ids[cross_best_index]).astype(np.float32)
    cross_top3_overlap = np.asarray(
        [
            len(set(classes) & set(top3[index])) / 3.0
            for classes, index in zip(top3, cross_best_index)
        ],
        dtype=np.float32,
    )

    one_hot = np.zeros((len(boxes), 20), dtype=np.float32)
    valid_classes = (class_ids >= 0) & (class_ids < 20)
    one_hot[np.arange(len(boxes))[valid_classes], class_ids[valid_classes]] = 1.0
    top3_hot = np.zeros((len(boxes), 20), dtype=np.float32)
    for rank in range(min(3, top3.shape[1])):
        values = top3[:, rank]
        valid = (values >= 0) & (values < 20)
        top3_hot[np.arange(len(boxes))[valid], values[valid]] = 1.0

    scalar_features = np.column_stack(
        (
            scores,
            np.log(np.clip(scores, 1e-6, 1.0)),
            source,
            normalized,
            centers,
            sizes,
            areas,
            aspects,
            edge_distance,
            within_overlap.max(axis=1),
            np.where(higher_score & same_source, overlap, 0.0).max(axis=1),
            (within_overlap >= 0.3).sum(axis=1),
            (within_overlap >= 0.5).sum(axis=1),
            (within_overlap >= 0.7).sum(axis=1),
            cross_best_iou,
            cross_best_score,
            cross_class_match,
            cross_top3_overlap,
            (cross_overlap >= 0.3).sum(axis=1),
            (cross_overlap >= 0.5).sum(axis=1),
            (cross_overlap >= 0.7).sum(axis=1),
            contains_other.max(axis=1),
            (contains_other >= 0.8).sum(axis=1),
            (contains_other >= 0.9).sum(axis=1),
            ((contains_other >= 0.8) & higher_score).sum(axis=1),
            inside_other.max(axis=1),
            (inside_other >= 0.8).sum(axis=1),
            (inside_other >= 0.9).sum(axis=1),
            ((inside_other >= 0.8) & higher_score).sum(axis=1),
        )
    ).astype(np.float32)
    return boxes, scores, class_ids, np.column_stack((scalar_features, one_hot, top3_hot))


def proposal_labels(record: dict[str, Any], boxes: np.ndarray) -> np.ndarray:
    return (proposal_qualities(record, boxes) >= 0.5).astype(np.int64)


def proposal_qualities(record: dict[str, Any], boxes: np.ndarray) -> np.ndarray:
    """Return each proposal's best IoU without assigning one proposal per target."""
    targets = np.asarray(
        [_xywh_to_xyxy(row["bbox_xywh"]) for row in record["annotations"]],
        dtype=np.float32,
    )
    if not len(targets):
        return np.zeros(len(boxes), dtype=np.float32)
    return _iou_matrix(boxes, targets).max(axis=1).astype(np.float32)


def proposal_assignment_labels(
    record: dict[str, Any],
    boxes: np.ndarray,
    *,
    minimum_iou: float = 0.5,
) -> np.ndarray:
    """Assign at most one proposal to each GT with maximum total IoU."""
    labels = np.zeros(len(boxes), dtype=np.int64)
    targets = np.asarray(
        [_xywh_to_xyxy(row["bbox_xywh"]) for row in record["annotations"]],
        dtype=np.float32,
    )
    if not len(boxes) or not len(targets):
        return labels
    ious = _iou_matrix(boxes, targets)
    proposal_indices, target_indices = linear_sum_assignment(-ious)
    assigned_ious = ious[proposal_indices, target_indices]
    labels[proposal_indices[assigned_ious >= minimum_iou]] = 1
    return labels


def select_ranked_predictions(
    ranked: list[dict[str, Any]],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    nms_mode: str = "class_agnostic",
    nms_center_distance_threshold: float = 0.3,
    containment_threshold: float = 0.8,
    group_minimum: int = 0,
) -> list[dict[str, Any]]:
    outputs = []
    for row in ranked:
        candidates = [
            Detection(*box, float(score), int(class_id))
            for box, score, class_id in zip(row["boxes_xyxy"], row["scores"], row["class_ids"])
            if score >= score_threshold
        ]
        if group_minimum:
            candidates = suppress_group_boxes(
                candidates,
                containment_threshold=containment_threshold,
                group_minimum=group_minimum,
            )
        if nms_mode == "class_agnostic":
            selected = nms(candidates, nms_iou_threshold)
        elif nms_mode == "class_aware":
            selected = []
            for class_id in sorted({item.class_id for item in candidates}):
                selected.extend(
                    nms(
                        [item for item in candidates if item.class_id == class_id],
                        nms_iou_threshold,
                    )
                )
            selected.sort(key=lambda item: item.score, reverse=True)
        elif nms_mode == "center_aware":
            selected = _center_aware_nms(
                candidates,
                iou_threshold=nms_iou_threshold,
                center_distance_threshold=nms_center_distance_threshold,
            )
        else:
            raise ValueError(f"unsupported NMS mode: {nms_mode}")
        outputs.append(
            {
                "image_id": row["image_id"],
                "boxes_xyxy": [[item.x1, item.y1, item.x2, item.y2] for item in selected],
                "scores": [item.score for item in selected],
                "class_ids": [item.class_id for item in selected],
            }
        )
    return outputs


def _center_aware_nms(
    detections: list[Detection],
    *,
    iou_threshold: float,
    center_distance_threshold: float,
) -> list[Detection]:
    if center_distance_threshold < 0.0:
        raise ValueError("center distance threshold must be non-negative")
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        remaining = []
        current_center = np.asarray(
            [(current.x1 + current.x2) * 0.5, (current.y1 + current.y2) * 0.5]
        )
        current_area = max(0.0, current.x2 - current.x1) * max(0.0, current.y2 - current.y1)
        for candidate in ordered:
            candidate_center = np.asarray(
                [(candidate.x1 + candidate.x2) * 0.5, (candidate.y1 + candidate.y2) * 0.5]
            )
            candidate_area = max(0.0, candidate.x2 - candidate.x1) * max(
                0.0, candidate.y2 - candidate.y1
            )
            scale = np.sqrt(max(min(current_area, candidate_area), 1e-12))
            center_distance = float(np.linalg.norm(current_center - candidate_center) / scale)
            duplicate = (
                box_iou(current, candidate) > iou_threshold
                and center_distance <= center_distance_threshold
            )
            if not duplicate:
                remaining.append(candidate)
        ordered = remaining
    return kept


def suppress_group_boxes(
    candidates: list[Detection],
    *,
    containment_threshold: float,
    group_minimum: int,
) -> list[Detection]:
    """Remove a box enclosing multiple spatially distinct smaller proposals."""
    if group_minimum < 2:
        raise ValueError("group_minimum must be zero or at least two")
    if not candidates:
        return []
    boxes = np.asarray(
        [[item.x1, item.y1, item.x2, item.y2] for item in candidates], dtype=np.float32
    )
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    intersection = _intersection_matrix(boxes, boxes)
    kept = []
    for index, candidate in enumerate(candidates):
        inner_indices = [
            other_index
            for other_index in range(len(candidates))
            if other_index != index
            and areas[other_index] < areas[index] * 0.8
            and areas[other_index] > 0.0
            and intersection[index, other_index] / areas[other_index] >= containment_threshold
        ]
        distinct = nms([candidates[item] for item in inner_indices], 0.3)
        if len(distinct) < group_minimum:
            kept.append(candidate)
    return kept


def _read_predictions(specs: list[str]) -> dict[int, dict[str, dict[str, Any]]]:
    outputs: dict[int, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        fold_text, separator, path_text = spec.partition("=")
        if not separator:
            raise ValueError("prediction spec must be FOLD=PATH")
        outputs[int(fold_text)] = {
            str(row["image_id"]): row
            for row in (
                json.loads(line)
                for line in Path(path_text).read_text(encoding="utf-8").splitlines()
                if line
            )
        }
    return outputs


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    folds = set(args.folds)
    records = [
        row
        for row in read_manifest(args.manifest)
        if row["record_type"] == "detection"
        and row["split"] == "development"
        and int(row["fold"]) in folds
        and row.get("expected_image_status") == "ANNOTATED"
    ]
    primary = _read_predictions(args.primary_predictions)
    recovery = _read_predictions(args.recovery_predictions)
    if set(primary) != folds or set(recovery) != folds:
        raise ValueError("prediction specs must cover every selected fold")

    cached: dict[int, list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]] = {}
    for fold in sorted(folds):
        fold_rows = [row for row in records if int(row["fold"]) == fold]
        expected_ids = {str(row["image_id"]) for row in fold_rows}
        if set(primary[fold]) != expected_ids or set(recovery[fold]) != expected_ids:
            raise ValueError(f"fold {fold} prediction coverage differs from manifest")
        cached[fold] = []
        for record in fold_rows:
            image_id = str(record["image_id"])
            boxes, _, class_ids, features = proposal_features(
                record,
                primary[fold][image_id],
                recovery[fold][image_id],
            )
            cached[fold].append((record, boxes, class_ids, features))

    ranked_by_id: dict[str, dict[str, Any]] = {}
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        train_parts = [item for fold in folds - {held_out_fold} for item in cached[fold]]
        train_x = np.concatenate([item[3] for item in train_parts])
        train_y = np.concatenate([proposal_labels(item[0], item[1]) for item in train_parts])
        model = ExtraTreesClassifier(
            n_estimators=args.estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed + held_out_fold,
        )
        model.fit(train_x, train_y)
        positive_count = int(train_y.sum())
        for record, boxes, class_ids, features in cached[held_out_fold]:
            scores = model.predict_proba(features)[:, 1]
            ranked_by_id[str(record["image_id"])] = {
                "image_id": record["image_id"],
                "boxes_xyxy": boxes.tolist(),
                "scores": scores.tolist(),
                "class_ids": class_ids.tolist(),
            }
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_candidate_count": len(train_y),
                "training_positive_count": positive_count,
                "training_positive_rate": positive_count / len(train_y),
            }
        )

    ranked = [ranked_by_id[str(row["image_id"])] for row in records]
    candidates = []
    group_cache: dict[tuple[float, float, int], list[dict[str, Any]]] = {}
    for score_threshold, containment_threshold, group_minimum in product(
        args.score_thresholds,
        args.containment_thresholds,
        args.group_minimums,
    ):
        cache_containment = containment_threshold if group_minimum else 0.0
        cache_key = (score_threshold, cache_containment, group_minimum)
        if cache_key not in group_cache:
            group_cache[cache_key] = select_ranked_predictions(
                ranked,
                score_threshold=score_threshold,
                nms_iou_threshold=1.0,
                containment_threshold=containment_threshold,
                group_minimum=group_minimum,
            )
    for score_threshold, nms_threshold, containment_threshold, group_minimum in product(
        args.score_thresholds,
        args.nms_thresholds,
        args.containment_thresholds,
        args.group_minimums,
    ):
        cache_containment = containment_threshold if group_minimum else 0.0
        grouped_predictions = group_cache[(score_threshold, cache_containment, group_minimum)]
        metrics = _metrics(
            records,
            grouped_predictions,
            score_threshold=0.0,
            nms_iou_threshold=nms_threshold,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        candidates.append(
            {
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                "containment_threshold": containment_threshold,
                "group_minimum": group_minimum,
                "metrics": metrics,
            }
        )
    zero_error = [
        row
        for row in candidates
        if row["metrics"]["false_positive_count"] == 0
        and row["metrics"]["false_negative_count"] == 0
    ]
    selected = max(
        zero_error or candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )
    full_recall = [row for row in candidates if row["metrics"]["false_negative_count"] == 0]
    high_recall_selected = max(
        full_recall or candidates,
        key=lambda row: (
            -row["metrics"]["false_negative_count"],
            -row["metrics"]["false_positive_count"],
            row["metrics"]["exact_image_rate"],
            row["score_threshold"],
        ),
    )
    selected_predictions = select_ranked_predictions(
        ranked,
        score_threshold=selected["score_threshold"],
        nms_iou_threshold=selected["nms_iou_threshold"],
        containment_threshold=selected["containment_threshold"],
        group_minimum=selected["group_minimum"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_proposal_ranker",
        "folds": sorted(folds),
        "model": "ExtraTreesClassifier",
        "estimators": args.estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "fold_diagnostics": fold_diagnostics,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": len(zero_error),
        "full_recall_candidate_count": len(full_recall),
        "selected": selected,
        "high_recall_selected": high_recall_selected,
        "error_images": detection_error_rows(
            records,
            selected_predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in selected_predictions),
            encoding="utf-8",
        )
    if args.high_recall_predictions_output:
        high_recall_predictions = select_ranked_predictions(
            ranked,
            score_threshold=high_recall_selected["score_threshold"],
            nms_iou_threshold=high_recall_selected["nms_iou_threshold"],
            containment_threshold=high_recall_selected["containment_threshold"],
            group_minimum=high_recall_selected["group_minimum"],
        )
        args.high_recall_predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.high_recall_predictions_output.write_text(
            "".join(json.dumps(row) + "\n" for row in high_recall_predictions),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a cross-fitted ranker for bread detector proposals"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--primary-predictions", nargs="+", required=True)
    parser.add_argument("--recovery-predictions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--high-recall-predictions-output", type=Path)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--containment-thresholds", type=float, nargs="+", default=[0.8])
    parser.add_argument("--group-minimums", type=int, nargs="+", default=[0])
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260814)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
