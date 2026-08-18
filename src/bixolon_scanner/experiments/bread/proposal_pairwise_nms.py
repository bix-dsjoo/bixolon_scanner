from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from ...evaluation.detector import _metrics, _xywh_to_xyxy, detection_error_rows
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions, _softmax
from .proposal_ranker import _intersection_matrix, _iou_matrix


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0.0,
    )


def pairwise_features(
    record: dict[str, Any],
    prediction: dict[str, Any],
    raw_embeddings: np.ndarray,
    adapted_embeddings: np.ndarray,
    logits: np.ndarray,
    *,
    minimum_iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    count = len(boxes)
    if not (len(raw_embeddings) == len(adapted_embeddings) == len(logits) == count):
        raise ValueError("pairwise proposal features are not aligned")
    ious = _iou_matrix(boxes, boxes)
    left, right = np.where(np.triu(ious >= minimum_iou, k=1))
    pairs = np.column_stack((left, right)).astype(np.int64)
    if not len(pairs):
        return pairs, np.empty((0, 16), dtype=np.float32)

    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    sizes = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2])
    intersection = _intersection_matrix(boxes[left], boxes[right]).diagonal()
    smaller_area = np.minimum(areas[left], areas[right])
    containment = np.divide(
        intersection,
        smaller_area,
        out=np.zeros_like(intersection),
        where=smaller_area > 0.0,
    )
    area_ratio = np.divide(
        np.minimum(areas[left], areas[right]),
        np.maximum(areas[left], areas[right]),
        out=np.zeros_like(intersection),
        where=np.maximum(areas[left], areas[right]) > 0.0,
    )
    normalized_center_delta = np.abs(centers[left] - centers[right]) / np.asarray(
        [record["width"], record["height"]], dtype=np.float32
    )
    relative_center_distance = np.linalg.norm(centers[left] - centers[right], axis=1) / np.sqrt(
        np.maximum(smaller_area, 1e-12)
    )
    relative_size_delta = np.abs(sizes[left] - sizes[right]) / np.maximum(
        np.maximum(sizes[left], sizes[right]), 1e-12
    )
    probabilities = _softmax(logits)
    class_ids = np.asarray(prediction["class_ids"], dtype=np.int64)
    top1 = np.argmax(probabilities, axis=1)
    probability_l1 = np.abs(probabilities[left] - probabilities[right]).sum(axis=1)
    features = np.column_stack(
        (
            ious[left, right],
            containment,
            area_ratio,
            normalized_center_delta,
            relative_center_distance,
            relative_size_delta,
            class_ids[left] == class_ids[right],
            top1[left] == top1[right],
            class_ids[left] == top1[right],
            class_ids[right] == top1[left],
            _cosine_rows(raw_embeddings[left], raw_embeddings[right]),
            _cosine_rows(adapted_embeddings[left], adapted_embeddings[right]),
            _cosine_rows(logits[left], logits[right]),
            probability_l1,
        )
    ).astype(np.float32)
    return pairs, features


def distinct_object_targets(
    record: dict[str, Any], boxes: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    targets = np.asarray(
        [_xywh_to_xyxy(row["bbox_xywh"]) for row in record["annotations"]],
        dtype=np.float32,
    )
    if not len(targets) or not len(pairs):
        return np.zeros(len(pairs), dtype=np.int64)
    ious = _iou_matrix(boxes, targets)
    assignments = np.argmax(ious, axis=1)
    assignments[ious.max(axis=1) < 0.5] = -1
    left = assignments[pairs[:, 0]]
    right = assignments[pairs[:, 1]]
    return ((left >= 0) & (right >= 0) & (left != right)).astype(np.int64)


def pairwise_select(
    prediction: dict[str, Any],
    distinct_probabilities: dict[tuple[int, int], float],
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    distinct_threshold: float,
) -> dict[str, Any]:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float64)
    class_ids = np.asarray(prediction["class_ids"], dtype=np.int64)
    eligible = np.flatnonzero(scores >= score_threshold)
    ordered = eligible[np.argsort(-scores[eligible], kind="stable")].tolist()
    kept = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        remaining = []
        for candidate in ordered:
            pair = (min(current, candidate), max(current, candidate))
            iou = float(_iou_matrix(boxes[[current]], boxes[[candidate]])[0, 0])
            same_object = (
                iou > nms_iou_threshold
                and distinct_probabilities.get(pair, 0.0) < distinct_threshold
            )
            if not same_object:
                remaining.append(candidate)
        ordered = remaining
    return {
        "image_id": prediction["image_id"],
        "boxes_xyxy": boxes[kept].tolist(),
        "scores": scores[kept].tolist(),
        "class_ids": class_ids[kept].tolist(),
    }


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
    ranked = _load_predictions(args.ranked_predictions, records)
    embedding_cache = np.load(args.embedding_cache)
    raw_embeddings = embedding_cache["raw_embeddings"].astype(np.float32)
    adapted_embeddings = embedding_cache["adapted_embeddings"].astype(np.float32)
    counts = embedding_cache["counts"]
    logits_cache = np.load(args.logits_cache)
    logits = logits_cache["logits"]
    if counts.tolist() != [len(row["scores"]) for row in ranked]:
        raise ValueError("pairwise cache does not match ranked prediction counts")

    cached = []
    offset = 0
    for record, prediction, count in zip(records, ranked, counts):
        end = offset + int(count)
        pairs, features = pairwise_features(
            record,
            prediction,
            raw_embeddings[offset:end],
            adapted_embeddings[offset:end],
            logits[offset:end],
            minimum_iou=min(args.nms_iou_thresholds),
        )
        labels = distinct_object_targets(
            record,
            np.asarray(prediction["boxes_xyxy"], dtype=np.float32),
            pairs,
        )
        cached.append((record, prediction, pairs, features, labels))
        offset = end

    pair_probabilities: list[dict[tuple[int, int], float]] = [dict() for _ in records]
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        training_rows = [
            row for row in cached if int(row[0]["fold"]) != held_out_fold and len(row[2])
        ]
        train_x = np.concatenate([row[3] for row in training_rows])
        train_y = np.concatenate([row[4] for row in training_rows])
        model = ExtraTreesClassifier(
            n_estimators=args.estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed + held_out_fold,
        )
        model.fit(train_x, train_y)
        for index, (record, _, pairs, features, _) in enumerate(cached):
            if int(record["fold"]) != held_out_fold or not len(pairs):
                continue
            probabilities = model.predict_proba(features)[:, 1]
            pair_probabilities[index] = {
                (int(pair[0]), int(pair[1])): float(probability)
                for pair, probability in zip(pairs, probabilities)
            }
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_pair_count": len(train_y),
                "training_distinct_pair_count": int(train_y.sum()),
                "training_distinct_pair_rate": float(train_y.mean()),
            }
        )

    candidates = []
    for score, nms_iou, distinct_threshold in product(
        args.score_thresholds,
        args.nms_iou_thresholds,
        args.distinct_thresholds,
    ):
        predictions = [
            pairwise_select(
                prediction,
                pair_scores,
                score_threshold=score,
                nms_iou_threshold=nms_iou,
                distinct_threshold=distinct_threshold,
            )
            for prediction, pair_scores in zip(ranked, pair_probabilities)
        ]
        metrics = _metrics(
            records,
            predictions,
            score_threshold=0.0,
            nms_iou_threshold=1.0,
            match_iou_threshold=0.5,
            max_queries=600,
        )
        candidates.append(
            {
                "score_threshold": score,
                "nms_iou_threshold": nms_iou,
                "distinct_threshold": distinct_threshold,
                "metrics": metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
            row["score_threshold"],
        ),
    )
    selected_predictions = [
        pairwise_select(
            prediction,
            pair_scores,
            score_threshold=selected["score_threshold"],
            nms_iou_threshold=selected["nms_iou_threshold"],
            distinct_threshold=selected["distinct_threshold"],
        )
        for prediction, pair_scores in zip(ranked, pair_probabilities)
    ]
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_pairwise_nms",
        "folds": sorted(folds),
        "model": "ExtraTreesClassifier",
        "pair_target": "keep both only when IoU>=0.5 assignments are distinct GT objects",
        "fold_diagnostics": fold_diagnostics,
        "candidate_count": len(candidates),
        "zero_error_candidate_count": sum(
            row["metrics"]["false_positive_count"] == 0
            and row["metrics"]["false_negative_count"] == 0
            for row in candidates
        ),
        "selected": selected,
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
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OOF pairwise duplicate-aware NMS")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--logits-cache", type=Path, required=True)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-iou-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--distinct-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
