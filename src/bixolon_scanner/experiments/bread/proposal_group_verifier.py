from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from ...evaluation.detector import _metrics, _xywh_to_xyxy, detection_error_rows
from ...pipeline.ports import Detection
from ...runtime.onnx import nms
from ...training.data import read_manifest
from .proposal_classifier_verifier import _load_predictions, _softmax, verifier_features
from .proposal_embedding_verifier import embedding_features
from .proposal_ranker import _intersection_matrix, _iou_matrix, select_ranked_predictions


def merged_group_labels(record: dict[str, Any], boxes: np.ndarray) -> np.ndarray:
    targets = np.asarray(
        [_xywh_to_xyxy(row["bbox_xywh"]) for row in record["annotations"]],
        dtype=np.float32,
    )
    if not len(targets):
        return np.zeros(len(boxes), dtype=np.int64)
    target_areas = np.prod(np.maximum(0.0, targets[:, 2:] - targets[:, :2]), axis=1)
    intersection = _intersection_matrix(boxes, targets)
    target_coverage = np.divide(
        intersection,
        target_areas[None, :],
        out=np.zeros_like(intersection),
        where=target_areas[None, :] > 0.0,
    )
    best_iou = _iou_matrix(boxes, targets).max(axis=1)
    return (((target_coverage >= 0.8).sum(axis=1) >= 2) & (best_iou < 0.5)).astype(np.int64)


def group_geometry_features(record: dict[str, Any], prediction: dict[str, Any]) -> np.ndarray:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    class_ids = np.asarray(prediction["class_ids"], dtype=np.int64)
    source_ids = np.asarray(prediction.get("source_ids", np.zeros(len(boxes))), dtype=np.int64)
    normalized = boxes / np.asarray(
        [record["width"], record["height"], record["width"], record["height"]],
        dtype=np.float32,
    )
    sizes = np.maximum(0.0, normalized[:, 2:] - normalized[:, :2])
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    normalized_areas = sizes[:, 0] * sizes[:, 1]
    aspects = np.divide(
        sizes[:, 0],
        sizes[:, 1],
        out=np.zeros(len(boxes), dtype=np.float32),
        where=sizes[:, 1] > 0.0,
    )
    intersection = _intersection_matrix(boxes, boxes)
    containment = np.divide(
        intersection,
        areas[None, :],
        out=np.zeros_like(intersection),
        where=areas[None, :] > 0.0,
    )
    np.fill_diagonal(containment, 0.0)
    smaller = areas[None, :] < areas[:, None] * 0.8
    contained = smaller & (containment >= 0.5)
    row_features = []
    for index in range(len(boxes)):
        inner_indices = np.flatnonzero(contained[index])
        inner_scores = scores[inner_indices]
        inner_sources = source_ids[inner_indices]
        inner_classes = class_ids[inner_indices]
        inner_detections = [
            Detection(*boxes[item], float(scores[item]), int(class_ids[item]))
            for item in inner_indices
        ]
        row_features.append(
            [
                float(containment[index].max()),
                *[
                    float((smaller[index] & (containment[index] >= threshold)).sum())
                    for threshold in (0.5, 0.7, 0.8, 0.9)
                ],
                float(inner_scores.max()) if len(inner_scores) else 0.0,
                float(inner_scores.mean()) if len(inner_scores) else 0.0,
                float((inner_scores > scores[index]).sum()),
                float((inner_sources != source_ids[index]).sum()),
                float(len(set(inner_sources.tolist()))),
                float(len(set(inner_classes.tolist()))),
                *[float(len(nms(inner_detections, threshold))) for threshold in (0.2, 0.3, 0.4)],
            ]
        )
    structure = np.asarray(row_features, dtype=np.float32)
    return np.column_stack(
        (
            scores,
            source_ids,
            normalized,
            sizes,
            normalized_areas,
            aspects,
            structure,
        )
    ).astype(np.float32)


def group_content_relation_features(prediction: dict[str, Any], logits: np.ndarray) -> np.ndarray:
    boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    if len(logits) != len(boxes):
        raise ValueError("group relation logits are not aligned with proposals")
    probabilities = _softmax(logits)
    top1 = np.argmax(probabilities, axis=1)
    top1_probability = probabilities[np.arange(len(boxes)), top1]
    entropy = -np.sum(probabilities * np.log(probabilities.clip(1e-12)), axis=1)
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    containment = np.divide(
        _intersection_matrix(boxes, boxes),
        areas[None, :],
        out=np.zeros((len(boxes), len(boxes)), dtype=np.float32),
        where=areas[None, :] > 0.0,
    )
    np.fill_diagonal(containment, 0.0)
    smaller = areas[None, :] < areas[:, None] * 0.8
    rows = []
    for index in range(len(boxes)):
        inner = np.flatnonzero(smaller[index] & (containment[index] >= 0.8))
        confident = inner[top1_probability[inner] >= 0.5]
        if len(inner):
            similarity = probabilities[inner] @ probabilities[index]
            weighted_classes = {}
            for item in inner:
                class_id = int(top1[item])
                weighted_classes[class_id] = max(
                    weighted_classes.get(class_id, 0.0),
                    float(scores[item] * top1_probability[item]),
                )
            class_support = sorted(weighted_classes.values(), reverse=True)
        else:
            similarity = np.asarray([], dtype=np.float32)
            class_support = []
        rows.append(
            [
                float(top1_probability[index]),
                float(entropy[index]),
                float(len(set(top1[inner].tolist()))),
                float((top1[inner] != top1[index]).sum()),
                float(len(set(top1[confident].tolist()))),
                float((top1[confident] != top1[index]).sum()),
                float(top1_probability[inner].max()) if len(inner) else 0.0,
                float(top1_probability[inner].mean()) if len(inner) else 0.0,
                float(similarity.max()) if len(similarity) else 0.0,
                float(similarity.min()) if len(similarity) else 0.0,
                float(similarity.mean()) if len(similarity) else 0.0,
                float(class_support[0]) if class_support else 0.0,
                float(class_support[1]) if len(class_support) > 1 else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def filter_group_predictions(
    ranked: list[dict[str, Any]],
    group_probabilities: list[np.ndarray],
    *,
    group_threshold: float,
) -> list[dict[str, Any]]:
    outputs = []
    for prediction, probabilities in zip(ranked, group_probabilities):
        if len(probabilities) != len(prediction["scores"]):
            raise ValueError("group probabilities are not aligned with proposals")
        kept = np.flatnonzero(probabilities < group_threshold)
        output = {
            **prediction,
            "boxes_xyxy": [prediction["boxes_xyxy"][index] for index in kept],
            "scores": [prediction["scores"][index] for index in kept],
            "class_ids": [prediction["class_ids"][index] for index in kept],
        }
        for field in ("source_ids", "top3_class_ids"):
            if field in prediction and len(prediction[field]) == len(probabilities):
                output[field] = [prediction[field][index] for index in kept]
        outputs.append(output)
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
    raw = _load_predictions(args.raw_predictions, records)
    ranked = _load_predictions(args.ranked_predictions, records)
    use_content_features = args.embedding_cache is not None or args.logits_cache is not None
    if use_content_features and (args.embedding_cache is None or args.logits_cache is None):
        raise ValueError("embedding and logits caches must be supplied together")
    if use_content_features:
        embedding_cache = np.load(args.embedding_cache)
        raw_embeddings = embedding_cache["raw_embeddings"].astype(np.float32)
        adapted_embeddings = embedding_cache["adapted_embeddings"].astype(np.float32)
        embedding_counts = embedding_cache["counts"]
        logits_cache = np.load(args.logits_cache)
        logits = logits_cache["logits"]
        ranking_logits = logits_cache["ranking_logits"]
        logits_counts = logits_cache["counts"]
        expected_counts = [len(row["scores"]) for row in raw]
        if (
            embedding_counts.tolist() != expected_counts
            or logits_counts.tolist() != expected_counts
        ):
            raise ValueError("group verifier caches do not match proposal counts")
    feature_parts = []
    label_parts = []
    fold_parts = []
    counts = []
    offset = 0
    for record, raw_row, ranked_row in zip(records, raw, ranked):
        if raw_row["boxes_xyxy"] != ranked_row["boxes_xyxy"]:
            raise ValueError("raw and ranked proposal boxes differ")
        boxes = np.asarray(raw_row["boxes_xyxy"], dtype=np.float32)
        geometry = group_geometry_features(record, raw_row)
        if use_content_features:
            end = offset + len(boxes)
            content = embedding_features(
                verifier_features(
                    record,
                    raw_row,
                    logits[offset:end],
                    ranking_logits[offset:end],
                ),
                raw_embeddings[offset:end],
                adapted_embeddings[offset:end],
            )
            relations = group_content_relation_features(
                raw_row,
                logits[offset:end],
            )
            feature_parts.append(np.column_stack((geometry, relations, content)))
            offset = end
        else:
            feature_parts.append(geometry)
        label_parts.append(merged_group_labels(record, boxes))
        fold_parts.append(np.full(len(boxes), int(record["fold"]), dtype=np.int64))
        counts.append(len(boxes))
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    candidate_folds = np.concatenate(fold_parts)
    probabilities = np.zeros(len(labels), dtype=np.float64)
    fold_diagnostics = []
    for held_out_fold in sorted(folds):
        training = candidate_folds != held_out_fold
        held_out = candidate_folds == held_out_fold
        model = ExtraTreesClassifier(
            n_estimators=args.estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed + held_out_fold,
        )
        model.fit(features[training], labels[training])
        probabilities[held_out] = model.predict_proba(features[held_out])[:, 1]
        fold_diagnostics.append(
            {
                "fold": held_out_fold,
                "training_candidate_count": int(training.sum()),
                "training_merged_group_count": int(labels[training].sum()),
            }
        )
    probability_rows = []
    offset = 0
    for count in counts:
        probability_rows.append(probabilities[offset : offset + count])
        offset += count

    candidates = []
    filtered_cache = {}
    for group_threshold in args.group_thresholds:
        filtered_cache[group_threshold] = filter_group_predictions(
            ranked,
            probability_rows,
            group_threshold=group_threshold,
        )
    for group_threshold, score_threshold, nms_threshold in product(
        args.group_thresholds, args.score_thresholds, args.nms_thresholds
    ):
        predictions = select_ranked_predictions(
            filtered_cache[group_threshold],
            score_threshold=score_threshold,
            nms_iou_threshold=nms_threshold,
        )
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
                "group_threshold": group_threshold,
                "score_threshold": score_threshold,
                "nms_iou_threshold": nms_threshold,
                "metrics": metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            -row["metrics"]["false_positive_count"] - row["metrics"]["false_negative_count"],
            row["metrics"]["exact_image_rate"],
            -row["metrics"]["false_negative_count"],
        ),
    )
    selected_predictions = select_ranked_predictions(
        filtered_cache[selected["group_threshold"]],
        score_threshold=selected["score_threshold"],
        nms_iou_threshold=selected["nms_iou_threshold"],
    )
    report = {
        "schema_version": "1.0",
        "evaluation": "bread_oof_merged_group_verifier",
        "folds": sorted(folds),
        "feature_count": int(features.shape[1]),
        "content_features": bool(use_content_features),
        "merged_group_count": int(labels.sum()),
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
    if args.filtered_ranked_output:
        args.filtered_ranked_output.parent.mkdir(parents=True, exist_ok=True)
        args.filtered_ranked_output.write_text(
            "".join(json.dumps(row) + "\n" for row in filtered_cache[selected["group_threshold"]]),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OOF merged-proposal verifier")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--logits-cache", type=Path)
    parser.add_argument("--group-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--nms-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--filtered-ranked-output", type=Path)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
